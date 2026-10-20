"""Run one start/end image pair through the local interpolation checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from tools.inference_io import (
    create_run_directory, export_generated_frames, file_metadata,
    load_input_image, load_resource_config, resolve_resource_paths,
    resolve_unet_directory, utc_now, validate_base_model,
    validate_generated_frames, validate_settings, write_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "resources.local.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Local resource JSON (optional when CLI paths are given)")
    parser.add_argument("--base-model", help="Local SVD directory supplying VAE, image encoder, scheduler and processor")
    parser.add_argument("--checkpoint", help="Trained 9-channel checkpoint root or its unet directory")
    parser.add_argument("--first", type=Path, required=True, help="Start image")
    parser.add_argument("--last", type=Path, required=True, help="End image")
    parser.add_argument("--output-dir", help="Output parent; every attempt creates a unique run subdirectory")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:<index>, or cpu")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--vae-dtype", choices=("auto", "float32"), default="auto", help="Optional full-precision VAE encoding/decoding if FP16 output is unstable")
    parser.add_argument("--offload", choices=("model", "sequential", "none"), default="model")
    parser.add_argument("--forward-chunk-size", type=int, default=1, help="Feed-forward chunk size; 0 disables chunking")
    parser.add_argument("--decode-chunk-size", type=int, default=1, help="VAE frames decoded together; small values save VRAM")
    parser.add_argument("--width", type=int, default=512, help="Output width; positive multiple of 64")
    parser.add_argument("--height", type=int, default=320, help="Output height; positive multiple of 64")
    parser.add_argument("--num-frames", type=int, default=14, help="Total generated frames, including generated endpoints")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-bucket-id", type=int, default=127, help="Motion condition; default 127 matches the included training code")
    parser.add_argument("--conditioning-fps", type=int, default=8, help="Model condition; default 8 encodes effective FPS 7 because the pipeline subtracts one")
    parser.add_argument("--export-fps", type=float, default=6.0, help="Playback FPS; independent of model conditioning")
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--min-guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-guidance-scale", type=float, default=3.0)
    return parser


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface-hub", "numpy", "Pillow", "imageio", "imageio-ffmpeg"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = {key: value for key, value in vars(args).items() if key not in (
        "config", "base_model", "checkpoint", "first", "last", "output_dir",
    )}
    started = time.perf_counter()
    # Establish a writable error-log destination even if resource validation fails.
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else PROJECT_ROOT / "outputs"
    config_error = None
    config = {}
    try:
        config = load_resource_config(args.config, optional=args.config == DEFAULT_CONFIG)
        if args.output_dir is None and config.get("output_dir") is not None:
            path = Path(config["output_dir"]).expanduser()
            output_root = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    except Exception as exc:
        config_error = exc
    run_dir = create_run_directory(output_root)
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": 1, "status": "started", "stage": "validation", "started_at": utc_now(),
        "settings": settings, "resource_config": str(args.config.resolve()),
        "python": sys.version, "platform": platform.platform(), "packages": package_versions(),
        "timings_seconds": {}, "cuda_memory": None,
        "output_semantics": "All output frames are generated; input endpoints are not substituted.",
        "arguments": argv if argv is not None else sys.argv[1:],
    }
    write_manifest(manifest_path, manifest)
    print(f"Run directory: {run_dir}", flush=True)
    torch = None
    device = None
    exit_code = 1
    try:
        if config_error is not None:
            raise config_error
        validate_settings(settings)
        resources = resolve_resource_paths(config, {
            "base_model_path": args.base_model,
            "interpolation_checkpoint_path": args.checkpoint,
            "output_dir": args.output_dir,
        }, PROJECT_ROOT)
        validate_base_model(resources["base_model_path"])
        unet_dir = resolve_unet_directory(resources["interpolation_checkpoint_path"])
        manifest["resources"] = {key: str(value) for key, value in resources.items()}
        manifest["resources"]["resolved_unet_directory"] = str(unet_dir)
        manifest["checkpoint_files"] = [file_metadata(path) for path in sorted(unet_dir.iterdir()) if path.is_file()]
        first, first_metadata = load_input_image(args.first, args.width, args.height)
        last, last_metadata = load_input_image(args.last, args.width, args.height)
        manifest["inputs"] = {"first": first_metadata, "last": last_metadata}
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir()
        first.save(inputs_dir / "first.png")
        last.save(inputs_dir / "last.png")
        manifest["stage"] = "loading"
        write_manifest(manifest_path, manifest)

        # Heavy dependencies are deliberately absent from --help and IO validation.
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
        import torch as torch_module
        from diffusers.models.attention_processor import AttnProcessor2_0
        from models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
        from pipelines.pipeline_stable_video_diffusion_interp import StableVideoDiffusionInterpPipeline

        torch = torch_module
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this environment. Install the matching PyTorch CUDA build.")
            device = torch.device("cuda", device.index if device.index is not None else 0)
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            properties = torch.cuda.get_device_properties(device)
            manifest["gpu"] = {"name": properties.name, "total_bytes": properties.total_memory, "capability": list(torch.cuda.get_device_capability(device)), "torch_cuda_version": torch.version.cuda}
            if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("This CUDA device does not support bfloat16")
        dtype = getattr(torch, args.dtype)
        load_started = time.perf_counter()
        print("Loading local interpolation checkpoint and base components...", flush=True)
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            str(unet_dir), torch_dtype=dtype, low_cpu_mem_usage=True,
            custom_resume=True, local_files_only=True,
        )
        unet.set_attn_processor(AttnProcessor2_0())
        if args.forward_chunk_size:
            unet.enable_forward_chunking(chunk_size=args.forward_chunk_size, dim=0)
        pipe = StableVideoDiffusionInterpPipeline.from_pretrained(
            str(resources["base_model_path"]), unet=unet, torch_dtype=dtype,
            variant="fp16", local_files_only=True, low_cpu_mem_usage=True,
        )
        if args.vae_dtype == "float32":
            pipe.vae.to(dtype=torch.float32)
        if args.offload == "model":
            pipe.enable_model_cpu_offload(gpu_id=device.index or 0, device="cuda")
        elif args.offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=device.index or 0, device="cuda")
        else:
            pipe.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        manifest["timings_seconds"]["load"] = time.perf_counter() - load_started
        manifest["attention"] = "PyTorch SDPA / AttnProcessor2_0"
        manifest["stage"] = "inference"
        write_manifest(manifest_path, manifest)
        torch.manual_seed(args.seed)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        inference_started = time.perf_counter()
        print(f"Generating {args.num_frames} frames at {args.width}x{args.height}...", flush=True)
        with torch.inference_mode():
            frames = pipe(
                first, last, height=args.height, width=args.width, num_frames=args.num_frames,
                num_inference_steps=args.steps, min_guidance_scale=args.min_guidance_scale,
                max_guidance_scale=args.max_guidance_scale, fps=args.conditioning_fps,
                motion_bucket_id=args.motion_bucket_id, noise_aug_strength=args.noise_aug_strength,
                decode_chunk_size=args.decode_chunk_size, generator=generator, output_type="np",
            ).frames[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        manifest["timings_seconds"]["inference"] = time.perf_counter() - inference_started
        frames, validation = validate_generated_frames(frames, count=args.num_frames, width=args.width, height=args.height)
        manifest["frame_validation"] = validation
        manifest["stage"] = "export"
        manifest["outputs"] = {"frames": "frames/", "video": "video.mp4", "contact_sheet": "contact_sheet.jpg", "export_complete": False}
        write_manifest(manifest_path, manifest)
        export_started = time.perf_counter()
        manifest["outputs"] = export_generated_frames(frames, run_dir, fps=args.export_fps)
        manifest["timings_seconds"]["export"] = time.perf_counter() - export_started
        manifest["stage"] = "complete"
        manifest["status"] = "succeeded"
        exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        print(f"{manifest['status'].capitalize()} during {manifest['stage']}: {exc}", file=sys.stderr, flush=True)
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        manifest["finished_at"] = utc_now()
        manifest["timings_seconds"]["total"] = time.perf_counter() - started
        if torch is not None and device is not None and device.type == "cuda" and torch.cuda.is_initialized():
            try:
                manifest["cuda_memory"] = {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "scope": "this PyTorch process; excludes other applications and driver allocations",
                }
            except Exception as exc:
                manifest["cuda_memory_error"] = str(exc)
        write_manifest(manifest_path, manifest)
    print(f"Status: {manifest['status']}; manifest: {manifest_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
