"""Stage 1: reproducible single-clip SVD inference from local weights."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from tools.svd_runtime import (
    PROJECT_ROOT, collect_images, inspect_environment, inspect_model,
    read_resources, resolve_path, sha256_file, write_json,
)


def build_parser(description=__doc__, *, input_help="One image or a directory searched recursively; relative to the project root."):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--resources", default="configs/resources.local.json",
                        help="Resource JSON, relative to the project root.")
    parser.add_argument("--model-path", "--model_path", help="Local Diffusers model root; overrides resource JSON.")
    parser.add_argument("--input", "--input_dir", default="data/demo.jpg",
                        help=input_help)
    parser.add_argument("--output-dir", "--output_dir", help="Output root; overrides resource JSON.")
    parser.add_argument("--run-name", help="New output folder name; existing runs are never overwritten.")
    parser.add_argument("--device", default="cuda:0", help="cuda:N or cpu (CPU requires float32 and no offload).")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--offload", choices=["model", "sequential", "none"], default="model")
    parser.add_argument("--forward-chunking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--width", type=int, default=512, help="Output width in pixels (default: %(default)s).")
    parser.add_argument("--height", type=int, default=320, help="Output height in pixels (default: %(default)s).")
    parser.add_argument("--num-frames", "--num_frames", type=int, default=25)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--decode-chunk-size", type=int, default=2)
    parser.add_argument("--conditioning-fps", type=int, default=6,
                        help="SVD conditioning input; pipeline internally subtracts one.")
    parser.add_argument("--export-fps", type=int, default=6, help="Playback FPS in the exported MP4.")
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--min-guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true", help="Check inputs, weights and CUDA without loading models.")
    return parser


def validate_args(args, parser):
    for key in ("width", "height", "num_frames", "steps", "decode_chunk_size", "conditioning_fps", "export_fps"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.width % 8 or args.height % 8:
        parser.error("width and height must be divisible by 8")
    if not 0 <= args.noise_aug_strength < float("inf"):
        parser.error("noise augmentation must be finite and nonnegative")
    if not 0 <= args.min_guidance_scale <= args.max_guidance_scale < float("inf"):
        parser.error("guidance scales must be finite and 0 <= min <= max")
    if not 0 <= args.motion_bucket_id <= 255:
        parser.error("motion bucket ID must be between 0 and 255")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        parser.error("run name must be a single folder name")
    return args


def get_args(argv=None):
    parser = build_parser()
    return validate_args(parser.parse_args(argv), parser)


def export_clip(frames, destination, fps):
    """Export and decode the result once to check the actual container contents."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Pipeline returned empty or inconsistent frames")
    if all(int(a.max()) == int(a.min()) for a in arrays):
        raise ValueError("All generated frames are uniform; inspect numerical stability before accepting this run")
    video_path = destination / "video.mp4"
    with imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p") as writer:
        for frame in arrays:
            writer.append_data(frame)
    with imageio.get_reader(str(video_path)) as reader:
        metadata = reader.get_meta_data()
        count = 0
        for decoded in reader:
            if decoded.shape[:2] != arrays[0].shape[:2]:
                raise ValueError("Exported video dimensions differ from generated frames")
            count += 1
    if count != len(arrays) or abs(metadata["fps"] - fps) > 0.01:
        raise ValueError("Exported video frame count or FPS differs from the request")
    indices = sorted(set(np.linspace(0, count - 1, min(5, count), dtype=int).tolist()))
    thumb_width = 256
    thumb_height = round(arrays[0].shape[0] * thumb_width / arrays[0].shape[1])
    sheet = Image.new("RGB", (thumb_width * len(indices), thumb_height + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for column, index in enumerate(indices):
        sheet.paste(frames[index].resize((thumb_width, thumb_height)), (column * thumb_width, 24))
        draw.text((column * thumb_width + 6, 5), f"Frame {index}", fill="black")
    sheet.save(destination / "preview.jpg", quality=90)
    differences = [float(np.abs(b.astype(np.float32) - a.astype(np.float32)).mean())
                   for a, b in zip(arrays, arrays[1:])]
    return {"video": str(video_path), "frame_count": count, "width": arrays[0].shape[1],
            "height": arrays[0].shape[0], "fps": metadata["fps"], "duration_seconds": count / fps,
            "mean_adjacent_pixel_difference_0_255": float(np.mean(differences)) if differences else None,
            "validation": "decoded all MP4 frames; count, dimensions and FPS match"}


def main(argv=None):
    args = get_args(argv)
    resources = read_resources(args.resources)
    model_value = args.model_path or resources.get("base_model_path")
    if not model_value:
        raise ValueError("Set --model-path or base_model_path in configs/resources.local.json")
    model_path = resolve_path(model_value)
    inputs = collect_images(resolve_path(args.input))
    model = inspect_model(model_path)
    environment = inspect_environment(args.device, args.dtype, args.offload)
    from PIL import Image, ImageOps
    for path in inputs:
        with Image.open(path) as source:
            source.verify()
    # Import validation belongs to preflight, without constructing or loading a model.
    from pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    report = {"stage": "01_base_generation", "model": model, "environment": environment,
              "inputs": [str(path) for path in inputs], "parameters": vars(args)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_only:
        print("Preflight passed. No model was loaded and no output directory was created.")
        return 0

    import torch
    output_root = resolve_path(args.output_dir or resources.get("output_dir") or "outputs")
    name = args.run_name or datetime.now(timezone.utc).strftime("stage1_%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    report.update({"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                   "run_dir": str(run_dir), "samples": [], "source_sha256": {
                       rel: sha256_file(PROJECT_ROOT / rel) for rel in (
                           "infer_svd.py", "tools/svd_runtime.py", "pipelines/pipeline_stable_video_diffusion.py")}})
    write_json(run_dir / "run.json", report)
    started = time.perf_counter()
    device = torch.device(environment["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        load_start = time.perf_counter()
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            str(model_path), torch_dtype=getattr(torch, args.dtype), variant="fp16",
            local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True,
        )
        if args.forward_chunking:
            pipe.unet.enable_forward_chunking()
        if args.offload == "model":
            pipe.enable_model_cpu_offload(gpu_id=device.index or 0)
        elif args.offload == "sequential":
            pipe.enable_sequential_cpu_offload(gpu_id=device.index or 0)
        else:
            pipe.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report["model_load_seconds"] = time.perf_counter() - load_start
        report["attention_processors"] = sorted({type(p).__name__ for p in pipe.unet.attn_processors.values()})
        for index, path in enumerate(inputs):
            destination = run_dir / f"{index:03d}_{path.stem}"
            destination.mkdir()
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").resize(
                    (args.width, args.height), Image.Resampling.LANCZOS)
            image.save(destination / "input.png")
            # CPU RNG also supports the noise added to the CPU-preprocessed image.
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            generation_start = time.perf_counter()
            with torch.inference_mode():
                out = pipe(image, width=args.width, height=args.height, num_frames=args.num_frames,
                           num_inference_steps=args.steps, decode_chunk_size=args.decode_chunk_size,
                           fps=args.conditioning_fps, motion_bucket_id=args.motion_bucket_id,
                           noise_aug_strength=args.noise_aug_strength, generator=generator,
                           min_guidance_scale=args.min_guidance_scale, max_guidance_scale=args.max_guidance_scale)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            generation_seconds = time.perf_counter() - generation_start
            if len(out.frames[0]) != args.num_frames:
                raise ValueError("Pipeline output frame count does not match --num-frames")
            sample = export_clip(out.frames[0], destination, args.export_fps)
            sample.update({"input_path": str(path), "input_sha256": sha256_file(path),
                           "generation_seconds": generation_seconds, "seed": args.seed,
                           "image_embedding_shape": list(out.image_embeddings.shape)})
            report["samples"].append(sample)
            write_json(run_dir / "run.json", report)
            print(f"Saved {sample['video']} ({generation_seconds:.1f}s generation)")
        report["status"] = "completed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if device.type == "cuda":
            report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            report["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        write_json(run_dir / "run.json", report)
    print(f"Run record: {run_dir / 'run.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
