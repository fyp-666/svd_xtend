"""Configurable SVD continuation: concatenate, latent overlap and initial-image anchoring."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

from infer_svd import build_parser, validate_args
from tools.svd_continuation import make_plan, save_segment, load_segment, assemble_video
from tools.svd_runtime import (
    PROJECT_ROOT, inspect_environment, inspect_model, read_resources,
    resolve_path, sha256_file, write_json,
)


SOURCE_FILES = (
    "infer_svd_extend.py", "infer_svd.py", "tools/svd_runtime.py",
    "tools/svd_continuation.py", "tools/svd_overlap.py",
    "pipelines/pipeline_stable_video_diffusion.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def get_args(argv=None):
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(__doc__, input_help="One starting image for this continuation run; relative to the project root.")
    # Native resolution is explicit here; small dimensions are an opt-in engineering test.
    parser.set_defaults(width=1024, height=576)
    parser.add_argument("--mode", choices=("concatenate", "overlap"), default="concatenate")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--segments", type=int, help="Number of full model calls; defaults to 2.")
    length.add_argument("--target-frames", type=int, help="Exact final length; crop the final assembled segment.")
    parser.add_argument("--conditioning", choices=("previous", "initial"), default="previous",
                        help="CLIP source: current segment input or the original image. VAE always uses the segment input.")
    parser.add_argument("--overlap-frames", type=int, default=5, help="Prefix/tail overlap k; used only in overlap mode.")
    parser.add_argument("--boundary", choices=("keep", "drop-first"), default="keep",
                        help="Concatenate only: keep all frames or explicitly omit each later segment's first frame.")
    parser.add_argument("--resume", help="Existing run directory. Uses its saved parameters; only --check-only may accompany it.")
    args = parser.parse_args(tokens)
    if args.resume:
        allowed = []
        skip = False
        for token in tokens:
            if skip:
                skip = False
            elif token == "--resume":
                skip = True
            elif token.startswith("--resume=") or token == "--check-only":
                continue
            else:
                allowed.append(token)
        if allowed:
            parser.error("--resume loads saved parameters; do not supply overrides: " + " ".join(allowed))
        manifest = resolve_path(args.resume) / "run.json"
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("format_version") != 1 or record.get("entry_point") != "infer_svd_extend.py":
            parser.error("--resume requires a continuation run created by this entry point")
        values = dict(record["parameters"])
        values.update(resume=str(manifest.parent), check_only=args.check_only)
        args = argparse.Namespace(**values)
    validate_args(args, parser)
    if args.mode == "overlap" and args.boundary != "keep":
        parser.error("overlap removes exactly k frames; --boundary applies only to concatenate")
    try:
        make_plan(args.num_frames, args.segments, args.target_frames,
                  args.mode, args.overlap_frames, args.boundary)
    except ValueError as error:
        parser.error(str(error))
    return args


@contextmanager
def run_lock(run_dir):
    """An OS lock survives stale lock files but is released when the process exits."""
    stream = (run_dir / ".run.lock").open("a+b")
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError(f"Another process is using this run: {run_dir}") from error
    try:
        yield
    finally:
        stream.seek(0)
        if sys.platform == "win32":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def discard_pending(path):
    """Remove only known, uncommitted checkpoint files, never a completed segment."""
    if not path.exists():
        return
    if path.is_symlink() or path.resolve().parent != path.parent.resolve() or not path.name.endswith(".pending"):
        raise ValueError(f"Unsafe pending checkpoint path: {path}")
    allowed = {"frames.npz", "state.safetensors", "segment.json"}
    allowed |= {name + ".tmp" for name in allowed}
    children = list(path.iterdir())
    if any(not p.is_file() or p.is_symlink() or p.name not in allowed for p in children):
        raise ValueError(f"Unexpected files in incomplete checkpoint; inspect manually: {path}")
    for child in children:
        child.unlink()
    path.rmdir()


def load_pipeline(args, model_path):
    import torch
    from pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(model_path), torch_dtype=getattr(torch, args.dtype), variant="fp16",
        local_files_only=True, use_safetensors=True, low_cpu_mem_usage=True,
    )
    if args.forward_chunking:
        pipe.unet.enable_forward_chunking()
    device = torch.device(args.device)
    if args.offload == "model":
        pipe.enable_model_cpu_offload(gpu_id=device.index or 0)
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload(gpu_id=device.index or 0)
    else:
        pipe.to(device)
    return pipe


def checkpoint_progress(run_dir, plan, fingerprint):
    root = run_dir / "segments"
    directories = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith(".pending")) if root.exists() else []
    previous_frames, tensors, metadata = None, {}, {}
    for index, path in enumerate(directories):
        if path.name != f"{index:04d}" or index >= plan["segments"] or path.is_symlink():
            raise ValueError(f"Non-contiguous or unexpected checkpoint: {path}")
        previous_frames, tensors, metadata = load_segment(path)
        if metadata.get("index") != index or metadata.get("fingerprint") != fingerprint:
            raise ValueError(f"Checkpoint belongs to a different configuration: {path}")
        if len(previous_frames) != plan["segment_frames"]:
            raise ValueError(f"Checkpoint frame count differs from the model call: {path}")
    return directories, previous_frames, tensors, metadata


def main(argv=None, *, pipeline_factory=None):
    args = get_args(argv)
    resources = read_resources(args.resources)
    model_value = args.model_path or resources.get("base_model_path")
    if not model_value:
        raise ValueError("Set --model-path or base_model_path in the local resources file")
    model_path = resolve_path(model_value)
    input_path = resolve_path(args.input)
    if not input_path.is_file():
        raise ValueError("Continuation takes one input image per run")
    from PIL import Image, ImageOps
    with Image.open(input_path) as source:
        first_image = ImageOps.exif_transpose(source).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.LANCZOS)
    model = inspect_model(model_path)
    environment = inspect_environment(args.device, args.dtype, args.offload)
    # Validate pipeline imports during preflight as well, without constructing a model.
    from pipelines.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
    args.device = environment["device"]
    plan = make_plan(args.num_frames, args.segments, args.target_frames,
                     args.mode, args.overlap_frames, args.boundary)
    parameters = {k: v for k, v in vars(args).items() if k not in ("resume", "check_only")}
    identity = {
        "parameters": parameters, "model": model, "input_sha256": sha256_file(input_path),
        "weight_mtime_ns": {name: (model_path / name).stat().st_mtime_ns for name in model["weight_files"]},
        "source_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES},
        "packages": environment["packages"],
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    record = {
        "format_version": 1, "entry_point": "infer_svd_extend.py",
        "stage": "02_concatenate" if args.mode == "concatenate" else "03_overlap",
        "parameters": parameters, "plan": plan, "identity": identity, "fingerprint": fingerprint,
        "environment": environment, "input": str(input_path), "model_path": str(model_path),
        "seed_policy": "segment i uses (base_seed + i) modulo 2**63, with i starting at zero",
        "conditioning_policy": "VAE uses previous final frame (concatenate) or previous frame -k (overlap); initial mode retains only the first CLIP embedding",
        "attempts": [], "completed_segments": 0,
    }
    if args.resume:
        run_dir = resolve_path(args.resume)
        saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint or saved.get("plan") != plan:
            raise ValueError("Resume rejected: input, model metadata, source code, environment packages or parameters changed")
        record = saved
    else:
        output_root = resolve_path(args.output_dir or resources.get("output_dir") or "outputs")
        name = args.run_name or datetime.now(timezone.utc).strftime("extend_%Y%m%dT%H%M%S_%fZ")
        run_dir = output_root / name
    print(json.dumps({"parameters": parameters, "plan": plan, "run_dir": str(run_dir),
                      "model_path": str(model_path), "fingerprint": fingerprint}, ensure_ascii=False, indent=2))
    if args.check_only:
        if args.resume:
            directories, _, _, _ = checkpoint_progress(run_dir, plan, fingerprint)
            print(f"Validated {len(directories)} saved checkpoints.")
        print("Preflight passed. No model loaded and no files written.")
        return 0
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        record.update(run_dir=str(run_dir), started_at_utc=utc_now(), status="created")
        first_image.save(run_dir / "input.png")
        write_json(run_dir / "run.json", record)

    import torch
    from tools.svd_overlap import LatentTrajectory
    with run_lock(run_dir):
        # Another attempt may have finished between preflight and acquiring the lock.
        record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if record.get("fingerprint") != fingerprint or record.get("plan") != plan:
            raise ValueError("Run configuration changed while waiting for its lock")
        directories, previous_frames, state, metadata = checkpoint_progress(run_dir, plan, fingerprint)
        if record.get("status") == "completed" and len(directories) == plan["segments"]:
            video = run_dir / "video.mp4"
            if video.exists() and sha256_file(video) == record.get("output_sha256"):
                print(f"Already complete: {video}")
                return 0
            if video.exists():
                raise ValueError("Completed output was modified; refusing to overwrite it")
        record["completed_segments"] = len(directories)
        attempt = {"started_at_utc": utc_now(), "resume_from_segment": len(directories),
                   "previous_status": record.get("status"), "status": "running"}
        record["attempts"].append(attempt)
        record["status"] = "running"
        record.pop("error", None)
        write_json(run_dir / "run.json", record)
        started = time.perf_counter()
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            anchor = state.get("initial_embedding")
            trajectory = None
            if directories and args.conditioning == "initial" and anchor is None:
                raise ValueError("Checkpoint is missing the original CLIP embedding")
            if directories and args.mode == "overlap":
                trajectory = LatentTrajectory(metadata=metadata["trajectory_metadata"], latents=state["trajectory"])
            pipe = (pipeline_factory or load_pipeline)(args, model_path) if len(directories) < plan["segments"] else None
            for index in range(len(directories), plan["segments"]):
                seed = (args.seed + index) % 2**63
                image = first_image if index == 0 else previous_frames[-args.overlap_frames if args.mode == "overlap" else -1]
                segment_started = time.perf_counter()
                print(f"Segment {index + 1}/{plan['segments']}: {args.mode}, seed={seed}", flush=True)
                with torch.inference_mode():
                    out = pipe(
                        image, width=args.width, height=args.height, num_frames=args.num_frames,
                        num_inference_steps=args.steps, decode_chunk_size=args.decode_chunk_size,
                        fps=args.conditioning_fps, motion_bucket_id=args.motion_bucket_id,
                        noise_aug_strength=args.noise_aug_strength,
                        min_guidance_scale=args.min_guidance_scale, max_guidance_scale=args.max_guidance_scale,
                        generator=torch.Generator(device="cpu").manual_seed(seed),
                        image_embeddings=anchor if args.conditioning == "initial" else None,
                        overlap_frames=args.overlap_frames if args.mode == "overlap" else 0,
                        reference_trajectory=trajectory, return_trajectory=args.mode == "overlap",
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                frames = out.frames[0]
                if len(frames) != args.num_frames or any(f.size != (args.width, args.height) for f in frames):
                    raise ValueError("Pipeline frame count or dimensions differ from the requested clip")
                tensors = {}
                if args.conditioning == "initial":
                    if anchor is None:
                        anchor = out.image_embeddings.detach().to("cpu").clone().contiguous()
                    tensors["initial_embedding"] = anchor
                trajectory = out.latent_trajectory if args.mode == "overlap" else None
                if args.mode == "overlap" and trajectory is None:
                    raise ValueError("Overlap pipeline did not return a trajectory")
                segment_metadata = {
                    "index": index, "fingerprint": fingerprint, "seed": seed,
                    "generation_seconds": time.perf_counter() - segment_started,
                    "trajectory_metadata": trajectory.metadata if trajectory is not None else None,
                }
                if trajectory is not None:
                    tensors["trajectory"] = trajectory.latents
                segment_root = run_dir / "segments"
                segment_root.mkdir(exist_ok=True)
                pending = segment_root / f"{index:04d}.pending"
                discard_pending(pending)
                save_segment(pending, frames, tensors, segment_metadata)
                destination = segment_root / f"{index:04d}"
                pending.rename(destination)
                directories.append(destination)
                previous_frames = frames
                record["completed_segments"] = index + 1
                record["updated_at_utc"] = utc_now()
                write_json(run_dir / "run.json", record)
                print(f"Checkpoint saved: {destination}", flush=True)
                del out
            record["output"] = assemble_video(directories, run_dir, args.export_fps,
                                               plan["trim_leading"], plan["output_frames"],
                                               recover_partial=bool(args.resume))
            record["output_sha256"] = sha256_file(run_dir / "video.mp4")
            record["status"] = attempt["status"] = "completed"
        except KeyboardInterrupt:
            record["status"] = attempt["status"] = "interrupted"
            raise
        except Exception as error:
            record["status"] = attempt["status"] = "failed"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            attempt["elapsed_seconds"] = time.perf_counter() - started
            attempt["finished_at_utc"] = utc_now()
            if device.type == "cuda":
                attempt["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                attempt["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
            record["updated_at_utc"] = utc_now()
            write_json(run_dir / "run.json", record)
    print(f"Saved {run_dir / 'video.mp4'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
