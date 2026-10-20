"""Input validation and result export for single-pair inference.

This module does not import torch, Diffusers, or the training data package.
Image/video dependencies are imported only by the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


RESOURCE_KEYS = ("base_model_path", "interpolation_checkpoint_path", "output_dir")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_resource_config(path: Path, *, optional: bool = False) -> dict[str, Any]:
    """Read local resources; a missing default config can be replaced by CLI args."""
    if optional and not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Resource config must contain a JSON object: {path}")
    for name in RESOURCE_KEYS:
        value = config.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"Resource config field {name!r} must be a non-empty path or null")
    return config


def resolve_resource_paths(
    config: Mapping[str, Any], overrides: Mapping[str, str | None], project_root: Path,
    *, working_directory: Path | None = None,
) -> dict[str, Path]:
    """Config-relative paths use project root; CLI-relative paths use the cwd."""
    working_directory = working_directory or Path.cwd()
    resolved = {}
    for key in RESOURCE_KEYS:
        override = overrides.get(key)
        value = override if override is not None else config.get(key)
        if value is None and key == "output_dir":
            value = "outputs"
        if value is None:
            flag = "--base-model" if key == "base_model_path" else "--checkpoint"
            raise ValueError(f"Missing {key}; set it in the resource config or pass {flag}")
        path = Path(value).expanduser()
        base = working_directory if override is not None else project_root
        resolved[key] = (path if path.is_absolute() else base / path).resolve()
    return resolved


def resolve_unet_directory(checkpoint: Path) -> Path:
    """Accept a checkpoint root or the U-Net directory, without duplicating /unet."""
    for candidate in (checkpoint, checkpoint / "unet"):
        config_path = candidate / "config.json"
        if not config_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("in_channels") != 9:
            raise ValueError(
                f"Expected a trained 9-channel interpolation U-Net, got "
                f"in_channels={config.get('in_channels')!r} in {config_path}"
            )
        names = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")
        if not any((candidate / name).is_file() for name in names):
            raise FileNotFoundError(f"No supported U-Net weight file in {candidate}")
        return candidate.resolve()
    raise FileNotFoundError(f"No U-Net config.json found in {checkpoint} or its unet subdirectory")


def validate_base_model(path: Path) -> None:
    """Check local component metadata without reading large tensor files."""
    required = (
        "model_index.json", "vae/config.json", "image_encoder/config.json",
        "scheduler/scheduler_config.json", "feature_extractor/preprocessor_config.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete base model at {path}; missing: {', '.join(missing)}")


def validate_settings(settings: Mapping[str, Any]) -> None:
    for name in ("width", "height"):
        if settings[name] <= 0 or settings[name] % 64:
            raise ValueError(f"{name} must be a positive multiple of 64 for the interpolation U-Net")
    if settings["num_frames"] < 3:
        raise ValueError("num_frames must be at least 3 (two endpoints and one intermediate frame)")
    for name in ("steps", "decode_chunk_size", "conditioning_fps"):
        if settings[name] < 1:
            raise ValueError(f"{name} must be at least 1")
    if settings["forward_chunk_size"] < 0:
        raise ValueError("forward_chunk_size must be 0 (disabled) or positive")
    for name in ("export_fps", "min_guidance_scale", "max_guidance_scale", "noise_aug_strength"):
        if not math.isfinite(settings[name]):
            raise ValueError(f"{name} must be finite")
    if settings["export_fps"] <= 0:
        raise ValueError("export_fps must be positive")
    if not 0 <= settings["min_guidance_scale"] <= settings["max_guidance_scale"]:
        raise ValueError("Guidance scales must satisfy 0 <= min <= max")
    if settings["noise_aug_strength"] < 0 or settings["motion_bucket_id"] < 0:
        raise ValueError("noise_aug_strength and motion_bucket_id must be non-negative")
    if not 0 <= settings["seed"] < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    device = settings["device"]
    if device != "cpu" and device != "cuda" and not (
        device.startswith("cuda:") and device[5:].isdigit()
    ):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if device == "cpu" and (settings["dtype"] != "float32" or settings["offload"] != "none"):
        raise ValueError("CPU inference requires --dtype float32 --offload none")


def create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"run_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir()
    return run_dir


def _json_safe_value(value: Any) -> Any:
    """Retain invalid numeric inputs as strings instead of non-standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically save strict JSON, including diagnostics for invalid float inputs.

    Non-finite numbers are recorded as NaN/Infinity/-Infinity strings. The
    in-memory values remain unchanged so settings validation still rejects them.
    """
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe_value(manifest), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def file_metadata(path: Path, *, sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    metadata = {"path": str(path.resolve()), "bytes": stat.st_size}
    if sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata["sha256"] = digest.hexdigest()
    return metadata


def load_input_image(path: Path, width: int, height: int):
    """Return an RGB input and explicit metadata about deterministic preprocessing."""
    from PIL import Image, ImageOps

    metadata = file_metadata(path, sha256=True)
    with Image.open(path) as source:
        metadata["source_size"] = list(source.size)
        metadata["source_mode"] = source.mode
        image = ImageOps.exif_transpose(source).convert("RGB")
        metadata["oriented_size"] = list(image.size)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    metadata["processed_size"] = [width, height]
    metadata["preprocessing"] = "EXIF transpose, RGB conversion, stretch resize with Lanczos"
    return image, metadata


def validate_generated_frames(frames, *, count: int, width: int, height: int):
    """Validate float output before quantization can hide NaN/Inf values."""
    import numpy as np

    array = np.asarray(frames)
    expected = (count, height, width, 3)
    if array.shape != expected:
        raise ValueError(f"Expected generated frames of shape {expected}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Expected floating-point pipeline output, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("Generated frames contain NaN or Inf")
    minimum, maximum = float(array.min()), float(array.max())
    if minimum < 0 or maximum > 1:
        raise ValueError(f"Expected frame values in [0, 1], got [{minimum}, {maximum}]")
    return array, {"shape": list(array.shape), "dtype": str(array.dtype), "min": minimum, "max": maximum}


def export_generated_frames(frames, run_dir: Path, *, fps: float) -> dict[str, Any]:
    """Save generated endpoints unchanged alongside intermediate frames and video."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    uint8_frames = (np.asarray(frames) * 255).round().astype(np.uint8)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()
    images = [Image.fromarray(frame) for frame in uint8_frames]
    for index, image in enumerate(images):
        image.save(frames_dir / f"{index:04d}.png")

    # Display only generated frames; provided endpoints are stored in inputs/.
    columns = min(4, len(images))
    tile_width = min(256, images[0].width)
    tile_height = round(images[0].height * tile_width / images[0].width)
    label_height = 24
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "#161616")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(images):
        left = (index % columns) * tile_width
        top = (index // columns) * (tile_height + label_height)
        sheet.paste(frame.resize((tile_width, tile_height), Image.Resampling.LANCZOS), (left, top))
        suffix = " (generated endpoint)" if index in (0, len(images) - 1) else ""
        draw.text((left + 4, top + tile_height + 5), f"{index:04d}{suffix}", fill="white")
    sheet.save(run_dir / "contact_sheet.jpg", quality=92)

    # macro_block_size=1 preserves the requested size; width/height are even.
    with imageio.get_writer(
        str(run_dir / "video.mp4"), format="FFMPEG", fps=fps,
        codec="libx264", pixelformat="yuv420p", macro_block_size=1,
    ) as writer:
        for frame in uint8_frames:
            writer.append_data(frame)
    # Decode the saved artifact rather than treating a successful writer call as
    # proof of a playable, correctly sized video.
    decoded_count = 0
    with imageio.get_reader(str(run_dir / "video.mp4"), format="FFMPEG") as reader:
        video_metadata = reader.get_meta_data()
        for decoded in reader:
            if decoded.shape != uint8_frames.shape[1:]:
                raise ValueError(f"MP4 changed frame dimensions: {decoded.shape} != {uint8_frames.shape[1:]}")
            decoded_count += 1
    if decoded_count != len(images):
        raise ValueError(f"MP4 contains {decoded_count} frames instead of {len(images)}")
    return {
        "frames": "frames/", "frame_count": len(images), "video": "video.mp4",
        "contact_sheet": "contact_sheet.jpg", "generated_endpoints_replaced": False,
        "png_conversion": "round(float32 [0,1] * 255) to uint8; no endpoint replacement",
        "video_codec": "libx264", "video_pixel_format": "yuv420p", "export_fps": fps,
        "export_complete": True, "mp4_decoded_frame_count": decoded_count,
        "mp4_reported_fps": video_metadata.get("fps"),
    }
