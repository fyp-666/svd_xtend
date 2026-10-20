"""Frame planning and recoverable, lossless storage for SVD continuation runs."""

from contextlib import closing
import hashlib
import json
import math
from pathlib import Path


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def make_plan(num_frames, segments=None, target_frames=None, mode="concatenate",
              overlap_frames=5, boundary="keep"):
    """Plan complete model calls, then optionally trim the final assembled video."""
    num_frames = _positive_integer(num_frames, "num_frames")
    if segments is not None and target_frames is not None:
        raise ValueError("segments and target_frames are mutually exclusive")
    if mode not in ("concatenate", "overlap"):
        raise ValueError("mode must be 'concatenate' or 'overlap'")
    if boundary not in ("keep", "drop-first"):
        raise ValueError("boundary must be 'keep' or 'drop-first'")
    if mode == "overlap":
        trim = _positive_integer(overlap_frames, "overlap_frames")
        if trim >= num_frames:
            raise ValueError("overlap_frames must be less than num_frames")
    else:
        trim = int(boundary == "drop-first")
        if trim >= num_frames:
            raise ValueError("drop-first requires at least two frames per segment")
    stride = num_frames - trim
    if target_frames is not None:
        target_frames = _positive_integer(target_frames, "target_frames")
        segments = 1 + max(0, (target_frames - num_frames + stride - 1) // stride)
    else:
        segments = 2 if segments is None else _positive_integer(segments, "segments")
    available = num_frames + (segments - 1) * stride
    return {"segments": segments, "segment_frames": num_frames, "trim_leading": trim,
            "output_frames": target_frames if target_frames is not None else available,
            "available_frames": available}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(directory):
    """A manifest is the commit marker; it never supplies paths to read."""
    path = Path(directory) / "segment.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Missing or invalid segment checkpoint: {path}") from error
    if not isinstance(data, dict) or data.get("format_version") != 1:
        raise ValueError(f"Unsupported segment checkpoint: {path}")
    for key in ("frame_count", "width", "height"):
        _positive_integer(data.get(key), key)
    if not isinstance(data.get("metadata"), dict):
        raise ValueError(f"Invalid segment metadata: {path}")
    files = data.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Missing segment file hashes: {path}")
    for filename in ("frames.npz", "state.safetensors"):
        entry = files.get(filename)
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Invalid checksum for {filename}: {path}")
    return data


def _checked_frames(directory, manifest):
    import numpy as np

    directory = Path(directory)
    for filename in ("frames.npz", "state.safetensors"):
        path = directory / filename
        if not path.is_file() or _sha256(path) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Missing or corrupted checkpoint file: {path}")
    try:
        with np.load(directory / "frames.npz", allow_pickle=False) as archive:
            if archive.files != ["frames"]:
                raise ValueError("Frame archive must contain only the 'frames' array")
            frames = archive["frames"]
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"Invalid lossless frame archive: {directory / 'frames.npz'}") from error
    expected = (manifest["frame_count"], manifest["height"], manifest["width"], 3)
    if frames.dtype != np.uint8 or frames.shape != expected:
        raise ValueError(f"Checkpoint frame dtype or dimensions disagree with segment.json: {directory}")
    return frames


def save_segment(directory, frames, tensors, metadata):
    """Commit one segment; an existing directory is never overwritten.

    Frames are stored as RGB uint8 with no pickle. Caller metadata lives under
    the manifest's ``metadata`` key. The final atomic manifest rename commits
    the checkpoint; an interrupted save has no valid commit marker.
    """
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    directory = Path(directory)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    json.dumps(metadata, allow_nan=False)
    if not isinstance(tensors, dict):
        raise TypeError("tensors must be a dictionary")
    arrays = []
    for frame in frames:
        if not isinstance(frame, Image.Image):
            raise TypeError("frames must contain PIL images")
        arrays.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Segment frames must be nonempty and have identical dimensions")
    packed = np.stack(arrays)
    cpu_tensors = {}
    for key, tensor in tensors.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("State must map string names to torch tensors")
        if tensor.layout != torch.strided:
            raise ValueError(f"State tensor must have strided layout: {key}")
        # Independent storage also allows callers to pass related tensor views.
        cpu_tensors[key] = tensor.detach().to("cpu").contiguous().clone()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "frames.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, frames=packed)
    (directory / "frames.npz.tmp").replace(directory / "frames.npz")
    save_file(cpu_tensors, str(directory / "state.safetensors.tmp"))
    (directory / "state.safetensors.tmp").replace(directory / "state.safetensors")
    manifest = {"format_version": 1, "frame_count": int(packed.shape[0]),
                "width": int(packed.shape[2]), "height": int(packed.shape[1]),
                "files": {name: {"sha256": _sha256(directory / name)}
                          for name in ("frames.npz", "state.safetensors")},
                "metadata": metadata}
    temporary = directory / "segment.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(directory / "segment.json")
    return manifest


def load_segment(directory):
    """Validate both checksums and return RGB PIL frames, CPU tensors, metadata."""
    from PIL import Image
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = _manifest(directory)
    arrays = _checked_frames(directory, manifest)
    try:
        tensors = load_file(str(directory / "state.safetensors"), device="cpu")
    except (OSError, ValueError, SafetensorError) as error:
        raise ValueError(f"Invalid safe tensor checkpoint: {directory / 'state.safetensors'}") from error
    return [Image.fromarray(array, mode="RGB") for array in arrays], tensors, manifest["metadata"]


def assemble_video(segment_dirs, destination, fps, trim_leading, target_frames=None,
                   recover_partial=False):
    """Stream one lossless segment at a time into a validated final MP4.

    The first segment is kept intact. Subsequent segments lose exactly
    ``trim_leading`` frames; ``target_frames`` optionally limits the final count.
    Only one segment and a small five-frame contact sheet are retained in RAM.
    Explicit recovery revalidates every checkpoint before removing only the
    four fixed derived export files, including interrupted FFmpeg outputs.
    """
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    trim_leading = _nonnegative_integer(trim_leading, "trim_leading")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number")
    directories = [Path(path) for path in segment_dirs]
    if not directories:
        raise ValueError("At least one segment is required")
    manifests = [_manifest(path) for path in directories]
    width, height = manifests[0]["width"], manifests[0]["height"]
    if width % 2 or height % 2:
        raise ValueError("H.264 yuv420p export requires even frame dimensions")
    available = 0
    for index, manifest in enumerate(manifests):
        if (manifest["width"], manifest["height"]) != (width, height):
            raise ValueError("All segments must have identical frame dimensions")
        trim = trim_leading if index else 0
        if trim >= manifest["frame_count"]:
            raise ValueError("trim_leading must leave at least one frame in every segment")
        available += manifest["frame_count"] - trim
    count_expected = available if target_frames is None else _positive_integer(target_frames, "target_frames")
    if count_expected > available:
        raise ValueError(f"Requested {count_expected} frames, but checkpoints supply only {available}")
    if not isinstance(recover_partial, bool):
        raise TypeError("recover_partial must be a boolean")
    if recover_partial:
        from safetensors import safe_open

        # Validate every supplied checkpoint before touching an earlier export,
        # including segments beyond an optional final frame limit. Opening the
        # safetensors header checks its layout without materializing the state.
        for directory, manifest in zip(directories, manifests):
            checked = _checked_frames(directory, manifest)
            del checked
            with safe_open(str(directory / "state.safetensors"), framework="pt", device="cpu") as state:
                state.keys()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    video_path, preview_path = destination / "video.mp4", destination / "preview.jpg"
    temporary_video = destination / ".video.partial.mp4"
    temporary_preview = destination / ".preview.partial.jpg"
    derived_paths = (video_path, preview_path, temporary_video, temporary_preview)
    # Resolve and inspect all targets before deleting any. Never follow a link,
    # delete a directory, or expand recovery beyond these four derived files.
    for path in derived_paths:
        if (path.is_symlink() or path.resolve().parent != destination
                or (path.exists() and not path.is_file())):
            raise ValueError(f"Unsafe derived export path; expected a regular file: {path}")
    if not recover_partial and any(path.exists() for path in derived_paths):
        raise FileExistsError(f"Final or partial export files already exist in {destination}")
    if recover_partial:
        for path in derived_paths:
            path.unlink(missing_ok=True)
    indices = sorted(set(np.linspace(0, count_expected - 1, min(5, count_expected), dtype=int).tolist()))
    thumb_width = 256
    thumb_height = max(1, round(height * thumb_width / width))
    sheet = Image.new("RGB", (thumb_width * len(indices), thumb_height + 24), "white")
    draw = ImageDraw.Draw(sheet)
    preview_columns = {index: column for column, index in enumerate(indices)}
    emitted = 0
    try:
        # imageio's own context manager does not close on exceptions. Closing
        # explicitly also waits for FFmpeg before partial files are removed.
        with closing(imageio.get_writer(str(temporary_video), fps=fps, codec="libx264", quality=8,
                                        macro_block_size=1, pixelformat="yuv420p")) as writer:
            for index, (directory, manifest) in enumerate(zip(directories, manifests)):
                arrays = _checked_frames(directory, manifest)
                start = trim_leading if index else 0
                for frame in arrays[start:]:
                    if emitted >= count_expected:
                        break
                    writer.append_data(frame)
                    if emitted in preview_columns:
                        column = preview_columns[emitted]
                        thumbnail = Image.fromarray(frame, mode="RGB").resize(
                            (thumb_width, thumb_height), Image.Resampling.LANCZOS)
                        sheet.paste(thumbnail, (column * thumb_width, 24))
                        draw.text((column * thumb_width + 6, 5), f"Frame {emitted}", fill="black")
                    emitted += 1
                del arrays
                if emitted >= count_expected:
                    break
        decoded_count = 0
        with closing(imageio.get_reader(str(temporary_video))) as reader:
            video_metadata = reader.get_meta_data()
            for decoded in reader:
                if decoded.shape[:2] != (height, width):
                    raise ValueError("Exported video dimensions differ from checkpoint frames")
                decoded_count += 1
        if (emitted != count_expected or decoded_count != count_expected
                or abs(video_metadata["fps"] - fps) > 0.01):
            raise ValueError("Exported video frame count or FPS differs from the plan")
        sheet.save(temporary_preview, quality=90)
        temporary_video.replace(video_path)
        temporary_preview.replace(preview_path)
    finally:
        # These fixed paths were created only for this assembly attempt.
        temporary_video.unlink(missing_ok=True)
        temporary_preview.unlink(missing_ok=True)
    return {"video": str(video_path), "preview": str(preview_path), "frame_count": decoded_count,
            "width": width, "height": height, "fps": video_metadata["fps"],
            "duration_seconds": decoded_count / fps, "available_frames": available,
            "trim_leading": trim_leading,
            "validation": "checkpoint hashes verified; all MP4 frames decoded; count, dimensions and FPS match"}
