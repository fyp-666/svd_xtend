"""Prepare two attributed Big Buck Bunny endpoint images; never runs a model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

import imageio_ffmpeg
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp" / "demo_bunny_source"
OUTPUT = ROOT / "data" / "demo_bunny"
ARCHIVE_NAME = "BigBuckBunny_640x360.m4v.zip"
VIDEO_NAME = "BigBuckBunny_640x360.m4v"
VIDEO_URL = "https://download.blender.org/peach/bigbuckbunny_movies/" + ARCHIVE_NAME
ARCHIVE_SHA256 = "7118242b6728d40c871479c5b3c0f0fb27d748089df15d7f1b469f297c74a2d6"
VIDEO_SHA256 = "738e2f999860553d056dd79c952f58f63cbb73892a57c72342ce9e5330d9d2d7"
SAMPLES = (("first.png", 50.0), ("last.png", 50.5))
SIZE = (512, 320)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace different existing content: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the fixed official 121 MB archive if absent.")
    args = parser.parse_args()
    for directory in (CACHE, OUTPUT):
        if not directory.resolve().is_relative_to(ROOT):
            raise ValueError(f"Output directory must remain inside the project: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    archive = CACHE / ARCHIVE_NAME
    if not archive.exists():
        if not args.download:
            raise FileNotFoundError(f"Missing {archive}; rerun with --download.")
        partial = archive.with_suffix(archive.suffix + ".download")
        with urllib.request.urlopen(VIDEO_URL, timeout=60) as response, partial.open("xb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(partial) != ARCHIVE_SHA256:
            raise ValueError(f"Downloaded archive checksum differs; inspect {partial}.")
        partial.rename(archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("The cached official archive does not match its recorded SHA-256.")
    video = CACHE / VIDEO_NAME
    with zipfile.ZipFile(archive) as source:
        entry = source.getinfo(VIDEO_NAME)
        if not video.exists():
            # Extract only the known file, without accepting archive-provided paths.
            with source.open(entry) as stream, video.open("xb") as destination:
                shutil.copyfileobj(stream, destination)
        elif video.stat().st_size != entry.file_size:
            raise ValueError("The cached video has an unexpected size.")
    if sha256(video) != VIDEO_SHA256:
        raise ValueError("The cached video does not match its recorded SHA-256.")

    reader = imageio_ffmpeg.read_frames(str(video))
    metadata = next(reader)
    reader.close()
    source_size = tuple(metadata["source_size"])
    if source_size != (640, 359) or metadata["fps"] != 24.0:
        raise ValueError(f"Unexpected source metadata: {metadata}")
    frame_hashes = []
    for filename, seconds in SAMPLES:
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-ss", str(seconds), "-i", str(video), "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
        with Image.open(io.BytesIO(result.stdout)) as original:
            frame = ImageOps.fit(original.convert("RGB"), SIZE, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        payload = buffer.getvalue()
        write_new_or_identical(OUTPUT / filename, payload)
        frame_hashes.append((filename, hashlib.sha256(payload).hexdigest()))

    hashes = "\n".join(f"- `{name}`: `{digest}`" for name, digest in frame_hashes)
    source_note = f"""# Big Buck Bunny endpoint example

These two images are derived from *Big Buck Bunny* (2008), an open movie by the Blender Foundation.

- Official project and license statement: https://peach.blender.org/about/
- Official download page: https://peach.blender.org/download/
- Downloaded archive: {VIDEO_URL}
- License: Creative Commons Attribution 3.0 Unported (CC BY 3.0), https://creativecommons.org/licenses/by/3.0/
- Required film-excerpt credit: (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org
- Changes: video-frame extraction, RGB conversion, a centered crop, and Lanczos resizing; no audio is included.

## Sampling and preprocessing

| Image | Source timestamp | Decoded source | Saved image |
| --- | --- | --- | --- |
| first.png | 00:50.000 | 640 x 359 | 512 x 320 RGB PNG |
| last.png | 00:50.500 | 640 x 359 | 512 x 320 RGB PNG |

The official filename says 640x360; this file actually decodes as 640 x 359 pixels at 24 fps. FFmpeg accurate seeking selects the frame at each requested time. The timestamps are 0.5 seconds apart (12 source-frame intervals). Both show the same shot while the bunny stretches its arms.

Pillow `ImageOps.fit` uses a centered 574.4 x 359 crop (source box approximately x=32.8..607.2, y=0..359) and resizes it to 512 x 320 with Lanczos. This preserves geometry and removes narrow strips from the left and right. The original video remains in the ignored `tmp/demo_bunny_source/` cache.

Only the two input endpoints are included. No ground-truth intermediate frames are included, and this pair is a qualitative demonstration, not a benchmark. Model-generated frames are not frames from the original film. A generated sequence length, conditioning FPS, and export FPS must be recorded independently of this source interval.

## Reproduce

From the project root, in the dedicated Conda environment:

```powershell
conda run -n svd_interp python -m tools.prepare_demo --download
```

The script uses the bundled executable from imageio-ffmpeg and Pillow. It reuses the verified archive, writes only its fixed project paths, and refuses to replace a differing existing endpoint. An existing source note is preserved. Decoder or Pillow versions can affect PNG bytes.

Tool versions and hashes for the latest successful extraction are recorded locally in the ignored `tmp/demo_bunny_source/reproduction.json`.

Archive SHA-256: `{ARCHIVE_SHA256}`

Video SHA-256: `{VIDEO_SHA256}`

Endpoint SHA-256:

{hashes}
"""
    source_path = OUTPUT / "SOURCE.md"
    if not source_path.exists():
        write_new_or_identical(source_path, source_note.encode("utf-8"))
    reproduction = {
        "source_url": VIDEO_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "video_sha256": VIDEO_SHA256,
        "source_size": list(source_size),
        "source_fps": metadata["fps"],
        "output_size": list(SIZE),
        "sample_seconds": dict(SAMPLES),
        "endpoint_sha256": dict(frame_hashes),
        "tools": {
            "imageio-ffmpeg": imageio_ffmpeg.__version__,
            "Pillow": PILLOW_VERSION,
            "FFmpeg": metadata["ffmpeg_version"],
        },
    }
    with (CACHE / "reproduction.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(reproduction, stream, indent=2)
        stream.write("\n")
    print(f"Prepared {OUTPUT / 'first.png'} and {OUTPUT / 'last.png'}")
    print(f"Source: {source_size}, {metadata['fps']} fps; saved: {SIZE}")


if __name__ == "__main__":
    main()
