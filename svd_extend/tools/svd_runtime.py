"""Local resources and reproducibility metadata shared by inference entry points."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def read_resources(value):
    path = resolve_path(value)
    if not path.is_file():
        if value == "configs/resources.local.json":
            return {}
        raise FileNotFoundError(f"Resource file not found: {path}")
    resources = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(resources, dict):
        raise ValueError("Resource JSON must contain an object")
    return resources


def collect_images(path):
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        images = sorted((p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
                        key=lambda p: p.relative_to(path).as_posix().lower())
        if images:
            return images
    raise ValueError(f"No supported images at {path}")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_model(root):
    """Check the local Diffusers layout without loading weight tensors."""
    configs = ("model_index.json", "unet/config.json", "vae/config.json", "image_encoder/config.json",
               "scheduler/scheduler_config.json", "feature_extractor/preprocessor_config.json")
    weights = ("unet/diffusion_pytorch_model.fp16.safetensors",
               "vae/diffusion_pytorch_model.fp16.safetensors", "image_encoder/model.fp16.safetensors")
    for relative in (*configs, *weights):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing model component: {path}. Use the model root, not its unet subfolder.")
    unet = json.loads((root / "unet/config.json").read_text(encoding="utf-8"))
    if unet.get("in_channels") != 8 or unet.get("out_channels") != 4:
        raise ValueError("Stage 1 requires the standard 8-input/4-output SVD U-Net, not an interpolation model")
    return {"path": str(root), "configured_num_frames": unet.get("num_frames"),
            "config_sha256": {rel: sha256_file(root / rel) for rel in configs},
            "weight_files": {rel: {"bytes": (root / rel).stat().st_size} for rel in weights},
            "verification": "component presence and config checks; weights are not content-hashed"}


def inspect_environment(device_name, dtype, offload):
    import torch
    import imageio_ffmpeg
    device = torch.device(device_name)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Only cuda:N and cpu are supported by this entry point")
    if device.type == "cpu" and (dtype != "float32" or offload != "none"):
        raise ValueError("CPU inference requires --dtype float32 --offload none")
    packages = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface-hub",
                "safetensors", "numpy", "Pillow", "imageio", "imageio-ffmpeg")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "packages": versions, "torch_cuda_version": torch.version.cuda,
              "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(), "device": str(device)}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment; check the PyTorch CUDA build and GPU driver")
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {device}")
        # A small real kernel catches unsupported GPU architectures before model loading.
        probe = torch.ones((2, 2), device=device, dtype=torch.float16)
        if (probe @ probe).sum().item() != 8:
            raise RuntimeError("CUDA arithmetic check failed")
        del probe
        result.update({"device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                       "gpu_capability": list(torch.cuda.get_device_capability(device)),
                       "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
                       "gpu_free_mib_before_model": torch.cuda.mem_get_info(device)[0] / 2**20})
    return result
