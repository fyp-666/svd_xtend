"""CPU storage and compatibility checks for step-aligned SVD overlap."""

from dataclasses import dataclass
import math
from collections.abc import Mapping

import numpy as np
import torch


@dataclass
class LatentTrajectory:
    """Tail states at every sigma, shaped [steps + 1, batch, overlap, C, H, W].

    The tensor is stored on CPU. State 0 is the initial sigma-scaled noise;
    state i + 1 is the corrected state after scheduler step i. The final state
    is therefore the one used for decoding. Metadata is JSON-compatible so a
    caller can save it separately from the tensor without pickling this class.
    """

    metadata: dict
    latents: torch.Tensor


def _plain(value):
    """Copy scheduler configuration into stable, JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_plain(item) for item in value)
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported or non-finite trajectory configuration value: {value!r}")


def trajectory_metadata(scheduler, latents, overlap_frames, *, height, width,
                        num_videos_per_prompt, do_classifier_free_guidance,
                        min_guidance_scale, max_guidance_scale, guidance_scale,
                        fps, motion_bucket_id, noise_aug_strength):
    """Describe the complete sampling contract; input images may differ by segment."""
    timesteps = scheduler.timesteps
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) != len(timesteps) + 1:
        raise ValueError("Overlap requires one scheduler sigma per latent state (steps + 1).")
    if scheduler.order != 1:
        raise ValueError("Overlap currently supports first-order schedulers only.")
    if latents.ndim != 5 or latents.shape[2] != 4:
        raise ValueError("Overlap requires SVD latents shaped [batch, frames, 4, height, width].")
    return _plain({
        "format_version": 1,
        "scheduler": {
            "class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
            "config": dict(scheduler.config),
            "order": scheduler.order,
            "timesteps": timesteps,
            "sigmas": sigmas,
        },
        "latent_shape": list(latents.shape),
        "latent_dtype": str(latents.dtype),
        "trajectory_shape": [len(timesteps) + 1, latents.shape[0], overlap_frames, *latents.shape[2:]],
        "height": height,
        "width": width,
        "overlap_frames": overlap_frames,
        "num_videos_per_prompt": num_videos_per_prompt,
        "classifier_free_guidance": do_classifier_free_guidance,
        "min_guidance_scale": min_guidance_scale,
        "max_guidance_scale": max_guidance_scale,
        "guidance_scale": guidance_scale,
        "conditioning_fps": fps,
        "motion_bucket_id": motion_bucket_id,
        "noise_aug_strength": noise_aug_strength,
    })


def validate_trajectory(trajectory, expected_metadata):
    """Reject incompatible or damaged caches before a denoising step is run."""
    if not isinstance(trajectory, LatentTrajectory):
        raise TypeError("reference_trajectory must be a LatentTrajectory.")
    if not isinstance(trajectory.metadata, dict):
        raise ValueError("Trajectory metadata must be a dictionary.")
    actual_metadata = _plain(trajectory.metadata)
    if actual_metadata != expected_metadata:
        keys = sorted(set(actual_metadata) | set(expected_metadata))
        changed = [key for key in keys if actual_metadata.get(key) != expected_metadata.get(key)]
        raise ValueError(f"Incompatible overlap trajectory metadata: {', '.join(changed)}.")
    states = trajectory.latents
    if not isinstance(states, torch.Tensor) or states.device.type != "cpu":
        raise ValueError("Trajectory states must be a CPU tensor.")
    if list(states.shape) != expected_metadata["trajectory_shape"]:
        raise ValueError("Trajectory tensor shape does not match its sampling contract.")
    if str(states.dtype) != expected_metadata["latent_dtype"]:
        raise ValueError("Trajectory tensor dtype does not match the current latents.")
    if not torch.isfinite(states).all().item():
        raise ValueError("Trajectory contains non-finite latent values.")


def copy_reference_state(latents, trajectory, state_index, overlap_frames):
    """Copy one CPU state into the current prefix without aliasing the reference."""
    reference = trajectory.latents[state_index].to(device=latents.device, dtype=latents.dtype)
    latents[:, :overlap_frames].copy_(reference)


def tail_snapshot(latents, overlap_frames):
    """Store only the corrected tail, independently of future in-place updates."""
    return latents[:, -overlap_frames:].detach().to(device="cpu").clone()
