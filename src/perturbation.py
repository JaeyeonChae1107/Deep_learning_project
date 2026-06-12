"""
Perturbation transforms for beef grading reliability analysis.

Training (RandomPerturbation):
  - Randomly selects one of 5 perturbation types per sample
  - Randomly samples the strength from the configured range
  - Applied on-the-fly; no pre-stored augmented images

Evaluation (FixedPerturbation):
  - Applies a single perturbation type at a fixed strength
  - Used for fair model comparison across all 3 models

NOTE: All perturbation functions receive ImageNet-normalised tensors from the
DataLoader.  They therefore denormalise to [0, 1] before perturbing, and
re-normalise afterwards so the model always receives correctly normalised input.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PERTURB_TYPES = Literal["brightness", "contrast", "gaussian_noise", "random_crop", "rotation"]

# ImageNet statistics (same as in dataset.py)
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


# ── normalisation helpers ─────────────────────────────────────────────────────

def _to_unit(img: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation: normalised tensor → [0, 1] float tensor."""
    mean = _MEAN.to(img.device).view(3, 1, 1)
    std  = _STD.to(img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0.0, 1.0)


def _from_unit(img: torch.Tensor) -> torch.Tensor:
    """Re-apply ImageNet normalisation: [0, 1] float tensor → normalised tensor."""
    mean = _MEAN.to(img.device).view(3, 1, 1)
    std  = _STD.to(img.device).view(3, 1, 1)
    return (img - mean) / std


# ── per-type perturbation helpers (all operate on [0, 1] tensors) ─────────────

def _apply_brightness(img: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale pixel values; clamp to [0, 1]."""
    return (img * scale).clamp(0.0, 1.0)


def _apply_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust contrast around channel mean; clamp to [0, 1]."""
    mean = img.mean(dim=(1, 2), keepdim=True)
    return ((img - mean) * factor + mean).clamp(0.0, 1.0)


def _apply_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise with given std; clamp to [0, 1]."""
    noise = torch.randn_like(img) * sigma
    return (img + noise).clamp(0.0, 1.0)


def _apply_random_crop(img: torch.Tensor, crop_ratio: float) -> torch.Tensor:
    """Centre-crop to crop_ratio × original size, then resize back."""
    _, h, w = img.shape
    ch = int(h * crop_ratio)
    cw = int(w * crop_ratio)
    top  = (h - ch) // 2
    left = (w - cw) // 2
    cropped = img[:, top:top + ch, left:left + cw]
    pil = TF.to_pil_image(cropped)
    pil = pil.resize((w, h), Image.Resampling.BILINEAR)
    return TF.to_tensor(pil)


def _apply_rotation(img: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate image; fill empty border with zeros."""
    pil = TF.to_pil_image(img)
    pil = TF.rotate(pil, degrees, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
    return TF.to_tensor(pil)


def apply_perturbation(
    img: torch.Tensor,
    perturb_type: str,
    strength: float,
) -> torch.Tensor:
    """
    Apply a single perturbation to an ImageNet-normalised tensor (C×H×W).

    Internally: denormalise → perturb in [0,1] → renormalise.
    The returned tensor is in the same normalised space as the input.
    """
    img_unit = _to_unit(img)          # → [0, 1]

    if perturb_type == "brightness":
        out = _apply_brightness(img_unit, strength)
    elif perturb_type == "contrast":
        out = _apply_contrast(img_unit, strength)
    elif perturb_type == "gaussian_noise":
        out = _apply_gaussian_noise(img_unit, strength)
    elif perturb_type == "random_crop":
        out = _apply_random_crop(img_unit, strength)
    elif perturb_type == "rotation":
        out = _apply_rotation(img_unit, strength)
    else:
        raise ValueError(f"Unknown perturbation type: {perturb_type}")

    return _from_unit(out)            # → normalised


# ── training perturbation ─────────────────────────────────────────────────────

class RandomPerturbation:
    """
    Randomly selects one perturbation type and samples its strength.

    Meant to be used INSIDE the training loop (not as a Dataset transform),
    so both the original x and the perturbed x' are available.

    Example:
        perturb = RandomPerturbation(cfg)
        x_prime = perturb(x)   # x is a batch tensor [B, C, H, W]
    """

    def __init__(self, cfg: dict):
        p = cfg.get("perturbation", cfg)
        self.types: list[str] = p.get("types", [
            "brightness", "contrast", "gaussian_noise", "random_crop", "rotation"
        ])
        self.ranges: dict[str, tuple[float, float]] = {
            "brightness":     tuple(p.get("brightness_range",              [0.8,  1.2])),
            "contrast":       tuple(p.get("contrast_range",                [0.8,  1.2])),
            "gaussian_noise": tuple(p.get("gaussian_noise_sigma_range",    [0.01, 0.05])),
            "random_crop":    tuple(p.get("random_crop_ratio_range",       [0.85, 0.95])),
            "rotation":       tuple(p.get("rotation_degrees_range",        [-10,  10])),
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a random perturbation to a batch x [B, C, H, W]."""
        ptype = random.choice(self.types)
        lo, hi = self.ranges[ptype]
        strength = random.uniform(lo, hi)
        return torch.stack([apply_perturbation(img, ptype, strength) for img in x])


# ── evaluation perturbation ───────────────────────────────────────────────────

class FixedPerturbation:
    """
    Applies a single perturbation type at a fixed strength.

    Args:
        perturb_type: one of the 5 types
        cfg:          full config dict (reads eval_* keys from perturbation section)
    """

    STRENGTH_KEYS = {
        "brightness":     "eval_brightness",
        "contrast":       "eval_contrast",
        "gaussian_noise": "eval_gaussian_noise_sigma",
        "random_crop":    "eval_random_crop_ratio",
        "rotation":       "eval_rotation_degrees",
    }

    def __init__(self, perturb_type: str, cfg: dict):
        if perturb_type not in self.STRENGTH_KEYS:
            raise ValueError(f"Unknown perturbation type: {perturb_type}")
        self.perturb_type = perturb_type
        p = cfg.get("perturbation", cfg)
        key = self.STRENGTH_KEYS[perturb_type]
        defaults = {
            "eval_brightness": 1.2,
            "eval_contrast": 1.2,
            "eval_gaussian_noise_sigma": 0.05,
            "eval_random_crop_ratio": 0.90,
            "eval_rotation_degrees": 10.0,
        }
        self.strength = p.get(key, defaults[key])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: single image [C, H, W] or batch [B, C, H, W]."""
        if x.dim() == 3:
            return apply_perturbation(x, self.perturb_type, self.strength)
        return torch.stack([apply_perturbation(img, self.perturb_type, self.strength) for img in x])
