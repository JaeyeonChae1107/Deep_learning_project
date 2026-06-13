from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


# Perturbations that preserve spatial layout — safe for attention map comparison
ATTENTION_SAFE_PERTURBATIONS = ["brightness", "contrast", "gaussian_noise"]
ALL_PERTURBATIONS = ["brightness", "contrast", "gaussian_noise", "random_crop", "rotation"]

_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD = torch.tensor([0.229, 0.224, 0.225])


def _to_unit(img: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(img.device).view(3, 1, 1)
    std = _STD.to(img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0.0, 1.0)


def _from_unit(img: torch.Tensor) -> torch.Tensor:
    mean = _MEAN.to(img.device).view(3, 1, 1)
    std = _STD.to(img.device).view(3, 1, 1)
    return (img - mean) / std


def _apply_brightness(img: torch.Tensor, scale: float) -> torch.Tensor:
    return (img * scale).clamp(0.0, 1.0)


def _apply_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    mean = img.mean(dim=(1, 2), keepdim=True)
    return ((img - mean) * factor + mean).clamp(0.0, 1.0)


def _apply_gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)


def _apply_random_crop(img: torch.Tensor, crop_ratio: float) -> torch.Tensor:
    _, h, w = img.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = img[:, top:top + ch, left:left + cw]
    pil = TF.to_pil_image(cropped)
    pil = pil.resize((w, h), Image.Resampling.BILINEAR)
    return TF.to_tensor(pil)


def _apply_rotation(img: torch.Tensor, degrees: float) -> torch.Tensor:
    pil = TF.to_pil_image(img)
    pil = TF.rotate(pil, degrees, interpolation=InterpolationMode.BILINEAR, expand=False, fill=0)
    return TF.to_tensor(pil)


def apply_perturbation(img: torch.Tensor, perturb_type: str, strength: float) -> torch.Tensor:
    """img: ImageNet-normalised [C,H,W]. Returns normalised tensor."""
    unit = _to_unit(img)
    if perturb_type == "brightness":
        out = _apply_brightness(unit, strength)
    elif perturb_type == "contrast":
        out = _apply_contrast(unit, strength)
    elif perturb_type == "gaussian_noise":
        out = _apply_gaussian_noise(unit, strength)
    elif perturb_type == "random_crop":
        out = _apply_random_crop(unit, strength)
    elif perturb_type == "rotation":
        out = _apply_rotation(unit, strength)
    else:
        raise ValueError(f"Unknown perturbation: {perturb_type}")
    return _from_unit(out)


class RandomPerturbation:
    """Randomly apply one perturbation to a single image [C,H,W] tensor."""

    _RANGES: dict[str, tuple[float, float]] = {
        "brightness":     (0.8,  1.2),
        "contrast":       (0.8,  1.2),
        "gaussian_noise": (0.01, 0.05),
        "random_crop":    (0.85, 0.95),
        "rotation":       (-10.0, 10.0),
    }

    def __init__(self, perturbation_names: list[str] | None = None) -> None:
        self.types = list(perturbation_names) if perturbation_names else ALL_PERTURBATIONS

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [C,H,W] single image."""
        ptype = random.choice(self.types)
        lo, hi = self._RANGES[ptype]
        strength = random.uniform(lo, hi)
        return apply_perturbation(x, ptype, strength)
