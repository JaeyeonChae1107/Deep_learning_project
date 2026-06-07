"""
Perturbation functions for Task 2 — Model Behavior Analysis.
All perturbations simulate realistic industrial variation, per Lim & Song (2025) Section 5.

Each function takes a PIL Image and returns a perturbed PIL Image.
PerturbationSet provides a unified interface over all perturbation types.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple


def perturb_brightness(img: Image.Image, delta: float = 0.2) -> Image.Image:
    """Scale pixel values by a factor uniformly sampled from [1-delta, 1+delta]."""
    factor = random.uniform(1.0 - delta, 1.0 + delta)
    arr = np.array(img, dtype=np.float32) * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def perturb_contrast(img: Image.Image, low: float = 0.8, high: float = 1.2) -> Image.Image:
    """Apply contrast scaling factor uniformly sampled from [low, high]."""
    factor = random.uniform(low, high)
    return ImageEnhance.Contrast(img).enhance(factor)


def perturb_gaussian_noise(img: Image.Image, sigma: float = 0.05) -> Image.Image:
    """Add Gaussian noise with std = sigma * 255 (normalized scale)."""
    arr = np.array(img, dtype=np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def perturb_random_crop(img: Image.Image, crop_ratio: float = 0.9) -> Image.Image:
    """Random crop keeping crop_ratio of each dimension, then resize back to original."""
    w, h = img.size
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)
    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.BILINEAR)


def perturb_rotation(img: Image.Image, max_degrees: float = 10.0) -> Image.Image:
    """Rotate by a random angle uniformly sampled from [-max_degrees, max_degrees]."""
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)


class PerturbationSet:
    """
    Applies all perturbation types one by one (deterministic enumeration).
    Useful for systematic evaluation: for each image, generate one perturbed
    version per perturbation type and compute consistency metrics.
    """

    PERTURBATIONS = {
        "brightness": perturb_brightness,
        "contrast": perturb_contrast,
        "gaussian_noise": perturb_gaussian_noise,
        "random_crop": perturb_random_crop,
        "rotation": perturb_rotation,
    }

    def __init__(self, config: dict = None):
        self.config = config or {}

    def apply(self, img: Image.Image, perturbation_name: str) -> Image.Image:
        fn = self.PERTURBATIONS[perturbation_name]
        kwargs = self._get_kwargs(perturbation_name)
        return fn(img, **kwargs)

    def apply_all(self, img: Image.Image) -> dict:
        return {name: self.apply(img, name) for name in self.PERTURBATIONS}

    def _get_kwargs(self, name: str) -> dict:
        cfg = self.config.get(name, {})
        if name == "brightness":
            return {"delta": cfg.get("delta", 0.2)}
        if name == "contrast":
            return {"low": cfg.get("low", 0.8), "high": cfg.get("high", 1.2)}
        if name == "gaussian_noise":
            return {"sigma": cfg.get("sigma", 0.05)}
        if name == "random_crop":
            return {"crop_ratio": cfg.get("crop_ratio", 0.9)}
        if name == "rotation":
            return {"max_degrees": cfg.get("max_degrees", 10.0)}
        return {}
