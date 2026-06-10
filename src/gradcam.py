"""
Grad-CAM implementation for ResNeXt-50 (baseline and CBAM variants).

Target layer: 'layer4' for all models (as specified in config).
For Model B/C, gradients from the classification head flow back through cbam4
into layer4, so the attention signal is already incorporated in the Grad-CAM
output even when hooking at layer4.

Usage:
    cam = GradCAM(model, target_layer_name='layer4')
    heatmap = cam.compute(img_tensor)   # img_tensor: [C, H, W] or [1, C, H, W]
    cam.remove_hooks()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self._features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self):
        modules = dict(self.model.named_modules())
        if self.target_layer_name not in modules:
            available = [k for k in modules if "layer" in k]
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found. "
                f"Available layer-like names: {available}"
            )
        target = modules[self.target_layer_name]

        def _fwd_hook(module, inp, out):
            self._features = out  # keep computation graph for backward

        def _bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles.append(target.register_forward_hook(_fwd_hook))
        self._handles.append(target.register_full_backward_hook(_bwd_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.enable_grad()
    def compute(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap.

        Args:
            x:          image tensor [C, H, W] or [1, C, H, W]
            class_idx:  class to explain; if None, uses argmax of the prediction

        Returns:
            heatmap: np.ndarray [H, W], values in [0, 1]
        """
        device = next(self.model.parameters()).device

        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)

        self.model.eval()
        self.model.zero_grad()

        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Scalar score for the target class
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM: global-average-pool the gradients over spatial dims
        # gradients: [1, C, H, W]  features: [1, C, H, W]
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1]
        cam = (weights * self._features.detach()).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=(x.shape[2], x.shape[3]),
            mode="bilinear", align_corners=False,
        )  # [1, 1, H, W]

        cam = cam.squeeze().cpu().numpy()  # [H, W]

        # Normalise to [0, 1]
        lo, hi = cam.min(), cam.max()
        if hi > lo:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)

        return cam


def top_k_mask(heatmap: np.ndarray, k: float = 0.20) -> np.ndarray:
    """
    Return a binary mask of the top-k fraction of activations.

    Args:
        heatmap: [H, W] array in [0, 1]
        k:       fraction of pixels to include (e.g. 0.20 for top-20%)

    Returns:
        binary mask [H, W] (bool)
    """
    threshold = np.quantile(heatmap, 1.0 - k)
    return heatmap >= threshold


def cam_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Intersection over Union of two binary activation masks.

    Returns:
        IoU in [0, 1]; 0 if union is empty
    """
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0
