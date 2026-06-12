"""
Grad-CAM implementation for ResNeXt-50 (baseline and CBAM variants).

Target layer: 'layer4' for all models (as specified in config).
For Model B/C, CBAM is applied inside each Bottleneck block (before the residual
add), so the features captured at layer4 and the gradients flowing back both
already reflect the CBAM attention signal.

Usage:
    cam = GradCAM(model, target_layer_name='layer4')
    heatmap = cam.compute(img_tensor)   # img_tensor: [C, H, W] or [1, C, H, W]
    cam.remove_hooks()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on headless servers)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


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


def save_gradcam_figure(
    img_norm: torch.Tensor,
    cam_clean: np.ndarray,
    save_path: Path,
    cam_perturbed: Optional[np.ndarray] = None,
    img_norm_perturbed: Optional[torch.Tensor] = None,
    class_name: str = "",
    perturb_type: str = "",
) -> None:
    """
    Save a Grad-CAM visualisation figure.

    Non-spatial perturbations (img_norm_perturbed=None):
      [Original] | [Grad-CAM clean] | [Grad-CAM perturbed]
      Both CAMs are overlaid on the same original image; spatial content identical.

    Spatial perturbations (img_norm_perturbed provided):
      [Original + Grad-CAM clean] | [Perturbed image + Grad-CAM perturbed]
      Each panel uses its own image so the crop/rotation is actually visible.

    Args:
        img_norm:             normalised clean image tensor [C, H, W]
        cam_clean:            Grad-CAM heatmap for clean image [H, W] in [0, 1]
        save_path:            file path to save the PNG
        cam_perturbed:        Grad-CAM heatmap for perturbed image (None → 2-panel)
        img_norm_perturbed:   normalised perturbed image tensor (spatial transforms only)
        class_name:           predicted class label for the title
        perturb_type:         perturbation type name for the title
    """
    def _denorm(t: torch.Tensor) -> np.ndarray:
        arr = t.cpu().numpy().transpose(1, 2, 0)
        return (arr * _IMAGENET_STD + _IMAGENET_MEAN).clip(0.0, 1.0)

    def _overlay(img: np.ndarray, cam: np.ndarray) -> np.ndarray:
        heatmap = plt.cm.jet(cam)[..., :3]
        return (0.55 * img + 0.45 * heatmap).clip(0.0, 1.0)

    img_np = _denorm(img_norm)

    if img_norm_perturbed is not None:
        # Spatial: each panel shows its own image so crop/rotation is visible
        img_pert_np = _denorm(img_norm_perturbed)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(_overlay(img_np, cam_clean))
        axes[0].set_title(f"Original – Grad-CAM\n({class_name})")
        axes[0].axis("off")

        axes[1].imshow(_overlay(img_pert_np, cam_perturbed))
        axes[1].set_title(f"{perturb_type} – Grad-CAM")
        axes[1].axis("off")
    else:
        # Photometric: overlay both CAMs on the same original image
        n_cols = 3 if cam_perturbed is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        axes[0].imshow(img_np)
        axes[0].set_title(f"Original\n({class_name})")
        axes[0].axis("off")

        axes[1].imshow(_overlay(img_np, cam_clean))
        axes[1].set_title("Grad-CAM (clean)")
        axes[1].axis("off")

        if cam_perturbed is not None:
            axes[2].imshow(_overlay(img_np, cam_perturbed))
            axes[2].set_title(f"Grad-CAM ({perturb_type})")
            axes[2].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
