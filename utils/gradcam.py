"""
Grad-CAM implementation for uniform visualization across ResNet50 and ResNet50+CBAM.
Generates class-discriminative localization maps using gradients of the target class
score w.r.t. the target convolutional layer (default: layer4).

Reference: Selvaraju et al., ICCV 2017 — https://arxiv.org/abs/1610.02391
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GradCAM:
    """Hook-based Grad-CAM for any model with a named target layer."""

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name: str):
        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            raise ValueError(f"Layer '{layer_name}' not found in model. "
                             f"Available: {list(dict(self.model.named_modules()).keys())}")

        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(target.register_forward_hook(fwd_hook))
        self._hook_handles.append(target.register_full_backward_hook(bwd_hook))

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Returns a normalized Grad-CAM heatmap in [0, 1], shape (H, W).
        If class_idx is None, uses the predicted class.
        """
        self.model.eval()
        x = x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global Average Pooling of gradients → channel weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        h, w = x.shape[2], x.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __del__(self):
        self.remove_hooks()
