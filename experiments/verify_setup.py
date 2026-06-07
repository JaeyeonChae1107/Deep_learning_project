"""
Quick sanity-check script: instantiate all models and run a dummy forward pass.
Run before training to confirm all dependencies are correctly installed.

Usage: python experiments/verify_setup.py
"""

import torch
import torch.nn.functional as F
import numpy as np

from models import ResNet50Baseline, ResNet50CBAM
from utils import GradCAM, PerturbationSet
from utils.metrics import (
    compute_prediction_consistency,
    compute_attention_consistency,
    compute_attention_entropy,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Models ───────────────────────────────────────────────────────────────
    baseline = ResNet50Baseline(num_classes=5, pretrained=False).to(device)
    cbam_net = ResNet50CBAM(num_classes=5, pretrained=False).to(device)

    dummy = torch.randn(2, 3, 512, 512).to(device)

    with torch.no_grad():
        out_b = baseline(dummy)
        out_c = cbam_net(dummy)

    assert out_b.shape == (2, 5), f"Unexpected baseline output shape: {out_b.shape}"
    assert out_c.shape == (2, 5), f"Unexpected CBAM output shape: {out_c.shape}"
    print("ResNet50Baseline      : OK", out_b.shape)
    print("ResNet50CBAM          : OK", out_c.shape)

    # ── Grad-CAM ─────────────────────────────────────────────────────────────
    gcam = GradCAM(baseline, target_layer_name="layer4")
    x_single = torch.randn(1, 3, 512, 512).to(device).requires_grad_(True)
    cam = gcam.generate(x_single)
    assert cam.shape == (512, 512)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
    print(f"GradCAM (baseline)    : OK  shape={cam.shape}  range=[{cam.min():.3f}, {cam.max():.3f}]")

    # ── Perturbation ─────────────────────────────────────────────────────────
    from PIL import Image
    pset = PerturbationSet()
    pil_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    perturbed = pset.apply_all(pil_img)
    assert set(perturbed.keys()) == {"brightness", "contrast", "gaussian_noise", "random_crop", "rotation"}
    print("PerturbationSet       : OK  types=" + ", ".join(perturbed.keys()))

    # ── Metrics ──────────────────────────────────────────────────────────────
    logits_a = torch.randn(4, 5)
    logits_b = torch.randn(4, 5)
    pc = compute_prediction_consistency(logits_a, logits_b)
    print(f"PredictionConsistency : OK  match_rate={pc['match_rate']:.2f}  kl={pc['mean_kl_divergence']:.4f}")

    cam_a = np.random.rand(512, 512).astype(np.float32)
    cam_b = np.random.rand(512, 512).astype(np.float32)
    ac = compute_attention_consistency(cam_a, cam_b)
    print(f"AttentionConsistency  : OK  ssim={ac['ssim']:.4f}  iou={ac['top_k_iou']:.4f}")

    ent = compute_attention_entropy(cam_a)
    print(f"AttentionEntropy      : OK  entropy={ent:.4f}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
