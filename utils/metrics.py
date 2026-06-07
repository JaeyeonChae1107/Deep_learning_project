"""
Evaluation metrics for Task 2 — Model Behavior Analysis.

Prediction Consistency : prediction match rate + KL divergence between softmax distributions
Attention Consistency  : SSIM(H, H') + Top-K% activation pixel IoU
Attention Entropy      : entropy of softmax-normalized Grad-CAM map
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy as scipy_entropy
import torch
import torch.nn.functional as F
from typing import Dict


# ── Prediction Consistency ────────────────────────────────────────────────────

def compute_prediction_consistency(
    logits_orig: torch.Tensor,
    logits_pert: torch.Tensor,
) -> Dict[str, float]:
    """
    Args:
        logits_orig: (N, C) raw model outputs for original images
        logits_pert: (N, C) raw model outputs for perturbed images
    Returns:
        dict with 'match_rate' and 'mean_kl_divergence'
    """
    probs_orig = F.softmax(logits_orig, dim=1).cpu().numpy()   # (N, C)
    probs_pert = F.softmax(logits_pert, dim=1).cpu().numpy()   # (N, C)

    pred_orig = probs_orig.argmax(axis=1)
    pred_pert = probs_pert.argmax(axis=1)
    match_rate = (pred_orig == pred_pert).mean().item()

    # KL divergence: sum p * log(p/q), averaged over samples
    eps = 1e-10
    kl_per_sample = (probs_orig * np.log((probs_orig + eps) / (probs_pert + eps))).sum(axis=1)
    mean_kl = float(kl_per_sample.mean())

    return {"match_rate": match_rate, "mean_kl_divergence": mean_kl}


# ── Attention Consistency ─────────────────────────────────────────────────────

def compute_attention_consistency(
    cam_orig: np.ndarray,
    cam_pert: np.ndarray,
    top_k: float = 0.20,
) -> Dict[str, float]:
    """
    Args:
        cam_orig: (H, W) Grad-CAM heatmap for original image, values in [0, 1]
        cam_pert: (H, W) Grad-CAM heatmap for perturbed image, values in [0, 1]
        top_k:   fraction of pixels considered "high-activation"
    Returns:
        dict with 'ssim' and 'top_k_iou'
    """
    assert cam_orig.shape == cam_pert.shape, "Heatmap shapes must match"

    ssim_score = ssim(cam_orig, cam_pert, data_range=1.0)

    # Top-K% IoU
    k = max(1, int(cam_orig.size * top_k))
    thresh_orig = np.partition(cam_orig.ravel(), -k)[-k]
    thresh_pert = np.partition(cam_pert.ravel(), -k)[-k]
    mask_orig = cam_orig >= thresh_orig
    mask_pert = cam_pert >= thresh_pert
    intersection = (mask_orig & mask_pert).sum()
    union = (mask_orig | mask_pert).sum()
    iou = float(intersection / union) if union > 0 else 0.0

    return {"ssim": float(ssim_score), "top_k_iou": iou}


# ── Attention Entropy ─────────────────────────────────────────────────────────

def compute_attention_entropy(cam: np.ndarray) -> float:
    """
    Entropy of the softmax-normalized Grad-CAM map.
    Higher entropy → more diffuse attention; lower → more focused.
    """
    flat = cam.ravel().astype(np.float64)
    flat = flat - flat.max()           # numerical stability
    exp = np.exp(flat)
    prob = exp / exp.sum()
    return float(scipy_entropy(prob))
