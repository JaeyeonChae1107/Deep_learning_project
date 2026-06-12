"""
Reliability evaluation for all four model variants (2×2 ablation).

For each model and each perturbation type, this script reports:

  Classification performance (clean images):
    - Accuracy, macro F1-score, per-class F1

  Prediction stability (clean vs. perturbed):
    - Top-1 agreement: fraction where argmax(p) == argmax(p')
    - KL divergence:   mean KL(p || p')
    - JSD:             mean Jensen-Shannon divergence

  Grad-CAM figures (10 fixed images, 2 per grade):
    - Same images across all four models for direct visual comparison
    - Photometric: [original | CAM clean | CAM perturbed]
    - Spatial:     [original + CAM clean | perturbed image + CAM perturbed]

Results are saved to results/{model_type}_eval.json and a summary CSV.

Usage:
    python src/evaluate.py --model baseline
    python src/evaluate.py --model cbam
    python src/evaluate.py --model cbam_loss
    python src/evaluate.py --model baseline_loss

    # Evaluate all four and print comparison table:
    python src/evaluate.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import BeefGradingDataset, CLASS_NAMES
from src.gradcam import GradCAM, save_gradcam_figure
from models import build_model
from src.perturbation import FixedPerturbation


PERTURB_TYPES_LIST = ["brightness", "contrast", "gaussian_noise", "random_crop", "rotation"]
SPATIAL_SKIP = {"random_crop", "rotation"}  # show perturbed image in figure (2-panel)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between probability vectors (in nats)."""
    m = 0.5 * (p + q)
    jsd = 0.5 * F.kl_div(m.log(), p, reduction="sum") \
        + 0.5 * F.kl_div(m.log(), q, reduction="sum")
    return jsd.item()


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p || q) in nats; adds small epsilon to avoid log(0)."""
    eps = 1e-8
    p = p + eps
    q = q + eps
    return F.kl_div(q.log(), p, reduction="sum").item()


# ── per-perturbation evaluation ───────────────────────────────────────────────

@torch.no_grad()
def eval_classification(
    model: torch.nn.Module,
    dataset: BeefGradingDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Accuracy and per-class F1 on clean images."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.eval()
    all_preds, all_targets = [], []

    for x, y in tqdm(loader, desc="classification", leave=False):
        x = x.to(device)
        logits = model(x)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(y.tolist())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc         = (all_preds == all_targets).mean()
    f1_macro    = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_per_class = f1_score(
        all_targets, all_preds,
        average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
    ).tolist()

    return {
        "accuracy":     float(acc),
        "f1_macro":     float(f1_macro),
        "f1_per_class": {c: float(f1_per_class[i]) for i, c in enumerate(CLASS_NAMES)},
    }


def eval_perturbation(
    model: torch.nn.Module,
    dataset: BeefGradingDataset,
    perturb_type: str,
    cfg: dict,
    device: torch.device,
    figures_dir: Path | None = None,
    n_save: int = 0,
) -> dict:
    """
    Compute prediction stability for a single perturbation type.

    Agreement / KL / JSD are computed over the full test set.
    Grad-CAM figures are saved only for the quota images (n_per_class per grade),
    so the same fixed images are visualised across all four models.

    Args:
        figures_dir: directory to save Grad-CAM figures (None → skip)
        n_save:      total Grad-CAM figures per perturbation type, split
                     evenly across grades (10 → 2 per grade)
    """
    perturb = FixedPerturbation(perturb_type, cfg)
    gradcam = GradCAM(model, target_layer_name=cfg["evaluation"]["gradcam_layer"])

    skip_spatial = perturb_type in SPATIAL_SKIP

    agreements, kl_divs, jsd_vals = [], [], []
    n_per_class = max(1, n_save // len(CLASS_NAMES))
    saved_per_class: dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}

    model.eval()
    for x, label in tqdm(dataset, desc=f"perturb={perturb_type}", leave=False):
        x = x.to(device)

        with torch.no_grad():
            p_logits = model(x.unsqueeze(0))
            p        = F.softmax(p_logits, dim=1).squeeze(0).cpu()

        x_prime = perturb(x).to(device)
        with torch.no_grad():
            p2_logits = model(x_prime.unsqueeze(0))
            p2        = F.softmax(p2_logits, dim=1).squeeze(0).cpu()

        pred_orig = int(p.argmax())
        pred_pert = int(p2.argmax())

        agreements.append(int(pred_orig == pred_pert))
        kl_divs.append(kl_divergence(p, p2))
        jsd_vals.append(js_divergence(p, p2))

        # Grad-CAM only for the fixed quota images (same 2 per grade across all models)
        if figures_dir is not None and saved_per_class[label] < n_per_class:
            cam_orig = gradcam.compute(x,       class_idx=pred_orig)
            cam_pert = gradcam.compute(x_prime, class_idx=pred_orig)
            idx = saved_per_class[label]

            if not skip_spatial:
                save_gradcam_figure(
                    img_norm      = x.cpu(),
                    cam_clean     = cam_orig,
                    cam_perturbed = cam_pert,
                    save_path     = figures_dir / perturb_type / f"grade{CLASS_NAMES[label]}_{idx:02d}.png",
                    class_name    = CLASS_NAMES[pred_orig],
                    perturb_type  = perturb_type,
                )
            else:
                save_gradcam_figure(
                    img_norm           = x.cpu(),
                    cam_clean          = cam_orig,
                    cam_perturbed      = cam_pert,
                    img_norm_perturbed = x_prime.cpu(),
                    save_path          = figures_dir / perturb_type / f"grade{CLASS_NAMES[label]}_{idx:02d}.png",
                    class_name         = CLASS_NAMES[pred_orig],
                    perturb_type       = perturb_type,
                )
            saved_per_class[label] += 1

    gradcam.remove_hooks()

    return {
        "top1_agreement": float(np.mean(agreements)),
        "kl_divergence":  float(np.mean(kl_divs)),
        "jsd":            float(np.mean(jsd_vals)),
    }


# ── full evaluation pipeline ──────────────────────────────────────────────────

def evaluate_model(model_type: str, cfg: dict, device: torch.device) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_type}")
    print(f"{'='*60}")

    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / f"{model_type}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python src/train.py --model {model_type}"
        )

    # Build model and load weights
    model = build_model(model_type, cfg["data"]["num_classes"], pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val acc {ckpt['val_acc']:.4f})")

    test_ds = BeefGradingDataset(
        cfg["data"]["root"], "test", cfg["data"]["image_size"]
    )
    print(f"Test set: {len(test_ds)} images")

    results: dict = {}

    # ── Clean-image classification ─────────────────────────────────────────
    print("\n[1/2] Classification performance (clean images) ...")
    results["classification"] = eval_classification(model, test_ds, device)
    cls = results["classification"]
    print(f"  Accuracy: {cls['accuracy']:.4f} | F1 macro: {cls['f1_macro']:.4f}")
    for grade, f1 in cls["f1_per_class"].items():
        print(f"    {grade}: {f1:.4f}")

    # ── Per-perturbation reliability ───────────────────────────────────────
    figures_dir = Path(cfg["paths"]["figures_dir"]) / model_type
    n_save      = cfg["evaluation"].get("n_gradcam_samples", 5)
    print(f"\n[2/2] Perturbation stability (saving {n_save} Grad-CAM samples → {figures_dir}) ...")
    results["perturbation"] = {}
    for ptype in PERTURB_TYPES_LIST:
        print(f"\n  Perturbation: {ptype}")
        r = eval_perturbation(
            model, test_ds, ptype, cfg, device,
            figures_dir = figures_dir,
            n_save      = n_save,
        )
        results["perturbation"][ptype] = r
        print(f"    Top-1 agreement : {r['top1_agreement']:.4f}")
        print(f"    KL divergence   : {r['kl_divergence']:.4f}")
        print(f"    JSD             : {r['jsd']:.4f}")

    return results


def save_results(model_type: str, results: dict, results_dir: str):
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_type}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


def print_comparison_table(all_results: dict[str, dict]):
    """Print a compact comparison table across all evaluated models."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'Metric':<35}" + "".join(f"{m:>15}" for m in all_results)
    print(header)
    print("-" * 80)

    # Classification
    def row(label, vals):
        return f"{label:<35}" + "".join(f"{v:>15.4f}" if v is not None else f"{'N/A':>15}" for v in vals)

    models = list(all_results.keys())
    print(row("Accuracy",     [all_results[m]["classification"]["accuracy"]  for m in models]))
    print(row("F1 macro",     [all_results[m]["classification"]["f1_macro"]  for m in models]))
    print()

    # Perturbation
    for ptype in PERTURB_TYPES_LIST:
        print(f"  [{ptype}]")
        print(row("  top1_agreement", [all_results[m]["perturbation"][ptype]["top1_agreement"] for m in models]))
        print(row("  kl_divergence",  [all_results[m]["perturbation"][ptype]["kl_divergence"]  for m in models]))
        print(row("  jsd",            [all_results[m]["perturbation"][ptype]["jsd"]            for m in models]))
        print()

    print("=" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate beef grading model reliability")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", choices=["baseline", "cbam", "cbam_loss", "baseline_loss"],
        help="Single model to evaluate",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Evaluate all four models and print comparison table",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    os.chdir(_PROJECT_ROOT)
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.all:
        all_results = {}
        for mtype in ["baseline", "cbam", "baseline_loss", "cbam_loss"]:
            try:
                r = evaluate_model(mtype, cfg, device)
                save_results(mtype, r, cfg["paths"]["results_dir"])
                all_results[mtype] = r
            except FileNotFoundError as e:
                print(f"[SKIP] {mtype}: {e}")
        if all_results:
            print_comparison_table(all_results)
    else:
        r = evaluate_model(args.model, cfg, device)
        save_results(args.model, r, cfg["paths"]["results_dir"])


if __name__ == "__main__":
    main()
