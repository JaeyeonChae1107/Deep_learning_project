"""
Training script for all four model variants (2×2 ablation).

Usage:
    python src/train.py --model baseline       # Model A: ResNeXt-50
    python src/train.py --model cbam           # Model B: ResNeXt-50 + CBAM
    python src/train.py --model cbam_loss      # Model C: ResNeXt-50 + CBAM + KL loss
    python src/train.py --model baseline_loss  # Model D: ResNeXt-50 + KL loss (no CBAM)

All models receive perturbed images during training (CE on clean + CE on perturbed).
Models C and D additionally apply KL consistency loss:
    L = CE(p, y) + CE(p', y) + lambda_kl * KL(p || p')

Optimizer is configured via config.yaml (training.optimizer: sgd | adamw).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import build_loaders, CLASS_NAMES
from models import build_model
from src.perturbation import RandomPerturbation


# ── utilities ─────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    ce_fn: nn.CrossEntropyLoss,
    device: torch.device,
    model_type: str,
    perturb: RandomPerturbation,
    lambda_kl: float,
) -> tuple[float, float]:
    """Returns (avg_loss, avg_acc)."""
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        x_prime = perturb(x).to(device)

        if model_type in ("cbam_loss", "baseline_loss"):
            # Models C & D: CE(clean) + CE(perturbed) + KL
            loss, logits = model.consistency_loss(x, x_prime, y, ce_fn, lambda_kl)
        else:
            # Models A & B: CE(clean) + CE(perturbed), no KL
            logits       = model(x)
            logits_pert  = model(x_prime)
            loss         = ce_fn(logits, y) + ce_fn(logits_pert, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.detach().item()
        total_acc  += accuracy(logits.detach(), y)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    ce_fn: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (avg_loss, avg_acc)."""
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += ce_fn(logits, y).item()
        total_acc  += accuracy(logits, y)
        n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train beef grading model")
    parser.add_argument(
        "--model", required=True,
        choices=["baseline", "cbam", "cbam_loss", "baseline_loss"],
        help="Model variant to train",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────────────────
    os.chdir(_PROJECT_ROOT)
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_loaders(
        data_root   = cfg["data"]["root"],
        image_size  = cfg["data"]["image_size"],
        batch_size  = cfg["training"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        model_type  = args.model,
        num_classes = cfg["data"]["num_classes"],
        pretrained  = cfg["model"]["pretrained"],
    ).to(device)

    # ── Loss ───────────────────────────────────────────────────────────────
    if cfg["training"]["use_class_weights"]:
        weights = train_loader.dataset.class_weights().to(device)
        print(f"Class weights: {weights.cpu().numpy().round(3)}")
        ce_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        ce_fn = nn.CrossEntropyLoss()

    lambda_kl = cfg["consistency_loss"]["lambda_kl"]
    perturb   = RandomPerturbation(cfg)  # all models see perturbed images during training

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    opt_name = cfg["training"].get("optimizer", "adamw").lower()
    if opt_name == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr           = cfg["training"]["lr"],
            momentum     = cfg["training"].get("momentum", 0.9),
            weight_decay = cfg["training"]["weight_decay"],
        )
        print(f"Optimizer: SGD (lr={cfg['training']['lr']}, momentum={cfg['training'].get('momentum', 0.9)})")
    else:
        optimizer = AdamW(
            model.parameters(),
            lr           = cfg["training"]["lr"],
            weight_decay = cfg["training"]["weight_decay"],
        )
        print(f"Optimizer: AdamW (lr={cfg['training']['lr']})")

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = cfg["training"]["epochs"],
        eta_min = 1e-6,
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch    = 1
    best_val_acc   = 0.0
    no_improve     = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']} (best val acc: {best_val_acc:.4f})")

    # ── Paths ──────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.model}_best.pth"
    last_path = ckpt_dir / f"{args.model}_last.pth"

    runs_dir = Path(cfg["paths"].get("runs_dir", "./runs")) / args.model
    writer   = SummaryWriter(log_dir=str(runs_dir))
    print(f"TensorBoard logs → {runs_dir}")

    # ── Training loop ──────────────────────────────────────────────────────
    epochs    = cfg["training"]["epochs"]
    patience  = cfg["training"]["early_stopping_patience"]

    print(f"\nStarting training for {epochs} epochs ...")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, ce_fn,
            device, args.model, perturb, lambda_kl,
        )
        val_loss, val_acc = evaluate(model, val_loader, ce_fn, device)
        scheduler.step()

        # ── TensorBoard logging ────────────────────────────────────────────
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("LR",         scheduler.get_last_lr()[0], epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        state = {
            "epoch":           epoch,
            "model_type":      args.model,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc":    best_val_acc,
            "val_acc":         val_acc,
        }
        torch.save(state, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(state, best_path)
            print(f"  -> New best val acc: {best_val_acc:.4f} (saved)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break

    writer.close()
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"TensorBoard:     tensorboard --logdir {runs_dir.parent}")


if __name__ == "__main__":
    main()
