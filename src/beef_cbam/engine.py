from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    consistency: bool = False,
    attention_consistency: bool = False,
    lambda_consistency: float = 0.5,
    lambda_attention: float = 0.2,
    max_batches: int | None = None,
) -> dict:
    model.train()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_correct = 0
    n_samples = 0
    n_batches = 0

    for batch in loader:
        if consistency:
            x, x_prime, y = batch
            x_prime = x_prime.to(device)
        else:
            x, y = batch

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if consistency:
            logits_clean = model(x)
            logits_pert = model(x_prime)
            p_clean = F.softmax(logits_clean, dim=1)
            log_p_pert = F.log_softmax(logits_pert, dim=1)
            loss = (
                ce_fn(logits_clean, y)
                + ce_fn(logits_pert, y)
                + lambda_consistency * F.kl_div(log_p_pert, p_clean, reduction="batchmean")
            )
            logits = logits_clean
        else:
            logits = model(x)
            loss = ce_fn(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.detach().item()
        n_correct += (logits.detach().argmax(dim=1) == y).sum().item()
        n_samples += y.size(0)
        n_batches += 1

        if max_batches is not None and n_batches >= max_batches:
            break

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": n_correct / max(n_samples, 1),
        "samples": n_samples,
    }


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_samples = 0
    n_batches = 0

    for batch in loader:
        x, y = batch[0], batch[-1]  # works for both (x,y) and (x,x',y)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += ce_fn(logits, y).item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        n_samples += y.size(0)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    try:
        from sklearn.metrics import f1_score
        macro_f1: float = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    except ImportError:
        n_cls = max(all_labels) + 1 if all_labels else 1
        f1s = []
        for c in range(n_cls):
            tp = sum(p == c and l == c for p, l in zip(all_preds, all_labels))
            fp = sum(p == c and l != c for p, l in zip(all_preds, all_labels))
            fn = sum(p != c and l == c for p, l in zip(all_preds, all_labels))
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    n_correct = sum(p == l for p, l in zip(all_preds, all_labels))
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": n_correct / max(n_samples, 1),
        "macro_f1": macro_f1,
        "samples": n_samples,
    }


def save_checkpoint(path: Path | str, model: nn.Module, optimizer, epoch: int, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
