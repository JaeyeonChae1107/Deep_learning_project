from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    SummaryWriter = None  # type: ignore[assignment]

from beef_cbam.data import describe_dataset, infer_num_classes, make_loader  # noqa: E402
from beef_cbam.engine import evaluate_classification, get_device, save_checkpoint, train_one_epoch  # noqa: E402
from beef_cbam.models import MODEL_SPECS, create_model, load_simclr_encoder_weights  # noqa: E402
from beef_cbam.perturbations import ATTENTION_SAFE_PERTURBATIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CBAM beef grading model variant.")
    data_root = PROJECT_ROOT / "data/075.축산물_품질(QC)_이미지(소,_닭,_달걀)/01.데이터"
    parser.add_argument("--train-dir", type=Path, default=data_root / "1.Training")
    parser.add_argument("--val-dir", type=Path, default=data_root / "2.Validation")
    parser.add_argument("--variant", default="resnext50", choices=sorted(MODEL_SPECS))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-consistency", type=float, default=0.5)
    parser.add_argument("--lambda-attention", type=float, default=0.2)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--simclr-checkpoint", type=Path, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard-logdir", type=Path, default=PROJECT_ROOT / "outputs/tensorboard")
    return parser.parse_args()


def write_epoch_to_tensorboard(writer: object | None, metrics: dict[str, float | str], step: int) -> None:
    if writer is None:
        return
    scalar_map = {
        "loss/train": "train_loss",
        "loss/val": "val_loss",
        "metrics/val_accuracy": "val_accuracy",
        "metrics/val_macro_f1": "val_macro_f1",
        "samples/train": "train_samples",
        "samples/val": "val_samples",
        "attention/train_mse": "train_attention_mse",
    }
    for tag, key in scalar_map.items():
        value = metrics.get(key)
        if isinstance(value, int | float):
            writer.add_scalar(tag, float(value), step)  # type: ignore[attr-defined]


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = get_device()
    num_classes = infer_num_classes(args.train_dir)
    spec = MODEL_SPECS[args.variant]

    print("device:", device)
    print("variant:", args.variant)
    print("train:", describe_dataset(args.train_dir))
    print("val:", describe_dataset(args.val_dir))

    perturbation_names = ATTENTION_SAFE_PERTURBATIONS if spec.uses_attention_consistency_loss else None
    if perturbation_names is not None:
        print("training_perturbations_for_attention_loss:", perturbation_names)

    train_loader = make_loader(
        args.train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=True,
        num_workers=args.num_workers,
        consistency=spec.uses_consistency_loss,
        perturbation_names=perturbation_names,
    )
    val_loader = make_loader(
        args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_model(args.variant, num_classes=num_classes, pretrained=args.pretrained).to(device)
    if args.simclr_checkpoint is not None:
        model = load_simclr_encoder_weights(model, str(args.simclr_checkpoint), map_location=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_dir = PROJECT_ROOT / "outputs/metrics"
    checkpoint_dir = PROJECT_ROOT / "outputs/checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_metrics: dict[str, float | str] | None = None
    history: list[dict[str, float | str]] = []
    writer = None
    if args.tensorboard and SummaryWriter is not None:
        tb_dir = args.tensorboard_logdir / "train_model" / args.variant
        writer = SummaryWriter(str(tb_dir))
        writer.add_text(
            "config",
            json.dumps(vars(args), ensure_ascii=False, default=str, indent=2),
            0,
        )
        print("tensorboard_logdir:", tb_dir)
    elif args.tensorboard:
        print("tensorboard_unavailable: install tensorboard to enable event logging")

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                consistency=spec.uses_consistency_loss,
                attention_consistency=spec.uses_attention_consistency_loss,
                lambda_consistency=args.lambda_consistency,
                lambda_attention=args.lambda_attention,
                max_batches=args.max_train_batches,
            )
            val_metrics = evaluate_classification(
                model,
                val_loader,
                device,
                max_batches=args.max_val_batches,
            )
            epoch_metrics: dict[str, float | str] = {
                "epoch": float(epoch),
                "variant": args.variant,
                "train_loss": train_metrics["loss"],
                "train_samples": train_metrics["samples"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_samples": val_metrics["samples"],
            }
            if "attention_mse" in train_metrics:
                epoch_metrics["train_attention_mse"] = train_metrics["attention_mse"]

            print("epoch_metrics:", json.dumps(epoch_metrics, ensure_ascii=False, indent=2))
            history.append(epoch_metrics)
            write_epoch_to_tensorboard(writer, epoch_metrics, epoch)
            if writer is not None:
                writer.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_metrics = epoch_metrics
                save_checkpoint(checkpoint_dir / f"{args.variant}_best.pt", model, optimizer, epoch, epoch_metrics)
    finally:
        if writer is not None:
            writer.close()

    history_path = metrics_dir / f"{args.variant}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("saved_history:", history_path)

    if best_metrics is not None:
        metrics_path = metrics_dir / f"{args.variant}_best_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        print("saved_best_metrics:", metrics_path)


if __name__ == "__main__":
    main()
