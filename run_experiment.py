"""
Full experiment runner: trains all four models sequentially, then evaluates.

Usage:
    python run_experiment.py
    python run_experiment.py --skip-train   # evaluate only (checkpoints must exist)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
MODELS = ["baseline", "cbam", "baseline_loss", "cbam_loss"]


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label} — exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  Finished in {elapsed / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and run evaluation only")
    args = parser.parse_args()

    t_start = time.time()

    if not args.skip_train:
        for model in MODELS:
            run(
                [PYTHON, str(ROOT / "src" / "train.py"), "--model", model],
                f"Training  [{model}]",
            )

    run(
        [PYTHON, str(ROOT / "src" / "evaluate.py"), "--all"],
        "Evaluating all models",
    )

    total_h = (time.time() - t_start) / 3600
    print(f"\n{'='*60}")
    print(f"  Experiment complete — total time: {total_h:.1f} h")
    print(f"  TensorBoard : tensorboard --logdir {ROOT / 'runs'}")
    print(f"  Results     : {ROOT / 'results'}")
    print(f"  Figures     : {ROOT / 'figures'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
