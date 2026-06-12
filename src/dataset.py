"""
BeefGradingDataset — loads extracted images from dataset/images/{split}/{grade_folder}/.

Expected directory structure (created by setup_dataset.py):
  dataset/images/
    train/  grade_1pp/  grade_1p/  grade_1/  grade_2/  grade_3/
    val/    ...
    test/   ...

Grade index mapping:
  0: 1++  (grade_1pp)
  1: 1+   (grade_1p)
  2: 1    (grade_1)
  3: 2    (grade_2)
  4: 3    (grade_3)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


CLASS_NAMES = ["1++", "1+", "1", "2", "3"]
FOLDER_TO_IDX = {
    "grade_1pp": 0,
    "grade_1p":  1,
    "grade_1":   2,
    "grade_2":   3,
    "grade_3":   4,
}


def _make_transforms(image_size: int, split: str) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


class BeefGradingDataset(Dataset):
    """
    Args:
        root:       path to dataset/images/
        split:      'train', 'val', or 'test'
        image_size: resize both sides to this value
        transform:  optional override; if None, uses default for split
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root) / split
        self.transform = transform or _make_transforms(image_size, split)
        self.samples: list[tuple[Path, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.root}\n"
                "Run setup_dataset.py first."
            )

        for folder_name, class_idx in FOLDER_TO_IDX.items():
            folder = self.root / folder_name
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*.jpg")):
                self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights for CrossEntropyLoss weight argument.

        Uses sqrt scaling to avoid extreme weights for very rare classes.
        grade_3 has ~377 train samples vs ~19 822 for the top grades
        (ratio ~52:1).  Linear inverse-frequency gives grade_3 a weight of
        ~37, which causes severe loss spikes.  The sqrt dampens this to ~6,
        which still up-weights the rare class without destabilising training.
        """
        counts = torch.zeros(len(CLASS_NAMES))
        for _, label in self.samples:
            counts[label] += 1
        weights = (counts.sum() / (len(CLASS_NAMES) * counts.clamp(min=1))).sqrt()
        return weights


def build_loaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = BeefGradingDataset(data_root, "train", image_size)
    val_ds   = BeefGradingDataset(data_root, "val",   image_size)
    test_ds  = BeefGradingDataset(data_root, "test",  image_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
