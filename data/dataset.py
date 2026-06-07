"""
AI-Hub Beef Carcass Dataset loader.
Dataset: 77,899 RGB images, 5 grades (1++, 1+, 1, 2, 3).

Expected directory structure:
    dataset/
        1++/  image1.jpg ...
        1+/   ...
        1/    ...
        2/    ...
        3/    ...

If the dataset is not yet downloaded, use a mock dataset for development.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


CLASS_NAMES = ["1++", "1+", "1", "2", "3"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def get_transforms(image_size: int = 512, split: str = "train") -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class BeefGradeDataset(Dataset):
    """Loads beef grade images from a class-per-folder directory structure."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        class_names: list = CLASS_NAMES,
    ):
        self.root = Path(root)
        self.transform = transform
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        self.samples = []
        for cls in class_names:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found under '{root}'. "
                "Download the AI-Hub beef carcass dataset and place it at the configured path."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_pil_image(self, idx: int) -> Image.Image:
        """Return the raw PIL image (for perturbation pipeline)."""
        img_path, _ = self.samples[idx]
        return Image.open(img_path).convert("RGB")


def get_dataloaders(
    root: str,
    image_size: int = 512,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
):
    full_dataset = BeefGradeDataset(root, transform=None)
    n = len(full_dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=__import__("torch").Generator().manual_seed(seed),
    )

    def make_loader(subset, split):
        tf = get_transforms(image_size, split)
        subset.dataset.transform = tf
        return DataLoader(subset, batch_size=batch_size, shuffle=(split == "train"),
                          num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_set, "train"),
        make_loader(val_set, "val"),
        make_loader(test_set, "test"),
    )
