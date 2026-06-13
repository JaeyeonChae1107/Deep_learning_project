from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Folder name → class index mapping (0=1++, 1=1+, 2=1, 3=2, 4=3)
FOLDER_TO_IDX: dict[str, int] = {
    "grade_1pp": 0,
    "grade_1p":  1,
    "grade_1":   2,
    "grade_2":   3,
    "grade_3":   4,
}


def infer_num_classes(data_dir: Path | str) -> int:
    return sum(1 for d in Path(data_dir).iterdir() if d.is_dir())


def describe_dataset(data_dir: Path | str) -> str:
    data_dir = Path(data_dir)
    total = 0
    n_classes = 0
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        n_classes += 1
        total += len(list(subdir.glob("*.jpg")))
    return f"{total} images, {n_classes} classes ({data_dir.name})"


class _BeefDataset(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        transform: transforms.Compose,
        consistency: bool = False,
        perturbation_fn=None,
    ) -> None:
        self.transform = transform
        self.consistency = consistency
        self.perturbation_fn = perturbation_fn
        self.samples: list[tuple[Path, int]] = []

        data_dir = Path(data_dir)
        for folder_name, class_idx in FOLDER_TO_IDX.items():
            folder = data_dir / folder_name
            if not folder.exists():
                continue
            for img_path in sorted(folder.glob("*.jpg")):
                self.samples.append((img_path, class_idx))

        if not self.samples:
            raise RuntimeError(f"No .jpg images found under {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        if self.consistency and self.perturbation_fn is not None:
            return x, self.perturbation_fn(x), label
        return x, label


def make_loader(
    data_dir: Path | str,
    batch_size: int,
    image_size: int,
    shuffle: bool,
    num_workers: int,
    consistency: bool = False,
    perturbation_names: list[str] | None = None,
) -> DataLoader:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if shuffle:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    perturbation_fn = None
    if consistency:
        from beef_cbam.perturbations import RandomPerturbation
        perturbation_fn = RandomPerturbation(perturbation_names)

    dataset = _BeefDataset(data_dir, transform, consistency=consistency, perturbation_fn=perturbation_fn)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
