from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from tlc.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    letterbox_image,
    load_image,
)
from train.augment import get_train_augmentation

LABEL_MAP = {"not_red": 0.0, "red": 1.0}

class TrafficLightDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        is_train: bool = False,
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.csv_path = Path(csv_path)
        self.is_train = is_train
        self.target_size = target_size
        self.augment = get_train_augmentation() if is_train else None
        self.records: list[dict[str, str]] = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[idx]
        img_path = rec["filepath"]
        label_val = LABEL_MAP[rec["label"]]

        pil_img = load_image(img_path)
        if self.is_train and self.augment is not None:
            pil_img = self.augment(pil_img)

        letterboxed = letterbox_image(pil_img, self.target_size)
        arr = np.asarray(letterboxed, dtype=np.float32)
        arr = arr.transpose(2, 0, 1) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

        tensor_x = torch.from_numpy(arr).float()
        tensor_y = torch.tensor([label_val], dtype=torch.float32)
        return tensor_x, tensor_y, img_path

def get_class_counts(records: list[dict[str, str]]) -> dict[str, int]:
    counts = {"not_red": 0, "red": 0}
    for r in records:
        lbl = r["label"]
        if lbl in counts:
            counts[lbl] += 1
    return counts

def compute_pos_weight(dataset: TrafficLightDataset) -> torch.Tensor:
    counts = get_class_counts(dataset.records)
    neg_count = counts["not_red"]
    pos_count = max(1, counts["red"])
    weight = float(neg_count) / float(pos_count)
    return torch.tensor([weight], dtype=torch.float32)

def create_weighted_sampler(dataset: TrafficLightDataset) -> WeightedRandomSampler:
    counts = get_class_counts(dataset.records)
    neg_count = max(1, counts["not_red"])
    pos_count = max(1, counts["red"])
    class_weights = {
        "not_red": 1.0 / neg_count,
        "red": 1.0 / pos_count,
    }
    sample_weights = [class_weights[r["label"]] for r in dataset.records]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
