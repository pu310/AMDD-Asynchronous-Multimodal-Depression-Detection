"""D-Vlog dataset loader for multimodal depression detection."""

from pathlib import Path
from typing import Union
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def normalize_data(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.clip(std, a_min=1e-8, a_max=None)
    return (x - mean) / std


class DVlog(data.Dataset):
    def __init__(
        self, root: Union[str, Path], fold: str = "train",
        gender: str = "both", transform=None, target_transform=None
    ):
        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.gender = gender
        self.transform = transform
        self.target_transform = target_transform

        self.features = []
        self.labels = []
        with open(self.root / "labels.csv", "r") as f:
            for line in f:
                sample = line.strip().split(",")
                if self.is_sample(sample):
                    s_id = sample[0]
                    s_label = int(sample[1] == "depression")
                    self.labels.append(s_label)

                    v_feature_path = self.root / s_id / f"{s_id}_visual.npy"
                    a_feature_path = self.root / s_id / f"{s_id}_acoustic.npy"
                    v_feature = np.load(v_feature_path)
                    a_feature = np.load(a_feature_path)

                    T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                    if T_v == T_a:
                        feature = np.concatenate(
                            (v_feature, a_feature), axis=1
                        ).astype(np.float32)
                    else:
                        T = min(T_v, T_a)
                        feature = np.concatenate(
                            (v_feature[:T], a_feature[:T]), axis=1
                        ).astype(np.float32)
                    feature = normalize_data(feature)
                    self.features.append(feature)

    def is_sample(self, sample) -> bool:
        gender, fold = sample[3], sample[4]
        if self.gender == "both":
            return fold == self.fold
        return (fold == self.fold) and (gender == self.gender)

    def __getitem__(self, i: int):
        feature = self.features[i]
        label = self.labels[i]
        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return feature, label

    def __len__(self):
        return len(self.labels)


def _collate_fn(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features], batch_first=True
    )
    labels = torch.tensor(labels)
    return padded_features, labels


def get_dvlog_dataloader(
    root: Union[str, Path], fold: str = "train", batch_size: int = 8,
    gender: str = "both", transform=None, target_transform=None,
):
    dataset = DVlog(root, fold, gender, transform, target_transform)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=(fold == "train"),
    )
    return dataloader
