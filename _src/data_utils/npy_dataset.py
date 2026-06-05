from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class NormStats:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray


class NpyRegressionDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        *,
        stats: NormStats | None = None,
        normalize_x: bool = True,
        normalize_y: bool = True,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y sample count differs: {self.x.shape[0]} vs {self.y.shape[0]}")
        if self.x.ndim != 4 or self.y.ndim != 4:
            raise ValueError(f"Expected channel-first 4D arrays, got X={self.x.shape}, y={self.y.shape}")

        self.stats = stats
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.x[index], dtype=np.float32)
        y = np.asarray(self.y[index], dtype=np.float32)

        if self.stats is not None and self.normalize_x:
            x = (x - self.stats.x_mean) / self.stats.x_std
        if self.stats is not None and self.normalize_y:
            y = (y - self.stats.y_mean) / self.stats.y_std

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def split_indices(n_samples: int, split: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    shuffle = bool(split.get("shuffle", False))
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0:
        raise ValueError("Invalid split fractions")

    indices = np.arange(n_samples, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)

    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def compute_channel_stats(arr: np.ndarray, indices: np.ndarray, eps: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    n_channels = arr.shape[1]
    total = np.zeros((n_channels,), dtype=np.float64)
    total_sq = np.zeros((n_channels,), dtype=np.float64)
    count = 0

    for idx in indices:
        sample = np.asarray(arr[int(idx)], dtype=np.float64)
        total += sample.sum(axis=(1, 2))
        total_sq += np.square(sample).sum(axis=(1, 2))
        count += sample.shape[1] * sample.shape[2]

    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), eps**2)
    std = np.sqrt(var)
    return mean.astype(np.float32)[:, None, None], std.astype(np.float32)[:, None, None]


def load_or_compute_stats(
    x_path: Path,
    y_path: Path,
    train_idx: np.ndarray,
    stats_path: Path,
) -> NormStats:
    if stats_path.exists():
        data = np.load(stats_path)
        return NormStats(
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            y_mean=data["y_mean"],
            y_std=data["y_std"],
        )

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    x_mean, x_std = compute_channel_stats(x, train_idx)
    y_mean, y_std = compute_channel_stats(y, train_idx)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return NormStats(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)


def build_dataloaders(
    config: dict,
    project_dir: Path,
    seed: int,
    *,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, NormStats]:
    data_cfg = config["data"]
    x_path = project_dir / data_cfg["x_path"]
    y_path = project_dir / data_cfg["y_path"]
    stats_path = project_dir / data_cfg.get("stats_path", "outputs/norm_stats.npz")

    probe = np.load(x_path, mmap_mode="r")
    train_idx, val_idx, test_idx = split_indices(probe.shape[0], data_cfg["split"], seed)
    stats = load_or_compute_stats(x_path, y_path, train_idx, stats_path)
    dataset = NpyRegressionDataset(
        x_path,
        y_path,
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=bool(data_cfg.get("normalize_y", True)),
    )

    loader_kwargs = dict(
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    shuffle_train = bool(data_cfg.get("shuffle_train_loader", False))
    train_sampler = DistributedSampler(train_subset, shuffle=shuffle_train, seed=seed) if distributed else None
    val_sampler = DistributedSampler(val_subset, shuffle=False, seed=seed) if distributed else None
    test_sampler = DistributedSampler(test_subset, shuffle=False, seed=seed) if distributed else None
    generator = None if distributed else torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset,
        shuffle=(train_sampler is None and shuffle_train),
        sampler=train_sampler,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_subset, shuffle=False, sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, sampler=test_sampler, **loader_kwargs)
    return train_loader, val_loader, test_loader, stats
