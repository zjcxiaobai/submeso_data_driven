"""Dataset loading, chronological splits, and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import PROJECT_ROOT


@dataclass
class NormStats:
    """Channel-wise normalization statistics."""

    x_mean: np.ndarray  # (channels, 1, 1)
    x_std: np.ndarray
    y_mean: np.ndarray  # (1, 1, 1)
    y_std: np.ndarray


class TemporalSequenceDataset(Dataset):
    """Build temporal sequences."""

    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        *,
        temporal_offsets: list[int],
        stats: NormStats,
        normalize_x: bool,
        normalize_y: bool,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.temporal_offsets = np.asarray(temporal_offsets, dtype=np.int64)
        self.stats = stats
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

        if self.x.ndim != 4 or self.y.ndim != 4:
            raise ValueError(f"Expected 4-D X and y arrays, got {self.x.shape} and {self.y.shape}")
        if len(self.x) != len(self.y):
            raise ValueError(f"X and y contain different sample counts: {len(self.x)} and {len(self.y)}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = np.asarray(
            self.x[index + self.temporal_offsets],
            dtype=np.float32,
        )
        target = np.asarray(self.y[index], dtype=np.float32)

        if self.normalize_x:
            sequence = (sequence - self.stats.x_mean[None, ...]) / self.stats.x_std[None, ...]
        if self.normalize_y:
            target = (target - self.stats.y_mean) / self.stats.y_std

        return torch.from_numpy(sequence.copy()), torch.from_numpy(target.copy())


def split_raw_blocks(
    n_samples: int,
    split: dict,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Split the raw chronology before temporal windows are constructed."""
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    test_frac = float(split.get("test", 1.0 - train_frac - val_frac))

    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))

    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_samples)


def centers_in_block(block: tuple[int, int], offsets: list[int]) -> np.ndarray:
    """Return target indices whose complete sequence stays inside a block."""
    block_start, block_stop = block
    offsets = np.asarray(offsets, dtype=np.int64)
    start = max(block_start, block_start - int(offsets.min()))
    stop = min(block_stop, block_stop - int(offsets.max()))

    if stop <= start:
        raise ValueError(f"No valid centers for block={block}, offsets={offsets}")

    return np.arange(start, stop, dtype=np.int64)


def _channel_stats(
    array: np.ndarray,
    indices: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and standard deviation of each channel."""
    total = np.zeros(array.shape[1], dtype=np.float64)
    total_squared = np.zeros_like(total)
    count = 0

    for index in indices:
        sample = np.asarray(array[int(index)], dtype=np.float64)
        total += sample.sum(axis=(1, 2))
        total_squared += np.square(sample).sum(axis=(1, 2))
        count += sample.shape[1] * sample.shape[2]

    mean = total / count
    variance = np.maximum(total_squared / count - mean**2, eps**2)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def compute_stats(
    x_path: Path,
    y_path: Path,
    train_idx: np.ndarray,
    offsets: list[int],
    eps: float = 1.0e-6,
) -> NormStats:
    """Fit channel-wise normalization using training sequences only."""
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    offsets = np.asarray(offsets, dtype=np.int64)
    x_indices = (train_idx[:, None] + offsets[None, :]).reshape(-1)

    x_mean, x_std = _channel_stats(x, x_indices, eps)
    y_mean, y_std = _channel_stats(y, train_idx, eps)

    return NormStats(
        x_mean=x_mean[:, None, None],
        x_std=x_std[:, None, None],
        y_mean=y_mean[:, None, None],
        y_std=y_std[:, None, None],
    )


def save_stats(path: Path, stats: NormStats, offsets: list[int]) -> None:
    """Save normalization statistics and their temporal offsets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x_mean=stats.x_mean,
        x_std=stats.x_std,
        y_mean=stats.y_mean,
        y_std=stats.y_std,
        temporal_offsets=np.asarray(offsets, dtype=np.int64),
    )


def load_stats(path: Path) -> NormStats:
    """Load saved normalization statistics."""
    with np.load(path) as data:
        return NormStats(
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            y_mean=data["y_mean"],
            y_std=data["y_std"],
        )


def make_loaders(config: dict, *, recompute_stats: bool = False):
    """Build train, validation, and test data loaders."""
    data_cfg = config["data"]
    x_path = PROJECT_ROOT / data_cfg["x_path"]
    y_path = PROJECT_ROOT / data_cfg["y_path"]
    offsets = [int(value) for value in data_cfg.get("temporal_offsets", [0])]

    n_samples = np.load(x_path, mmap_mode="r").shape[0]
    blocks = split_raw_blocks(n_samples, data_cfg["split"])
    train_idx, val_idx, test_idx = [
        centers_in_block(block, offsets)
        for block in blocks
    ]

    stats_path = PROJECT_ROOT / config["output"].get(
        "stats",
        "analysis/temporal_bottleneck/outputs/norm_stats.npz",
    )

    if recompute_stats or not stats_path.exists():
        stats = compute_stats(x_path, y_path, train_idx, offsets)
        save_stats(stats_path, stats, offsets)
    else:
        stats = load_stats(stats_path)

    dataset = TemporalSequenceDataset(
        x_path,
        y_path,
        temporal_offsets=offsets,
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=bool(data_cfg.get("normalize_y", True)),
    )

    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 128)),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }

    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, stats
