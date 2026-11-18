"""Temporal datasets, time-block splits, and normalization statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .config import project_dir, resolve


@dataclass
class NormStats:
    """Channel-wise statistics fitted using the training time block."""

    x_mean: np.ndarray  # (C, 1, 1)
    x_std: np.ndarray  # (C, 1, 1)
    y_mean: np.ndarray  # (1, 1, 1)
    y_std: np.ndarray  # (1, 1, 1)


class TemporalSequenceDataset(Dataset):
    """Construct temporal input sequences lazily from memory-mapped arrays."""

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
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y sample count differs: {self.x.shape[0]} vs {self.y.shape[0]}")
        if self.x.ndim != 4 or self.y.ndim != 4:
            raise ValueError(f"Expected X/y as 4D arrays, got X={self.x.shape}, y={self.y.shape}")
        self.temporal_offsets = temporal_offsets
        self.stats = stats
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = np.stack(
            [np.asarray(self.x[index + offset], dtype=np.float32) for offset in self.temporal_offsets],
            axis=0,
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
    """Partition the raw chronology before constructing temporal windows."""
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    test_frac = float(split.get("test", 1.0 - train_frac - val_frac))
    if abs(train_frac + val_frac + test_frac - 1.0) > 1.0e-6:
        raise ValueError("Split fractions must sum to 1.0")
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("Invalid split fractions")
    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    if n_train <= 0 or n_train + n_val >= n_samples:
        raise ValueError("Split fractions leave an empty train or test block")
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_samples)


def centers_in_block(block: tuple[int, int], offsets: list[int]) -> np.ndarray:
    """Return target times whose complete input sequence lies in one block."""
    block_start, block_stop = block
    start = max(block_start, block_start - min(offsets))
    stop = min(block_stop, block_stop - max(offsets))
    if stop <= start:
        raise ValueError(f"No valid centers for block={block}, offsets={offsets}")
    return np.arange(start, stop, dtype=np.int64)


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
    x_sum = np.zeros((x.shape[1],), dtype=np.float64)
    x_sum2 = np.zeros((x.shape[1],), dtype=np.float64)
    y_sum = np.zeros((y.shape[1],), dtype=np.float64)
    y_sum2 = np.zeros((y.shape[1],), dtype=np.float64)
    x_count = 0
    y_count = 0
    for idx in train_idx:
        for offset in offsets:
            sample_x = np.asarray(x[int(idx) + offset], dtype=np.float64)
            x_sum += sample_x.sum(axis=(1, 2))
            x_sum2 += np.square(sample_x).sum(axis=(1, 2))
            x_count += sample_x.shape[1] * sample_x.shape[2]
        sample_y = np.asarray(y[int(idx)], dtype=np.float64)
        y_sum += sample_y.sum(axis=(1, 2))
        y_sum2 += np.square(sample_y).sum(axis=(1, 2))
        y_count += sample_y.shape[1] * sample_y.shape[2]
    x_mean = x_sum / x_count
    x_std = np.sqrt(np.maximum(x_sum2 / x_count - x_mean**2, eps**2))
    y_mean = y_sum / y_count
    y_std = np.sqrt(np.maximum(y_sum2 / y_count - y_mean**2, eps**2))
    return NormStats(
        x_mean=x_mean.astype(np.float32)[:, None, None],
        x_std=x_std.astype(np.float32)[:, None, None],
        y_mean=y_mean.astype(np.float32)[:, None, None],
        y_std=y_std.astype(np.float32)[:, None, None],
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
    data = np.load(path)
    return NormStats(
        x_mean=data["x_mean"],
        x_std=data["x_std"],
        y_mean=data["y_mean"],
        y_std=data["y_std"],
    )


def make_loaders(config: dict, *, recompute_stats: bool = False):
    """Build train, validation, and test loaders from raw chronological blocks."""
    root = project_dir()
    data_cfg = config["data"]
    x_path = resolve(root, data_cfg["x_path"])
    y_path = resolve(root, data_cfg["y_path"])
    offsets = [int(value) for value in data_cfg.get("temporal_offsets", [0])]
    n_samples = np.load(x_path, mmap_mode="r").shape[0]
    blocks = split_raw_blocks(n_samples, data_cfg["split"])
    train_idx, val_idx, test_idx = [centers_in_block(block, offsets) for block in blocks]
    stats_path = resolve(
        root,
        config["output"].get("stats", "analysis/temporal_bottleneck/outputs/norm_stats.npz"),
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
