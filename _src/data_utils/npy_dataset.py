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
        temporal_offsets: list[int] | None = None,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y sample count differs: {self.x.shape[0]} vs {self.y.shape[0]}")
        if self.x.ndim != 4 or self.y.ndim != 4:
            raise ValueError(f"Expected channel-first 4D arrays, got X={self.x.shape}, y={self.y.shape}")

        self.temporal_offsets = temporal_offsets or [0]
        if not self.temporal_offsets:
            raise ValueError("temporal_offsets must not be empty")

        self.stats = stats
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temporal_offsets == [0]:
            x = np.asarray(self.x[index], dtype=np.float32)
        else:
            frames = [np.asarray(self.x[index + offset], dtype=np.float32) for offset in self.temporal_offsets]
            x = np.concatenate(frames, axis=0)
        y = np.asarray(self.y[index], dtype=np.float32)

        if self.stats is not None and self.normalize_x:
            x = (x - self.stats.x_mean) / self.stats.x_std
        if self.stats is not None and self.normalize_y:
            y = (y - self.stats.y_mean) / self.stats.y_std

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


class RandomXRollDataset(Dataset):
    def __init__(self, dataset: Dataset, *, generator: torch.Generator | None = None) -> None:
        self.dataset = dataset
        self.generator = generator

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[index]
        width = x.shape[-1]
        shift = int(torch.randint(width, (1,), generator=self.generator).item())
        if shift == 0:
            return x, y
        return torch.roll(x, shifts=shift, dims=-1), torch.roll(y, shifts=shift, dims=-1)


def split_indices(n_samples: int, split: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    shuffle = bool(split.get("shuffle", False))
    shuffle_train_val = bool(split.get("shuffle_train_val", False))
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0:
        raise ValueError("Invalid split fractions")

    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    indices = np.arange(n_samples, dtype=np.int64)

    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        return train_idx, val_idx, test_idx

    train_val_idx = indices[: n_train + n_val]
    test_idx = indices[n_train + n_val :]
    if shuffle_train_val:
        rng = np.random.default_rng(seed)
        train_val_idx = rng.permutation(train_val_idx)

    train_idx = train_val_idx[:n_train]
    val_idx = train_val_idx[n_train : n_train + n_val]
    return train_idx, val_idx, test_idx


def split_raw_time_blocks(n_samples: int, split: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    test_frac = float(split.get("test", 1.0 - train_frac - val_frac))
    if abs(train_frac + val_frac + test_frac - 1.0) > 1.0e-6:
        raise ValueError("Split fractions must sum to 1.0")
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("Invalid split fractions")

    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    if n_train <= 0 or n_val < 0 or n_train + n_val >= n_samples:
        raise ValueError("Split fractions leave an empty train or test block")

    train_block = (0, n_train)
    val_block = (n_train, n_train + n_val)
    test_block = (n_train + n_val, n_samples)
    return train_block, val_block, test_block


def valid_center_indices(n_samples: int, temporal_offsets: list[int]) -> np.ndarray:
    min_offset = min(temporal_offsets)
    max_offset = max(temporal_offsets)
    start = max(0, -min_offset)
    stop = n_samples - max(0, max_offset)
    if stop <= start:
        raise ValueError(
            f"No valid samples for n_samples={n_samples} and temporal_offsets={temporal_offsets}"
        )
    return np.arange(start, stop, dtype=np.int64)


def valid_center_indices_for_block(
    block: tuple[int, int],
    temporal_offsets: list[int],
) -> np.ndarray:
    block_start, block_stop = block
    min_offset = min(temporal_offsets)
    max_offset = max(temporal_offsets)
    start = max(block_start, block_start - min_offset)
    stop = min(block_stop, block_stop - max_offset)
    if stop <= start:
        raise ValueError(
            "No valid samples for raw time block "
            f"{block} and temporal_offsets={temporal_offsets}"
        )
    return np.arange(start, stop, dtype=np.int64)


def parse_temporal_offsets(data_cfg: dict) -> list[int]:
    offsets = data_cfg.get("temporal_offsets", [0])
    if offsets is None:
        offsets = [0]
    if isinstance(offsets, int):
        offsets = [offsets]
    offsets = [int(offset) for offset in offsets]
    if not offsets:
        raise ValueError("data.temporal_offsets must not be empty")
    return offsets


def compute_channel_stats(
    arr: np.ndarray,
    indices: np.ndarray,
    temporal_offsets: list[int] | None = None,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    temporal_offsets = temporal_offsets or [0]
    n_channels = arr.shape[1] * len(temporal_offsets)
    total = np.zeros((n_channels,), dtype=np.float64)
    total_sq = np.zeros((n_channels,), dtype=np.float64)
    count = 0

    for idx in indices:
        if temporal_offsets == [0]:
            sample = np.asarray(arr[int(idx)], dtype=np.float64)
        else:
            frames = [np.asarray(arr[int(idx) + offset], dtype=np.float64) for offset in temporal_offsets]
            sample = np.concatenate(frames, axis=0)
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
    stats_idx: np.ndarray,
    stats_path: Path,
    temporal_offsets: list[int] | None = None,
    stats_scope: str = "train",
) -> NormStats:
    temporal_offsets = temporal_offsets or [0]
    stats_scope = stats_scope.lower()
    if stats_scope not in {"train", "all"}:
        raise ValueError(f"data.stats_scope must be 'train' or 'all', got {stats_scope!r}")

    expected_x_channels = np.load(x_path, mmap_mode="r").shape[1] * len(temporal_offsets)
    if stats_path.exists():
        data = np.load(stats_path)
        saved_offsets = data["temporal_offsets"].tolist() if "temporal_offsets" in data.files else [0]
        saved_scope = str(data["stats_scope"].item()) if "stats_scope" in data.files else "train"
        if (
            data["x_mean"].shape[0] == expected_x_channels
            and saved_offsets == temporal_offsets
            and saved_scope == stats_scope
        ):
            return NormStats(
                x_mean=data["x_mean"],
                x_std=data["x_std"],
                y_mean=data["y_mean"],
                y_std=data["y_std"],
            )

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    x_mean, x_std = compute_channel_stats(x, stats_idx, temporal_offsets=temporal_offsets)
    y_mean, y_std = compute_channel_stats(y, stats_idx, temporal_offsets=[0])
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stats_path,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        temporal_offsets=np.asarray(temporal_offsets, dtype=np.int64),
        stats_scope=np.asarray(stats_scope),
    )
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
    temporal_offsets = parse_temporal_offsets(data_cfg)
    split_cfg = data_cfg["split"]
    split_method = str(split_cfg.get("method", "center_indices")).lower()
    if split_method in {"raw_time_blocks", "time_blocks", "raw_blocks"}:
        if bool(split_cfg.get("shuffle", False)) or bool(split_cfg.get("shuffle_train_val", False)):
            raise ValueError("raw_time_blocks split does not support shuffling")
        train_block, val_block, test_block = split_raw_time_blocks(probe.shape[0], split_cfg)
        train_idx = valid_center_indices_for_block(train_block, temporal_offsets)
        val_idx = valid_center_indices_for_block(val_block, temporal_offsets)
        test_idx = valid_center_indices_for_block(test_block, temporal_offsets)
        centers = np.concatenate([train_idx, val_idx, test_idx])
    elif split_method in {"center_indices", "valid_centers"}:
        centers = valid_center_indices(probe.shape[0], temporal_offsets)
        train_pos, val_pos, test_pos = split_indices(centers.shape[0], split_cfg, seed)
        train_idx = centers[train_pos]
        val_idx = centers[val_pos]
        test_idx = centers[test_pos]
    else:
        raise ValueError(
            "data.split.method must be 'raw_time_blocks' or 'center_indices', "
            f"got {split_method!r}"
        )
    stats_scope = str(data_cfg.get("stats_scope", "train")).lower()
    if stats_scope == "train":
        stats_idx = train_idx
    elif stats_scope == "all":
        stats_idx = centers
    else:
        raise ValueError(f"data.stats_scope must be 'train' or 'all', got {stats_scope!r}")
    stats = load_or_compute_stats(
        x_path,
        y_path,
        stats_idx,
        stats_path,
        temporal_offsets=temporal_offsets,
        stats_scope=stats_scope,
    )
    dataset = NpyRegressionDataset(
        x_path,
        y_path,
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=bool(data_cfg.get("normalize_y", True)),
        temporal_offsets=temporal_offsets,
    )

    loader_kwargs = dict(
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    augment_x_roll = bool(data_cfg.get("augment_x_roll", False))
    if augment_x_roll:
        roll_generator = torch.Generator().manual_seed(seed + 1009)
        train_subset = RandomXRollDataset(train_subset, generator=roll_generator)

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
