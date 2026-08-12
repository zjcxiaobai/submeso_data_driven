#!/usr/bin/env python3
"""Save the exact 80/10/10 split indices used by the rho upper40x20 model.

This reproduces the same split logic as _src/data_utils/npy_dataset.py and
checks that the reproduced test indices match the saved model evaluation file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "analysis/shuffle_upper40x20_mld_rho/data"
PREDICTIONS = PROJECT_DIR / "analysis/shuffle_upper40x20_mld_rho/outputs_split_80_10_10/test_predictions_denorm.npz"
OUT_NPZ = ANALYSIS_DIR / "split_indices.npz"
OUT_SUMMARY = ANALYSIS_DIR / "split_indices_summary.txt"

SEED = 42
TEMPORAL_OFFSETS = [-8, -7, -6, -5, -4, -3, -2, -1, 0]
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1, "shuffle": True, "shuffle_train_val": False}


def valid_center_indices(n_samples: int, temporal_offsets: list[int]) -> np.ndarray:
    min_offset = min(temporal_offsets)
    max_offset = max(temporal_offsets)
    start = max(0, -min_offset)
    stop = n_samples - max(0, max_offset)
    if stop <= start:
        raise ValueError(f"No valid samples for n_samples={n_samples} and temporal_offsets={temporal_offsets}")
    return np.arange(start, stop, dtype=np.int64)


def split_indices(n_samples: int, split: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    indices = np.arange(n_samples, dtype=np.int64)
    if bool(split.get("shuffle", False)):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
        return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
    train_val_idx = indices[:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if bool(split.get("shuffle_train_val", False)):
        rng = np.random.default_rng(seed)
        train_val_idx = rng.permutation(train_val_idx)
    return train_val_idx[:n_train], train_val_idx[n_train:n_train + n_val], test_idx


def main() -> None:
    y = np.load(DATA_DIR / "y.npy", mmap_mode="r")
    time_index = np.load(DATA_DIR / "time_index.npy", mmap_mode="r")
    pred = np.load(PREDICTIONS)
    saved_test_sample = pred["sample_index"].astype(np.int64)

    centers = valid_center_indices(int(y.shape[0]), TEMPORAL_OFFSETS)
    train_pos, val_pos, test_pos = split_indices(centers.shape[0], SPLIT, SEED)
    train_sample = centers[train_pos]
    val_sample = centers[val_pos]
    test_sample = centers[test_pos]

    if not np.array_equal(test_sample, saved_test_sample):
        raise RuntimeError(
            "Reproduced test sample indices do not match outputs_split_80_10_10/test_predictions_denorm.npz. "
            f"First reproduced={test_sample[:5].tolist()}, saved={saved_test_sample[:5].tolist()}"
        )

    train_time = np.asarray(time_index[train_sample], dtype=np.int64)
    val_time = np.asarray(time_index[val_sample], dtype=np.int64)
    test_time = np.asarray(time_index[test_sample], dtype=np.int64)

    np.savez_compressed(
        OUT_NPZ,
        centers=centers.astype(np.int64),
        train_sample_index=train_sample.astype(np.int64),
        val_sample_index=val_sample.astype(np.int64),
        test_sample_index=test_sample.astype(np.int64),
        train_time_index=train_time,
        val_time_index=val_time,
        test_time_index=test_time,
        temporal_offsets=np.asarray(TEMPORAL_OFFSETS, dtype=np.int64),
        seed=np.asarray([SEED], dtype=np.int64),
        split_train=np.asarray([SPLIT["train"]], dtype=np.float64),
        split_val=np.asarray([SPLIT["val"]], dtype=np.float64),
        split_test=np.asarray([SPLIT["test"]], dtype=np.float64),
        split_shuffle=np.asarray([SPLIT["shuffle"]], dtype=bool),
        source_y_path=np.asarray([str(DATA_DIR / "y.npy")]),
        source_time_path=np.asarray([str(DATA_DIR / "time_index.npy")]),
        checked_predictions_path=np.asarray([str(PREDICTIONS)]),
    )

    summary = {
        "source_y": str(DATA_DIR / "y.npy"),
        "source_time_index": str(DATA_DIR / "time_index.npy"),
        "checked_predictions": str(PREDICTIONS),
        "n_total_samples": int(y.shape[0]),
        "n_valid_centers": int(centers.size),
        "n_train": int(train_sample.size),
        "n_val": int(val_sample.size),
        "n_test": int(test_sample.size),
        "train_sample_min": int(train_sample.min()),
        "train_sample_max": int(train_sample.max()),
        "val_sample_min": int(val_sample.min()),
        "val_sample_max": int(val_sample.max()),
        "test_sample_min": int(test_sample.min()),
        "test_sample_max": int(test_sample.max()),
        "test_order_matches_saved_predictions": True,
    }
    with OUT_SUMMARY.open("w") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved {OUT_NPZ}")
    print(f"Saved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
