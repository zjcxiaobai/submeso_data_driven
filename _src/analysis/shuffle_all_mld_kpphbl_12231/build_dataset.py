#!/usr/bin/env python3
"""Build the expanded mld_kpphbl dataset for the shuffle-all rerun.

The channel order intentionally matches the original eight-feature model:
U, V, W, zeta, strain, T, speed, kpphbl. The target is wT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "data/mld"
OUTPUT_DIR = HERE / "data"

FEATURES = ("U", "V", "W", "zeta", "strain", "T", "speed", "kpphbl")
TARGET = "wT"


def load_npz_var(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = SOURCE_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as npz:
        if "time_index" not in npz.files or "data" not in npz.files:
            raise ValueError(f"{path} must contain time_index and data arrays")
        time = np.asarray(npz["time_index"], dtype=np.int64)
        data = np.asarray(npz["data"], dtype=np.float32)
    if time.ndim != 1 or data.shape[0] != time.size:
        raise ValueError(f"Invalid shapes in {path}: time={time.shape}, data={data.shape}")
    return time, data


def align_times(times: dict[str, np.ndarray]) -> np.ndarray:
    common = set(map(int, times[FEATURES[0]]))
    for name in (*FEATURES[1:], TARGET):
        common &= set(map(int, times[name]))
    if not common:
        raise ValueError("No common times across features and target")
    return np.asarray(sorted(common), dtype=np.int64)


def take_by_time(time: np.ndarray, data: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    index = {int(t): i for i, t in enumerate(time)}
    indices = np.asarray([index[int(t)] for t in target_time], dtype=np.int64)
    return data[indices]


def main() -> None:
    times: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {}
    for name in (*FEATURES, TARGET):
        times[name], arrays[name] = load_npz_var(name)

    final_time = align_times(times)
    stacked_features = [take_by_time(times[name], arrays[name], final_time) for name in FEATURES]
    target = take_by_time(times[TARGET], arrays[TARGET], final_time)

    spatial_shape = stacked_features[0].shape[1:]
    for name, arr in zip(FEATURES, stacked_features):
        if arr.shape[1:] != spatial_shape:
            raise ValueError(f"Spatial shape mismatch for {name}: {arr.shape[1:]} vs {spatial_shape}")
    if target.shape[1:] != spatial_shape:
        raise ValueError(f"Target spatial shape mismatch: {target.shape[1:]} vs {spatial_shape}")

    x = np.stack(stacked_features, axis=1).astype(np.float32, copy=False)
    y = np.expand_dims(target, axis=1).astype(np.float32, copy=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", x)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "time_index.npy", final_time)

    metadata = {
        "source_dir": str(SOURCE_DIR),
        "features": list(FEATURES),
        "target": TARGET,
        "shape_X": list(x.shape),
        "shape_y": list(y.shape),
        "time_count": int(final_time.size),
        "time_start": int(final_time[0]),
        "time_end": int(final_time[-1]),
        "time_step_first": int(final_time[1] - final_time[0]) if final_time.size > 1 else None,
        "nan_X": int(np.isnan(x).sum()),
        "nan_y": int(np.isnan(y).sum()),
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Saved {OUTPUT_DIR / 'X.npy'} {x.shape}", flush=True)
    print(f"Saved {OUTPUT_DIR / 'y.npy'} {y.shape}", flush=True)
    print(f"Saved {OUTPUT_DIR / 'time_index.npy'} count={final_time.size}", flush=True)


if __name__ == "__main__":
    main()
