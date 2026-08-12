#!/usr/bin/env python3
"""Build upper_40x20 data from src/data/mld_rho_no_speed arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_DIR / "src/data/mld_rho_no_speed"
OUT_DIR = Path(__file__).resolve().parent / "data"
FEATURES = ["U", "V", "W", "zeta", "strain", "T", "rho", "kpphbl"]
Y_SLICE = slice(30, 50)
X_SLICE = slice(10, 50)


def main() -> None:
    x = np.load(SOURCE_DIR / "X.npy", mmap_mode="r")
    y = np.load(SOURCE_DIR / "y.npy", mmap_mode="r")
    time_index = np.load(SOURCE_DIR / "time_index.npy", mmap_mode="r")

    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected X/y 4-D arrays, got X={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[0] != time_index.shape[0]:
        raise ValueError(f"Sample mismatch: X={x.shape}, y={y.shape}, time={time_index.shape}")
    if x.shape[1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} channels, got {x.shape[1]}")

    x_upper = np.asarray(x[:, :, Y_SLICE, X_SLICE], dtype=np.float32)
    y_upper = np.asarray(y[:, :, Y_SLICE, X_SLICE], dtype=np.float32)
    if x_upper.shape[-2:] != (20, 40) or y_upper.shape[-2:] != (20, 40):
        raise ValueError(f"Expected upper_40x20 shape, got X={x_upper.shape}, y={y_upper.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X.npy", x_upper)
    np.save(OUT_DIR / "y.npy", y_upper)
    np.save(OUT_DIR / "time_index.npy", np.asarray(time_index, dtype=np.int64))

    metadata = {
        "source_dir": str(SOURCE_DIR),
        "features": FEATURES,
        "target": "wT",
        "domain": "upper_40x20",
        "y_start": Y_SLICE.start,
        "y_stop_exclusive": Y_SLICE.stop,
        "x_start": X_SLICE.start,
        "x_stop_exclusive": X_SLICE.stop,
        "shape_X": list(x_upper.shape),
        "shape_y": list(y_upper.shape),
        "time_count": int(time_index.shape[0]),
        "time_start": int(time_index[0]),
        "time_end": int(time_index[-1]),
        "nan_X": int(np.isnan(x_upper).sum()),
        "nan_y": int(np.isnan(y_upper).sum()),
    }
    with (OUT_DIR / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Saved X: {OUT_DIR / 'X.npy'} {x_upper.shape}")
    print(f"Saved y: {OUT_DIR / 'y.npy'} {y_upper.shape}")
    print(f"Saved time_index: {OUT_DIR / 'time_index.npy'} {time_index.shape}")


if __name__ == "__main__":
    main()
