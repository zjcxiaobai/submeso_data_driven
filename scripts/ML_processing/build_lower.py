"""Crop the lower 20 x 40 subdomain from the full-domain dataset."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data/interim/full_domain"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/lower"

Y_SLICE = slice(10, 30)
X_SLICE = slice(10, 50)


def main():
    X = np.load(INPUT_DIR / "X.npy", mmap_mode="r")
    y = np.load(INPUT_DIR / "y.npy", mmap_mode="r")
    time_index = np.load(INPUT_DIR / "time_index.npy")

    X = np.asarray(X[:, :, Y_SLICE, X_SLICE], dtype=np.float32)
    y = np.asarray(y[:, :, Y_SLICE, X_SLICE], dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "time_index.npy", time_index)

    print("X:", X.shape)
    print("y:", y.shape)
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
