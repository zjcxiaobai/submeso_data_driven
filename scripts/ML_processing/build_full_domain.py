"""Assemble the full-domain machine-learning dataset."""

from pathlib import Path

import numpy as np


FEATURES = ("U", "V", "W", "zeta", "strain", "T", "rho", "kpphbl")
TARGET = "wT"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "data/interim/feature_target_4D_coarse"
OUTPUT_DIR = PROJECT_ROOT / "data/interim/full_domain"


def load_variable(name):
    """Load one processed variable and its time index."""
    with np.load(INPUT_DIR / f"{name}.npz") as npz:
        time = npz["time_index"].astype(np.int64)
        data = npz["data"].astype(np.float32)
    return time, data


def main():
    variables = {
        name: load_variable(name)
        for name in (*FEATURES, TARGET)
    }

    # Keep times available for every feature and the target.
    common_time = variables[FEATURES[0]][0]
    for name in (*FEATURES[1:], TARGET):
        common_time = np.intersect1d(
            common_time,
            variables[name][0],
        )

    # Select each variable at the common times.
    def align(name):
        time, data = variables[name]
        lookup = {int(t): index for index, t in enumerate(time)}
        indices = [lookup[int(t)] for t in common_time]
        return data[indices]

    X = np.stack(
        [align(name) for name in FEATURES],
        axis=1,
    ).astype(np.float32)

    y = align(TARGET)[:, None].astype(np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "time_index.npy", common_time)

    print("X:", X.shape)
    print("y:", y.shape)
    print("time_index:", common_time.shape)


if __name__ == "__main__":
    main()