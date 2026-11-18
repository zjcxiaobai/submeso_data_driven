#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean standalone bottleneck feature-map images for architecture schematics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--sample-index", type=int, default=None, help="Optional sample_index. Default uses the best test R2 sample.")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--transparent", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def best_position(sample_index: np.ndarray, r2: np.ndarray, requested: int | None) -> int:
    if requested is not None:
        matches = np.where(sample_index == int(requested))[0]
        if matches.size == 0:
            raise ValueError(f"sample_index {requested} was not found")
        return int(matches[0])
    valid = np.where(np.isfinite(r2))[0]
    if valid.size == 0:
        raise ValueError("No finite R2 values found")
    return int(valid[np.argmax(r2[valid])])


def symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def save_feature(image: np.ndarray, output: Path, cmap: str, vlim: float, dpi: int, transparent: bool) -> None:
    height, width = image.shape
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        latent = np.asarray(data["latent_features"], dtype=np.float64)
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
    if latent.ndim != 4 or latent.shape[1] != 4:
        raise ValueError(f"Expected latent_features shape (N,4,H,W), got {latent.shape}")
    pos = best_position(sample_index, r2, args.sample_index)
    features = latent[pos]
    vlim = symmetric_limit(features)
    out = args.output_dir.expanduser()
    for i in range(4):
        save_feature(features[i], out / f"bottleneck_feature_{i + 1}.png", args.cmap, vlim, args.dpi, args.transparent)
    with (out / "selected_sample.txt").open("w") as handle:
        handle.write(f"sample_index: {int(sample_index[pos])}\n")
        handle.write(f"test_position: {pos}\n")
        handle.write(f"r2: {float(r2[pos])}\n")
        handle.write(f"shared_color_limit: +/-{vlim}\n")
    print(f"Saved 4 architecture feature-map PNGs under {out}")


if __name__ == "__main__":
    main()
