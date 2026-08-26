#!/usr/bin/env python3
"""Plot feature maps from the final decoder layer."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "last_decoder_pca_sample_1379/last_decoder_all_channel_feature_maps.npz"
DEFAULT_OUTPUT_DIR = HERE / "last_decoder_feature_maps_sample_1379"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-channels", type=int, default=128, help="Number of channels to plot after sorting.")
    parser.add_argument("--n-cols", type=int, default=8)
    parser.add_argument("--sort-by", choices=("rms", "mean_abs", "std", "none"), default="rms")
    parser.add_argument("--sample-position", type=int, default=0, help="Position inside the cached feature_map batch.")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile for symmetric color scaling.")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--prefix", default="last_decoder_feature_maps")
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.55,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    })


def channel_scores(feature_maps: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "rms": np.sqrt(np.nanmean(feature_maps**2, axis=(1, 2))),
        "mean_abs": np.nanmean(np.abs(feature_maps), axis=(1, 2)),
        "std": np.nanstd(feature_maps, axis=(1, 2)),
        "none": np.arange(feature_maps.shape[0], 0, -1, dtype=float),
    }


def load_feature_maps(path: Path, sample_position: int) -> tuple[np.ndarray, int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing decoder feature cache: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "feature_maps" not in data.files:
            raise ValueError(f"{path} is missing feature_maps")
        feature_maps = np.asarray(data["feature_maps"], dtype=np.float64)
        sample_index = np.asarray(data["sample_index"], dtype=np.int64).reshape(-1)
        test_position = np.asarray(data["test_position"], dtype=np.int64).reshape(-1)
    if feature_maps.ndim == 3:
        feature_maps = feature_maps[None, ...]
    if feature_maps.ndim != 4:
        raise ValueError(f"Expected feature_maps shape (N,C,H,W), got {feature_maps.shape}")
    if sample_position < 0 or sample_position >= feature_maps.shape[0]:
        raise ValueError(f"sample_position {sample_position} outside cached batch length {feature_maps.shape[0]}")
    sample = int(sample_index[sample_position]) if sample_index.size > sample_position else -1
    position = int(test_position[sample_position]) if test_position.size > sample_position else -1
    return feature_maps[sample_position], sample, position


def plot_grid(
    maps: np.ndarray,
    channels: np.ndarray,
    scores: np.ndarray,
    output: Path,
    sample_index: int,
    sort_by: str,
    n_cols: int,
    percentile: float,
    cmap: str,
) -> None:
    apply_style()
    n = maps.shape[0]
    n_rows = int(math.ceil(n / n_cols))
    panel_w = 1.25
    panel_h = 0.82
    fig_w = max(6.0, n_cols * panel_w + 0.45)
    fig_h = max(2.2, n_rows * panel_h + 0.55)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    finite = maps[np.isfinite(maps)]
    vmax = float(np.nanpercentile(np.abs(finite), percentile)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    image = None
    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis("off")
            continue
        image = ax.imshow(maps[idx], origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"ch {int(channels[idx])}\n{scores[idx]:.2g}", fontsize=6.2, pad=1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.tolist(), fraction=0.012, pad=0.006)
        cbar.ax.tick_params(labelsize=6.0, length=1.8, width=0.5)
        cbar.set_label("Activation", fontsize=6.5, labelpad=2)
    fig.suptitle(f"Final decoder feature maps, sample {sample_index} ({sort_by})", fontsize=8.4, y=1.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def write_rank_csv(path: Path, channels: np.ndarray, scores_by_name: dict[str, np.ndarray], order: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "channel", "rms", "mean_abs", "std"])
        writer.writeheader()
        for rank, channel in enumerate(order, start=1):
            writer.writerow({
                "rank": rank,
                "channel": int(channel),
                "rms": float(scores_by_name["rms"][channel]),
                "mean_abs": float(scores_by_name["mean_abs"][channel]),
                "std": float(scores_by_name["std"][channel]),
            })


def main() -> None:
    args = parse_args()
    maps, sample_index, test_position = load_feature_maps(args.feature_cache, args.sample_position)
    scores_by_name = channel_scores(maps)
    if args.sort_by == "none":
        order = np.arange(maps.shape[0], dtype=np.int64)
    else:
        order = np.argsort(scores_by_name[args.sort_by])[::-1]
    n_plot = min(int(args.n_channels), maps.shape[0])
    selected = order[:n_plot]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.prefix}_top{n_plot}_by_{args.sort_by}.png"
    plot_grid(
        maps[selected],
        selected,
        scores_by_name[args.sort_by][selected],
        output,
        sample_index,
        args.sort_by,
        args.n_cols,
        args.percentile,
        args.cmap,
    )
    rank_csv = args.output_dir / f"{args.prefix}_channel_rank.csv"
    write_rank_csv(rank_csv, np.arange(maps.shape[0]), scores_by_name, order)
    print(f"Feature cache: {args.feature_cache}")
    print(f"sample_index={sample_index}, test_position={test_position}, channels={maps.shape[0]}")
    print(f"Saved {output}")
    print(f"Saved {rank_csv}")


if __name__ == "__main__":
    main()
