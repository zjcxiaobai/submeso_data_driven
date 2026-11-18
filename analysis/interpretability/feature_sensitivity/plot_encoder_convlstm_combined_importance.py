#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

PALETTE = {
    "normal": "#B64342",
    "ablated": "#0F4D92",
    "bar": "#0F4D92",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine final model_analysis separated-time feature-ablation time series with sorted feature importance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "model_analysis_feature_importance.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "model_analysis_feature_importance_combined",
    )
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_result(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature-importance result: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"sample_index", "baseline_r2", "feature_r2", "feature_drop", "feature_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return (
            np.asarray(data["sample_index"], dtype=np.int64),
            np.asarray(data["baseline_r2"], dtype=np.float64),
            np.asarray(data["feature_r2"], dtype=np.float64),
            np.asarray(data["feature_drop"], dtype=np.float64),
            [str(name) for name in data["feature_names"]],
        )


def draw_figure(
    sample_index: np.ndarray,
    baseline_r2: np.ndarray,
    feature_r2: np.ndarray,
    feature_drop: np.ndarray,
    feature_names: list[str],
    output_prefix: Path,
) -> None:
    apply_style()
    n_features = len(feature_names)
    order_desc = np.argsort(feature_drop)[::-1]
    order_asc = np.argsort(feature_drop)

    fig = plt.figure(figsize=(7.2, 8.9), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=(5.9, 1.55), hspace=0.08)
    ts_grid = outer[0].subgridspec(4, 2, hspace=0.16, wspace=0.12)
    ts_axes = np.asarray([[fig.add_subplot(ts_grid[r, c]) for c in range(2)] for r in range(4)])

    for rank, (feature_index, ax) in enumerate(zip(order_desc, ts_axes.ravel()[:n_features]), start=1):
        ax.plot(sample_index, baseline_r2, color=PALETTE["normal"], linewidth=0.85, label="Normal")
        ax.plot(sample_index, feature_r2[feature_index], color=PALETTE["ablated"], linewidth=0.85, label="Feature replaced")
        ax.axhline(0.0, color=PALETTE["neutral_light"], linewidth=0.6, zorder=0)
        ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
        ax.tick_params(labelsize=6, colors=PALETTE["neutral_dark"])
        ax.set_title(
            f"{rank}. {feature_names[feature_index]}: mean $R^2$ drop={feature_drop[feature_index]:.3f}",
            fontsize=7,
            pad=3,
        )

    for ax in ts_axes[-1, :]:
        ax.set_xlabel("Test sample index")
    for ax in ts_axes[:, 0]:
        ax.set_ylabel(r"$R^2$")
    ts_axes.ravel()[1].legend(loc="best", fontsize=6, handlelength=2.0)

    ax_bar = fig.add_subplot(outer[1])
    y = np.arange(n_features)
    ax_bar.barh(y, feature_drop[order_asc], color=PALETTE["bar"], height=0.72)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([feature_names[index] for index in order_asc])
    ax_bar.set_xlabel(r"Mean $R^2$ drop")
    ax_bar.set_ylabel("Feature")
    ax_bar.set_title("Sorted feature importance", fontsize=8, pad=4)
    ax_bar.axvline(0.0, color=PALETTE["neutral_mid"], linewidth=0.7)
    ax_bar.grid(axis="x", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
    ax_bar.tick_params(labelsize=6, colors=PALETTE["neutral_dark"])

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_index, baseline_r2, feature_r2, feature_drop, feature_names = load_result(args.input)
    draw_figure(sample_index, baseline_r2, feature_r2, feature_drop, feature_names, args.output_prefix)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
