#!/usr/bin/env python3
"""Plot feature-ablation time series and mean feature importance."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT = Path(__file__).resolve().with_name("model_analysis_feature_importance.npz")
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/interpretability/feature_sensitivity/model_analysis_feature_importance_combined.png"

RED = "#B64342"
BLUE = "#0F4D92"
DARK_GRAY = "#4D4D4D"
MID_GRAY = "#767676"
LIGHT_GRAY = "#CFCECE"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    }
)


def main() -> None:
    with np.load(INPUT) as data:
        sample_index = data["sample_index"]
        baseline_r2 = data["baseline_r2"]
        feature_r2 = data["feature_r2"]
        feature_drop = data["feature_drop"]
        feature_names = data["feature_names"].astype(str)

    strongest_first = np.argsort(feature_drop)[::-1]
    weakest_first = strongest_first[::-1]

    fig = plt.figure(figsize=(7.2, 8.9), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=(5.9, 1.55), hspace=0.08)
    time_grid = outer[0].subgridspec(4, 2, hspace=0.16, wspace=0.12)
    axes = [fig.add_subplot(time_grid[row, column]) for row in range(4) for column in range(2)]

    for rank, (feature, axis) in enumerate(zip(strongest_first, axes), start=1):
        axis.plot(sample_index, baseline_r2, color=RED, linewidth=0.85, label="Normal")
        axis.plot(sample_index, feature_r2[feature], color=BLUE, linewidth=0.85, label="Feature replaced")
        axis.axhline(0, color=LIGHT_GRAY, linewidth=0.6)
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.35, alpha=0.55)
        axis.set_title(
            f"{rank}. {feature_names[feature]}: mean $R^2$ drop={feature_drop[feature]:.3f}",
            fontsize=7,
            pad=3,
        )
        axis.tick_params(labelsize=6, colors=DARK_GRAY)

    for axis in axes[-2:]:
        axis.set_xlabel("Test sample index")
    for axis in axes[::2]:
        axis.set_ylabel(r"$R^2$")
    axes[1].legend(loc="best", fontsize=6, handlelength=2)

    bar_axis = fig.add_subplot(outer[1])
    positions = np.arange(len(feature_names))
    bar_axis.barh(positions, feature_drop[weakest_first], color=BLUE, height=0.72)
    bar_axis.set_yticks(positions, feature_names[weakest_first])
    bar_axis.set_xlabel(r"Mean $R^2$ drop")
    bar_axis.set_ylabel("Feature")
    bar_axis.set_title("Sorted feature importance", fontsize=8, pad=4)
    bar_axis.axvline(0, color=MID_GRAY, linewidth=0.7)
    bar_axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.35, alpha=0.55)
    bar_axis.tick_params(labelsize=6, colors=DARK_GRAY)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
