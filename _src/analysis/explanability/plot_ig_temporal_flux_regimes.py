#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE
    / "integrated_gradients_encoder_convlstm"
    / "all_test_mean_ig_mean_abs_zero_steps64.npz"
)
DEFAULT_OUTPUT = (
    HERE
    / "integrated_gradients_encoder_convlstm"
    / "all_test_ig_annual_monthly_flux_regimes.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot annual and 30-model-day mean Integrated Gradients feature importance, "
            "plus low- versus high-flux composites."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--snapshots-per-day", type=int, default=2)
    parser.add_argument("--month-days", type=int, default=30)
    parser.add_argument("--year-days", type=int, default=365)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--min-samples-per-window", type=int, default=5)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def window_means(
    sample_index: np.ndarray,
    fractions: np.ndarray,
    width: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_id = sample_index // width
    ids = np.unique(window_id)
    means = []
    centers = []
    counts = []
    for current_id in ids:
        mask = window_id == current_id
        count = int(mask.sum())
        if count < min_samples:
            continue
        means.append(np.mean(fractions[mask], axis=0))
        centers.append((current_id + 0.5) * width)
        counts.append(count)
    if not means:
        raise ValueError(f"No windows contain at least {min_samples} held-out samples")
    return np.asarray(centers), np.asarray(means), np.asarray(counts)


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png; this script exports PNG only")
    if args.snapshots_per_day < 1:
        raise ValueError("--snapshots-per-day must be positive")
    if not 0.0 < args.low_quantile < args.high_quantile < 1.0:
        raise ValueError("Require 0 < low-quantile < high-quantile < 1")

    with np.load(args.input, allow_pickle=True) as data:
        required = {
            "sample_indices",
            "objective_values",
            "sample_feature_importance",
            "feature_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"{args.input} lacks {sorted(missing)}. Rerun Integrated Gradients "
                "with the updated --all-test mode."
            )
        sample_index = np.asarray(data["sample_indices"], dtype=np.int64)
        objective = np.asarray(data["objective_values"], dtype=np.float64)
        importance = np.asarray(data["sample_feature_importance"], dtype=np.float64)
        feature_names = [str(value) for value in data["feature_names"]]

    totals = importance.sum(axis=1, keepdims=True)
    valid = np.isfinite(totals[:, 0]) & (totals[:, 0] > 0) & np.isfinite(objective)
    sample_index = sample_index[valid]
    objective = objective[valid]
    fractions = importance[valid] / totals[valid] * 100.0

    month_width = args.month_days * args.snapshots_per_day
    year_width = args.year_days * args.snapshots_per_day
    annual_x, annual, annual_n = window_means(
        sample_index, fractions, year_width, args.min_samples_per_window
    )
    monthly_x, monthly, monthly_n = window_means(
        sample_index, fractions, month_width, args.min_samples_per_window
    )

    low_cut = float(np.quantile(objective, args.low_quantile))
    high_cut = float(np.quantile(objective, args.high_quantile))
    low = fractions[objective <= low_cut]
    high = fractions[objective >= high_cut]
    low_mean = low.mean(axis=0)
    high_mean = high.mean(axis=0)
    low_se = low.std(axis=0, ddof=1) / np.sqrt(len(low))
    high_se = high.std(axis=0, ddof=1) / np.sqrt(len(high))

    apply_style()
    fig = plt.figure(figsize=(7.2, 6.7), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.35, 1.45))
    axes = [fig.add_subplot(grid[index]) for index in range(3)]

    vmax = float(np.nanpercentile(np.concatenate([annual.ravel(), monthly.ravel()]), 99))
    extent_annual = [
        annual_x[0] / year_width + 0.5,
        annual_x[-1] / year_width + 0.5,
        -0.5,
        len(feature_names) - 0.5,
    ]
    annual_mesh = axes[0].imshow(
        annual.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        extent=extent_annual,
        interpolation="nearest",
    )
    axes[0].set_yticks(np.arange(len(feature_names)))
    axes[0].set_yticklabels(feature_names)
    axes[0].set_xlabel("Model year")
    axes[0].set_title("Annual mean relative feature attribution", fontsize=8, pad=4)
    fig.colorbar(annual_mesh, ax=axes[0], pad=0.015, label="Mean relative |IG| (%)")

    extent_month = [
        monthly_x[0] / month_width + 0.5,
        monthly_x[-1] / month_width + 0.5,
        -0.5,
        len(feature_names) - 0.5,
    ]
    monthly_mesh = axes[1].imshow(
        monthly.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        extent=extent_month,
        interpolation="nearest",
    )
    axes[1].set_yticks(np.arange(len(feature_names)))
    axes[1].set_yticklabels(feature_names)
    axes[1].set_xlabel("30-model-day window")
    axes[1].set_title("Monthly mean relative feature attribution", fontsize=8, pad=4)
    fig.colorbar(monthly_mesh, ax=axes[1], pad=0.015, label="Mean relative |IG| (%)")

    order = np.argsort(high_mean - low_mean)
    y = np.arange(len(feature_names))
    offset = 0.18
    axes[2].barh(
        y - offset,
        low_mean[order],
        xerr=low_se[order],
        height=0.34,
        color="#6B8EAD",
        label=f"Low flux (bottom {args.low_quantile:.0%}, n={len(low)})",
        capsize=2,
    )
    axes[2].barh(
        y + offset,
        high_mean[order],
        xerr=high_se[order],
        height=0.34,
        color="#C75B4A",
        label=f"High flux (top {1 - args.high_quantile:.0%}, n={len(high)})",
        capsize=2,
    )
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([feature_names[index] for index in order])
    axes[2].set_xlabel("Mean relative |IG| (%)")
    axes[2].set_title(
        f"Flux-regime comparison (J thresholds: {low_cut:.2f}, {high_cut:.2f})",
        fontsize=8,
        pad=4,
    )
    axes[2].grid(axis="x", color="#D0D0D0", linewidth=0.4, alpha=0.65)
    axes[2].legend(loc="lower right", fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Annual windows: {len(annual)}; held-out n range: {annual_n.min()}-{annual_n.max()}")
    print(f"Monthly windows: {len(monthly)}; held-out n range: {monthly_n.min()}-{monthly_n.max()}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
