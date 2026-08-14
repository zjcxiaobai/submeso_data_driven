#!/usr/bin/env python3
"""Plot per-snapshot R2 for upper-model evaluation on lower_40x20."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a lower_40x20 R2 time series.")
    parser.add_argument(
        "--r2-series",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_model_analysis_r2_series.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_model_analysis_r2_series",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--x-axis",
        choices=("sample_index", "sample_number"),
        default="sample_index",
        help="Use chronological lower-domain sample_index or sequential sample_number.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = np.asarray([data])
    return np.asarray(data["sample_index"], dtype=np.int64), np.asarray(data["r2"], dtype=np.float64)


def main() -> None:
    args = parse_args()
    sample_index, r2 = load_csv(args.r2_series)
    finite = np.isfinite(r2)
    if not np.any(finite):
        raise ValueError(f"No finite R2 values in {args.r2_series}")

    if args.x_axis == "sample_number":
        x = np.arange(r2.size, dtype=np.int64)
        xlabel = "Lower 40x20 sample number"
    else:
        x = sample_index
        xlabel = "Lower 40x20 sample index"

    mean_r2 = float(np.nanmean(r2))
    median_r2 = float(np.nanmedian(r2))
    p05 = float(np.nanpercentile(r2, 5))
    p95 = float(np.nanpercentile(r2, 95))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    ax.plot(x, r2, color="#1F5D8C", linewidth=0.75)
    ax.axhline(0.0, color="#777777", linewidth=0.7)
    ax.axhline(mean_r2, color="#B9473A", linewidth=1.0, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Per-snapshot $R^2$")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.7)
    ax.text(
        0.02,
        0.08,
        rf"mean $R^2$={mean_r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#B9473A",
    )
    low = float(np.nanpercentile(r2[finite], 1))
    high = float(np.nanpercentile(r2[finite], 99))
    pad = max(0.05 * (high - low), 0.05)
    ax.set_ylim(low - pad, high + pad)

    png_output = args.output_prefix.with_suffix(".png")
    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_output = args.summary_output or args.output_prefix.with_name(args.output_prefix.name + "_summary.txt")
    with summary_output.open("w") as handle:
        handle.write(f"Input: {args.r2_series}\n")
        handle.write(f"Samples: {r2.size}\n")
        handle.write(f"First sample_index: {int(sample_index[0])}\n")
        handle.write(f"Last sample_index: {int(sample_index[-1])}\n")
        handle.write(f"Mean R2: {mean_r2:.8f}\n")
        handle.write(f"Median R2: {median_r2:.8f}\n")
        handle.write(f"5th percentile R2: {p05:.8f}\n")
        handle.write(f"95th percentile R2: {p95:.8f}\n")
        handle.write(f"PNG: {png_output}\n")

    print(f"Saved {png_output}", flush=True)
    print(f"Saved {summary_output}", flush=True)


if __name__ == "__main__":
    main()
