#!/usr/bin/env python3
"""Plot the best one-year lower-domain R2 window and a diagnosed heat-flux snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the contiguous 365*2 snapshot window with maximum mean R2, "
            "then plot its R2 series beside a diagnosed heat-flux map."
        )
    )
    parser.add_argument(
        "--r2-series",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_r2_series.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_predictions_denorm.npz",
    )
    parser.add_argument(
        "--upper-truth",
        type=Path,
        default=PROJECT_DIR / "data/train_on_small_domain/upper_40x20/y.npy",
    )
    parser.add_argument(
        "--lower-truth",
        type=Path,
        default=PROJECT_DIR / "data/train_on_small_domain/lower_40x20/y.npy",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_best_1yr_window",
    )
    parser.add_argument("--window", type=int, default=365 * 2, help="Snapshots in the one-year window.")
    parser.add_argument(
        "--snapshot-choice",
        choices=("best", "median", "center"),
        default="best",
        help="Fallback snapshot within the best window if the requested sample is unavailable.",
    )
    parser.add_argument(
        "--snapshot-sample-index",
        type=int,
        default=3000,
        help="Sample index used for the right 40x40 diagnosed heat-flux snapshot.",
    )
    parser.add_argument(
        "--snapshot-field",
        choices=("prediction", "target"),
        default="prediction",
        help="Fallback field to map if --map-source is predictions.",
    )
    parser.add_argument(
        "--map-source",
        choices=("diagnosed_full_40x40", "predictions"),
        default="diagnosed_full_40x40",
        help="Right-panel source. Default stacks upper/lower diagnosed targets into 40x40.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Centered rolling window for the red R2 curve.",
    )
    return parser.parse_args()


def load_r2_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = np.asarray([data])
    return (
        np.asarray(data["test_position"], dtype=np.int64),
        np.asarray(data["sample_index"], dtype=np.int64),
        np.asarray(data["r2"], dtype=np.float64),
    )


def best_window(r2: np.ndarray, window: int) -> tuple[int, int, float]:
    if window <= 0:
        raise ValueError("--window must be positive")
    if r2.size < window:
        raise ValueError(f"R2 series has {r2.size} samples, shorter than window={window}")
    valid = np.isfinite(r2).astype(np.float64)
    values = np.where(np.isfinite(r2), r2, 0.0)
    value_sum = np.convolve(values, np.ones(window, dtype=np.float64), mode="valid")
    count_sum = np.convolve(valid, np.ones(window, dtype=np.float64), mode="valid")
    means = np.full_like(value_sum, np.nan, dtype=np.float64)
    np.divide(value_sum, count_sum, out=means, where=count_sum > 0.0)
    if not np.any(np.isfinite(means)):
        raise ValueError("No finite R2 values available for any window")
    start = int(np.nanargmax(means))
    stop = start + window
    return start, stop, float(means[start])


def choose_snapshot(window_r2: np.ndarray, start: int, choice: str) -> int:
    if choice == "center":
        return start + window_r2.size // 2
    if choice == "best":
        return start + int(np.nanargmax(window_r2))
    finite = np.isfinite(window_r2)
    if not np.any(finite):
        return start + window_r2.size // 2
    median_value = float(np.nanmedian(window_r2))
    local = int(np.nanargmin(np.abs(window_r2 - median_value)))
    return start + local


def symmetric_limits(arrays: list[np.ndarray], pct: float = 99.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(values), pct))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values.astype(np.float64), (half, half), mode="edge")
    valid = np.isfinite(padded).astype(np.float64)
    filled = np.where(np.isfinite(padded), padded, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    total = np.convolve(filled, kernel, mode="valid")
    count = np.convolve(valid, kernel, mode="valid")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0.0)
    return out


def load_diagnosed_full_40x40(upper_truth: Path, lower_truth: Path, sample_idx: int) -> np.ndarray:
    upper = np.load(upper_truth, mmap_mode="r")
    lower = np.load(lower_truth, mmap_mode="r")
    if sample_idx < 0 or sample_idx >= upper.shape[0] or sample_idx >= lower.shape[0]:
        raise IndexError(f"sample_index={sample_idx} is outside upper/lower truth arrays")
    upper_field = np.squeeze(np.asarray(upper[sample_idx], dtype=np.float32))
    lower_field = np.squeeze(np.asarray(lower[sample_idx], dtype=np.float32))
    if upper_field.shape != lower_field.shape:
        raise ValueError(f"Upper/lower field shapes differ: {upper_field.shape} vs {lower_field.shape}")
    return np.vstack([lower_field, upper_field])


def main() -> None:
    args = parse_args()
    test_position, sample_index, r2 = load_r2_csv(args.r2_series)
    start, stop, mean_r2 = best_window(r2, args.window)
    fallback_pos = choose_snapshot(r2[start:stop], start, args.snapshot_choice)
    sample_positions = np.where(sample_index == args.snapshot_sample_index)[0]
    snapshot_pos = int(sample_positions[0]) if sample_positions.size else fallback_pos

    if args.map_source == "diagnosed_full_40x40":
        heat_flux = load_diagnosed_full_40x40(args.upper_truth, args.lower_truth, args.snapshot_sample_index)
        map_sample_index = args.snapshot_sample_index
        map_r2_text = ""
    else:
        with np.load(args.predictions) as data:
            prediction_sample_index = np.asarray(data["sample_index"], dtype=np.int64)
            field_stack = np.asarray(data[args.snapshot_field], dtype=np.float32)
            if not np.array_equal(prediction_sample_index, sample_index):
                order = {int(idx): n for n, idx in enumerate(prediction_sample_index)}
                field_pos = order[int(sample_index[snapshot_pos])]
            else:
                field_pos = snapshot_pos
            heat_flux = np.squeeze(field_stack[field_pos])
        map_sample_index = int(sample_index[snapshot_pos])
        map_r2_text = f", R2={r2[snapshot_pos]:.3f}"

    window_x = sample_index[start:stop]
    window_r2 = r2[start:stop]
    window_sample_index = sample_index[start:stop]
    smooth_r2 = rolling_mean(window_r2, args.smooth_window)
    band_low = float(np.nanpercentile(window_r2, 20))
    band_high = float(np.nanpercentile(window_r2, 80))
    vmin, vmax = symmetric_limits([heat_flux])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.1,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(10.2, 4.2), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.72, 1.0), wspace=0.24)
    ax_ts = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    ax_ts.fill_between(window_x, band_low, band_high, color="#B8C7D9", alpha=0.18, linewidth=0.0)
    ax_ts.plot(window_x, window_r2, color="#5C8FB1", linewidth=1.0, alpha=0.85)
    ax_ts.plot(window_x, smooth_r2, color="#B9473A", linewidth=2.2)
    ax_ts.axhline(mean_r2, color="#333333", linewidth=1.1, linestyle=(0, (3, 3)))
    ax_ts.set_title(r"Non-local $R^2$ VS time", fontsize=17, pad=8)
    ax_ts.set_xlabel("Evaluation subdomain data index", fontsize=15)
    ax_ts.set_ylabel(r"Per-snapshot $R^2$", fontsize=15)
    ax_ts.grid(axis="y", color="#E3E3E3", linewidth=0.7, alpha=0.9)
    ax_ts.text(
        0.012,
        0.96,
        rf"mean $R^2$={mean_r2:.3f}",
        transform=ax_ts.transAxes,
        ha="left",
        va="top",
        color="#333333",
        fontsize=14,
    )

    im = ax_map.imshow(
        heat_flux,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=(0, heat_flux.shape[1], 0, heat_flux.shape[0]),
    )
    ny, nx = heat_flux.shape
    ax_map.set_title("Submesoscale VHF", fontsize=17, pad=8)
    if ny == 40:
        ax_map.axhline(20, color="#4C4C4C", linewidth=1.4)
        ax_map.text(-1.4, 30, "Training\nsubdomain", ha="right", va="center", fontsize=12, color="#333333")
        ax_map.text(-1.4, 10, "Evaluation\nSubdomain", ha="right", va="center", fontsize=12, color="#333333")
    else:
        ax_map.text(-1.4, ny / 2, "Evaluation\nSubdomain", ha="right", va="center", fontsize=12, color="#333333")
    ax_map.set_xlabel("x index", fontsize=15)
    ax_map.set_ylabel("y index", fontsize=15)
    ax_map.set_xlim(0, nx)
    ax_map.set_ylim(0, ny)
    ax_map.set_xticks([0, nx // 2, nx])
    ax_map.set_yticks([0, ny // 2, ny])
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.03)
    cbar.set_label(r"W m$^{-2}$", fontsize=13)

    png_output = args.output_prefix.with_suffix(".png")
    pdf_output = args.output_prefix.with_suffix(".pdf")
    svg_output = args.output_prefix.with_suffix(".svg")
    summary_output = args.output_prefix.with_name(args.output_prefix.name + "_summary.txt")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.18, top=0.88, wspace=0.24)
    fig.savefig(png_output, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_output, bbox_inches="tight")
    fig.savefig(svg_output, bbox_inches="tight")
    plt.close(fig)

    with summary_output.open("w") as handle:
        handle.write(f"R2 input: {args.r2_series}\n")
        handle.write(f"Prediction input: {args.predictions}\n")
        handle.write(f"Window snapshots: {args.window}\n")
        handle.write(f"Best window test_position: {int(test_position[start])} ... {int(test_position[stop - 1])}\n")
        handle.write(f"Best window sample_index: {int(window_sample_index[0])} ... {int(window_sample_index[-1])}\n")
        handle.write(f"Best window mean R2: {mean_r2:.8f}\n")
        handle.write(f"Map source: {args.map_source}\n")
        handle.write(f"Requested map sample_index: {args.snapshot_sample_index}\n")
        handle.write(f"Map sample_index: {map_sample_index}\n")
        if sample_positions.size:
            handle.write(f"Map sample R2: {float(r2[snapshot_pos]):.8f}\n")
        handle.write(f"Heat-flux map shape: {ny} x {nx}\n")
        handle.write(f"PNG: {png_output}\n")
        handle.write(f"PDF: {pdf_output}\n")
        handle.write(f"SVG: {svg_output}\n")

    print(f"Best window sample_index: {int(window_sample_index[0])} ... {int(window_sample_index[-1])}", flush=True)
    print(f"Best window mean R2: {mean_r2:.6f}", flush=True)
    print(f"Map sample_index: {map_sample_index}{map_r2_text}", flush=True)
    print(f"Saved {png_output}", flush=True)
    print(f"Saved {summary_output}", flush=True)


if __name__ == "__main__":
    main()
