#!/usr/bin/env python3
"""Time-mean skill for the best one-year non-local lower-domain window."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PALETTE = {
    "fluxnet": "#1F5A85",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time-mean global R2 for the best one-year lower-domain non-local test window."
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
        "--output-prefix",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_best_1yr_time_mean_skill",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_best_1yr_time_mean_skill.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_best_1yr_time_mean_skill_summary.txt",
    )
    parser.add_argument("--window-snapshots", type=int, default=365 * 2)
    parser.add_argument("--snapshots-per-day", type=int, default=2)
    parser.add_argument(
        "--windows-days",
        type=float,
        nargs="*",
        default=[1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 75, 90, 120, 150, 180, 240, 300, 365],
    )
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 14,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 1.0,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


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
    valid = np.isfinite(r2).astype(np.float64)
    values = np.where(np.isfinite(r2), r2, 0.0)
    sums = np.convolve(values, np.ones(window, dtype=np.float64), mode="valid")
    counts = np.convolve(valid, np.ones(window, dtype=np.float64), mode="valid")
    means = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0.0)
    start = int(np.nanargmax(means))
    return start, start + window, float(means[start])


def block_average(values: np.ndarray, window: int) -> np.ndarray:
    n_blocks = values.shape[0] // window
    if n_blocks < 1:
        raise ValueError(f"Window {window} is longer than time series length {values.shape[0]}")
    trimmed = values[: n_blocks * window]
    reshaped = trimmed.reshape(n_blocks, window, *values.shape[1:])
    return np.nanmean(reshaped, axis=1)


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((prediction - target) ** 2)))


def compute_rows(
    prediction: np.ndarray,
    target: np.ndarray,
    windows_days: list[float],
    snapshots_per_day: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    used_windows: set[int] = set()
    for window_days in windows_days:
        window_snapshots = int(round(float(window_days) * snapshots_per_day))
        if window_snapshots < 1 or window_snapshots in used_windows or window_snapshots > target.shape[0]:
            continue
        used_windows.add(window_snapshots)
        target_avg = block_average(target, window_snapshots)
        prediction_avg = block_average(prediction, window_snapshots)
        rows.append(
            {
                "window_days": float(window_snapshots) / float(snapshots_per_day),
                "window_snapshots": int(window_snapshots),
                "n_blocks": int(target_avg.shape[0]),
                "used_snapshots": int(target_avg.shape[0] * window_snapshots),
                "submesofluxnet_r2": r2_score(prediction_avg, target_avg),
                "submesofluxnet_rmse": rmse(prediction_avg, target_avg),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_days",
        "window_snapshots",
        "n_blocks",
        "used_snapshots",
        "submesofluxnet_r2",
        "submesofluxnet_rmse",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows: list[dict[str, float | int]], output_prefix: Path) -> None:
    apply_style()
    x = np.asarray([row["window_days"] for row in rows], dtype=np.float64)
    r2 = np.asarray([row["submesofluxnet_r2"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
    ax.plot(
        x,
        r2,
        color=PALETTE["fluxnet"],
        marker="o",
        markersize=6.0,
        linewidth=2.3,
        label="SubmesoFluxNet",
    )

    ymin = max(-1.0, float(np.nanmin(r2)) - 0.04)
    ymax = min(1.02, float(np.nanmax(r2)) + 0.06)
    if ymax - ymin < 0.12:
        ymax = min(1.02, ymax + 0.06)
        ymin = max(-1.0, ymin - 0.06)
    ax.set_ylim(ymin, ymax)

    for window, label in ((30, "Monthly"), (90, "Seasonal"), (365, "Annual")):
        if x.min() <= window <= x.max():
            ax.axvline(window, color=PALETTE["neutral_mid"], linewidth=1.0, linestyle=(0, (3, 2)), zorder=0)
            ax.text(
                window,
                ymax - 0.012 * (ymax - ymin),
                label,
                rotation=90,
                ha="right",
                va="top",
                fontsize=13,
                color=PALETTE["neutral_dark"],
            )

    ax.set_xscale("log")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("Averaging window (model days)", fontsize=16)
    ax.set_ylabel(r"Time-mean global $R^2$", fontsize=16)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.7, alpha=0.55)
    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])
    ax.legend(loc="lower right", fontsize=14, handlelength=2.0)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    rows: list[dict[str, float | int]],
    best_start_sample: int,
    best_stop_sample: int,
    best_window_mean_r2: float,
    n_time: int,
    snapshots_per_day: int,
) -> None:
    best = max(rows, key=lambda row: float(row["submesofluxnet_r2"]))
    with path.open("w") as handle:
        handle.write("Best one-year non-local time-mean skill\n")
        handle.write("metric: global R2 after block averaging prediction and truth over consecutive snapshots\n")
        handle.write(f"snapshots_per_day: {snapshots_per_day}\n")
        handle.write(f"n_snapshots_in_best_year: {n_time}\n")
        handle.write(f"best_year_sample_index: {best_start_sample} ... {best_stop_sample}\n")
        handle.write(f"best_year_mean_per_snapshot_r2: {best_window_mean_r2:.10g}\n")
        handle.write(f"best_time_mean_window_days: {float(best['window_days']):.10g}\n")
        handle.write(f"best_time_mean_r2: {float(best['submesofluxnet_r2']):.10g}\n")
        for mark in (30, 90, 365):
            match = next((row for row in rows if abs(float(row["window_days"]) - mark) < 1.0e-9), None)
            if match is not None:
                handle.write(f"\nwindow_{mark}_days\n")
                handle.write(f"n_blocks: {match['n_blocks']}\n")
                handle.write(f"used_snapshots: {match['used_snapshots']}\n")
                handle.write(f"submesofluxnet_r2: {float(match['submesofluxnet_r2']):.10g}\n")


def main() -> None:
    args = parse_args()
    _, sample_index, r2 = load_r2_csv(args.r2_series)
    start, stop, best_mean_r2 = best_window(r2, args.window_snapshots)
    best_sample_index = sample_index[start:stop]

    with np.load(args.predictions) as data:
        pred_sample = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)[:, 0]
        target = np.asarray(data["target"], dtype=np.float64)[:, 0]

    position_by_sample = {int(sample): pos for pos, sample in enumerate(pred_sample)}
    positions = np.asarray([position_by_sample[int(sample)] for sample in best_sample_index], dtype=np.int64)
    prediction = prediction[positions]
    target = target[positions]

    rows = compute_rows(prediction, target, args.windows_days, args.snapshots_per_day)
    write_csv(args.csv_output, rows)
    write_summary(
        args.summary_output,
        rows,
        int(best_sample_index[0]),
        int(best_sample_index[-1]),
        best_mean_r2,
        int(target.shape[0]),
        int(args.snapshots_per_day),
    )
    draw_figure(rows, args.output_prefix)

    print(f"Best one-year sample_index: {int(best_sample_index[0])} ... {int(best_sample_index[-1])}", flush=True)
    print(f"Mean per-snapshot R2: {best_mean_r2:.6f}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    for row in rows:
        if float(row["window_days"]) in {30.0, 90.0, 365.0}:
            print(f"window={row['window_days']:.0f} days: R2={float(row['submesofluxnet_r2']):.6f}", flush=True)


if __name__ == "__main__":
    main()
