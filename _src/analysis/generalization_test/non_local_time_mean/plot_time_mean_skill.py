#!/usr/bin/env python3
"""Plot time-mean skill for the non-local lower 40x20 evaluation subdomain."""

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
NON_LOCAL_DIR = HERE.parent / "non_local"

PALETTE = {
    "fluxnet": "#1F5A85",
    "fk": "#B64A3B",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fluxnet-predictions",
        type=Path,
        default=NON_LOCAL_DIR / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz",
    )
    parser.add_argument(
        "--fk-predictions",
        type=Path,
        default=NON_LOCAL_DIR / "evaluation_subdomain_fk_prediction.npz",
        help="Assembled cropped FK prediction file, used if it exists.",
    )
    parser.add_argument(
        "--fk-progress",
        type=Path,
        default=NON_LOCAL_DIR / "evaluation_subdomain_fk_progress_fullgrid.npz",
        help="Full-grid FK progress file used when --fk-predictions does not exist.",
    )
    parser.add_argument(
        "--fk-crop",
        type=int,
        nargs=4,
        default=(72, 92, 12, 52),
        metavar=("J0", "J1", "I0", "I1"),
        help="Crop from the FK full grid to the lower 40x20 evaluation subdomain.",
    )
    parser.add_argument("--output-prefix", type=Path, default=HERE / "time_mean_skill_comparison")
    parser.add_argument("--csv-output", type=Path, default=HERE / "time_mean_skill_comparison.csv")
    parser.add_argument("--summary-output", type=Path, default=HERE / "time_mean_skill_comparison_summary.txt")
    parser.add_argument("--fk-display-label", default="MLE parameterization")
    parser.add_argument(
        "--window-csv",
        type=Path,
        default=NON_LOCAL_DIR / "evaluation_subdomain_best_1_year_window.csv",
        help="One-year evaluation window CSV used to select the paper time window.",
    )
    parser.add_argument("--snapshots-per-day", type=float, default=2.0)
    parser.add_argument(
        "--windows-days",
        type=float,
        nargs="*",
        default=[1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 240, 365],
        help="Averaging windows in model days. Defaults stop at one year for the selected one-year window.",
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


def load_fluxnet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SubmesoFluxNet predictions: {path}")
    with np.load(path) as data:
        required = {"sample_index", "time_index", "prediction", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        time_index = np.asarray(data["time_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected SubmesoFluxNet prediction shape (N,1,H,W), got {prediction.shape}")
    return sample, time_index, prediction[:, 0], target[:, 0]


def load_window_time_index(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing one-year window CSV: {path}")
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "time_index" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing time_index")
        values = [int(row["time_index"]) for row in reader if row["time_index"] != ""]
    if not values:
        raise ValueError(f"No time_index values found in {path}")
    return np.asarray(values, dtype=np.int64)


def load_fk_prediction(prediction_path: Path, progress_path: Path, crop: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if prediction_path.exists():
        with np.load(prediction_path) as data:
            if "time_index" not in data.files:
                raise ValueError(f"{prediction_path} is missing time_index")
            pred_key = "prediction" if "prediction" in data.files else "fk_heat_flux"
            if pred_key not in data.files:
                raise ValueError(f"{prediction_path} is missing prediction/fk_heat_flux")
            time_index = np.asarray(data["time_index"], dtype=np.int64)
            prediction = np.asarray(data[pred_key], dtype=np.float64)
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError(f"Expected FK prediction shape (N,1,H,W), got {prediction.shape}")
        return time_index, prediction[:, 0]

    if not progress_path.exists():
        raise FileNotFoundError(f"Missing FK prediction/progress files: {prediction_path}, {progress_path}")
    with np.load(progress_path) as data:
        required = {"time_index", "data"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{progress_path} is missing arrays: {sorted(missing)}")
        time_index = np.asarray(data["time_index"], dtype=np.int64)
        full_grid = np.asarray(data["data"], dtype=np.float64)
    j0, j1, i0, i1 = crop
    prediction = full_grid[:, j0:j1, i0:i1]
    if prediction.shape[-2:] != (20, 40):
        raise ValueError(f"Expected lower 40x20 FK crop, got {prediction.shape}")
    return time_index, prediction


def align_by_time(
    flux_time: np.ndarray,
    flux_prediction: np.ndarray,
    target: np.ndarray,
    fk_time: np.ndarray,
    fk_prediction: np.ndarray,
    selected_time: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fk_lookup = {int(t): n for n, t in enumerate(fk_time)}
    flux_lookup = {int(t): n for n, t in enumerate(flux_time)}
    if selected_time is None:
        selected_time = flux_time
    aligned_time: list[int] = []
    flux_pos: list[int] = []
    fk_pos: list[int] = []
    for time in selected_time:
        flux_match = flux_lookup.get(int(time))
        fk_match = fk_lookup.get(int(time))
        if flux_match is not None and fk_match is not None:
            aligned_time.append(int(time))
            flux_pos.append(flux_match)
            fk_pos.append(fk_match)
    if not flux_pos:
        raise ValueError("No overlapping selected time_index values between SubmesoFluxNet and FK predictions.")
    flux_idx = np.asarray(flux_pos, dtype=np.int64)
    fk_idx = np.asarray(fk_pos, dtype=np.int64)
    return np.asarray(aligned_time, dtype=np.int64), flux_prediction[flux_idx], target[flux_idx], fk_prediction[fk_idx]


def block_average(values: np.ndarray, window_snapshots: int) -> np.ndarray:
    n_blocks = values.shape[0] // window_snapshots
    if n_blocks < 1:
        raise ValueError(f"Window {window_snapshots} snapshots is longer than the series length {values.shape[0]}")
    trimmed = values[: n_blocks * window_snapshots]
    reshaped = trimmed.reshape(n_blocks, window_snapshots, *values.shape[1:])
    return np.nanmean(reshaped, axis=1)


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((prediction - target) ** 2)))


def compute_skill_by_window(
    target: np.ndarray,
    flux_prediction: np.ndarray,
    fk_prediction: np.ndarray,
    windows_days: list[float],
    snapshots_per_day: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    n_time = int(target.shape[0])
    seen: set[int] = set()
    for window_days in windows_days:
        window_snapshots = max(1, int(round(float(window_days) * snapshots_per_day)))
        if window_snapshots in seen or window_snapshots > n_time:
            continue
        seen.add(window_snapshots)
        target_avg = block_average(target, window_snapshots)
        flux_avg = block_average(flux_prediction, window_snapshots)
        fk_avg = block_average(fk_prediction, window_snapshots)
        rows.append(
            {
                "window_days": float(window_snapshots / snapshots_per_day),
                "window_snapshots": int(window_snapshots),
                "n_blocks": int(target_avg.shape[0]),
                "used_snapshots": int(target_avg.shape[0] * window_snapshots),
                "submesofluxnet_r2": r2_score(flux_avg, target_avg),
                "fk_r2": r2_score(fk_avg, target_avg),
                "submesofluxnet_rmse": rmse(flux_avg, target_avg),
                "fk_rmse": rmse(fk_avg, target_avg),
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
        "fk_r2",
        "submesofluxnet_rmse",
        "fk_rmse",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_figure(rows: list[dict[str, float | int]], output_prefix: Path, fk_label: str) -> None:
    apply_style()
    x = np.asarray([row["window_days"] for row in rows], dtype=np.float64)
    flux_r2 = np.asarray([row["submesofluxnet_r2"] for row in rows], dtype=np.float64)
    fk_r2 = np.asarray([row["fk_r2"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(4.9, 2.75), constrained_layout=True)
    ax.plot(
        x,
        flux_r2,
        color=PALETTE["fluxnet"],
        marker="o",
        markersize=3.0,
        linewidth=1.35,
        label="SubmesoFluxNet",
        zorder=3,
    )
    ax.plot(
        x,
        fk_r2,
        color=PALETTE["fk"],
        marker="s",
        markersize=3.0,
        linewidth=1.35,
        label=fk_label,
        zorder=3,
    )

    ymin = min(float(np.nanmin(flux_r2)), float(np.nanmin(fk_r2))) - 0.04
    ymax = min(1.0, max(float(np.nanmax(flux_r2)), float(np.nanmax(fk_r2))) + 0.08)
    ax.set_ylim(ymin, ymax)
    for window, label in ((30, "Monthly"), (90, "Seasonal"), (365, "Annual")):
        if x.min() <= window <= x.max():
            ax.axvline(window, color=PALETTE["neutral_mid"], linewidth=0.65, linestyle=(0, (3, 2)), zorder=0)
            ax.text(
                window,
                ymax - 0.015,
                label,
                rotation=90,
                ha="right",
                va="top",
                fontsize=6.2,
                color=PALETTE["neutral_dark"],
            )

    ax.set_xscale("log")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("Averaging window (model days)")
    ax.set_ylabel(r"Time-mean global $R^2$")
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])
    ax.legend(loc="lower right", fontsize=6.7, handlelength=1.7)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_prefix}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    rows: list[dict[str, float | int]],
    n_aligned: int,
    time_index: np.ndarray,
    snapshots_per_day: float,
) -> None:
    best_flux = max(rows, key=lambda row: float(row["submesofluxnet_r2"]))
    best_fk = max(rows, key=lambda row: float(row["fk_r2"]))
    with path.open("w") as handle:
        handle.write("Non-local time-mean skill comparison\n")
        handle.write("metric: global R2 after block averaging prediction and truth over consecutive evaluation-subdomain snapshots\n")
        handle.write("time_window: evaluation_subdomain_best_1_year_window.csv\n")
        handle.write(f"snapshots_per_day: {snapshots_per_day:.10g}\n")
        handle.write(f"n_aligned_snapshots: {n_aligned}\n")
        handle.write(f"start_time_index: {int(time_index[0])}\n")
        handle.write(f"end_time_index: {int(time_index[-1])}\n")
        if time_index.size > 1:
            diffs = np.diff(time_index)
            handle.write(f"time_index_median_spacing: {float(np.median(diffs)):.10g}\n")
            handle.write(f"time_index_min_spacing: {int(np.min(diffs))}\n")
            handle.write(f"time_index_max_spacing: {int(np.max(diffs))}\n")
        handle.write(f"best_submesofluxnet_window_days: {float(best_flux['window_days']):.10g}\n")
        handle.write(f"best_submesofluxnet_r2: {float(best_flux['submesofluxnet_r2']):.10g}\n")
        handle.write(f"best_fk_window_days: {float(best_fk['window_days']):.10g}\n")
        handle.write(f"best_fk_r2: {float(best_fk['fk_r2']):.10g}\n")
        for mark in (30, 90, 365):
            match = next((row for row in rows if abs(float(row["window_days"]) - mark) < 1e-9), None)
            if match is not None:
                handle.write(f"\nwindow_{mark}_days\n")
                handle.write(f"window_snapshots: {match['window_snapshots']}\n")
                handle.write(f"n_blocks: {match['n_blocks']}\n")
                handle.write(f"submesofluxnet_r2: {float(match['submesofluxnet_r2']):.10g}\n")
                handle.write(f"fk_r2: {float(match['fk_r2']):.10g}\n")


def main() -> None:
    args = parse_args()
    _, flux_time, flux_prediction, target = load_fluxnet(args.fluxnet_predictions)
    fk_time, fk_prediction = load_fk_prediction(args.fk_predictions, args.fk_progress, tuple(args.fk_crop))
    selected_time = load_window_time_index(args.window_csv)
    aligned_time, flux_prediction, target, fk_prediction = align_by_time(
        flux_time,
        flux_prediction,
        target,
        fk_time,
        fk_prediction,
        selected_time,
    )
    rows = compute_skill_by_window(
        target,
        flux_prediction,
        fk_prediction,
        args.windows_days,
        args.snapshots_per_day,
    )
    write_csv(args.csv_output, rows)
    write_summary(args.summary_output, rows, int(target.shape[0]), aligned_time, args.snapshots_per_day)
    draw_figure(rows, args.output_prefix, args.fk_display_label)

    print(f"Aligned snapshots: {target.shape[0]}", flush=True)
    print(f"Time index: {int(aligned_time[0])}-{int(aligned_time[-1])}", flush=True)
    print(f"Saved {args.output_prefix}.png", flush=True)
    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    for row in rows:
        if float(row["window_days"]) in {30.0, 90.0, 365.0}:
            print(
                f"window={row['window_days']:.0f} days: "
                f"SubmesoFluxNet R2={float(row['submesofluxnet_r2']):.6f}, "
                f"{args.fk_display_label} R2={float(row['fk_r2']):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
