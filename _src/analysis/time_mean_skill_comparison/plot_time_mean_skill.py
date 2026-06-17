#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

PALETTE = {
    "fluxnet": "#1F5A85",
    "fk": "#B64A3B",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SubmesoFluxNet and FK skill as a function of temporal averaging window."
    )
    parser.add_argument(
        "--fluxnet-predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_predictions_denorm.npz",
    )
    parser.add_argument(
        "--fk-predictions",
        type=Path,
        default=PROJECT_DIR / "analysis/fk_mle/fk_test_Ce2p13062237_prediction_physical.npz",
    )
    parser.add_argument("--output-prefix", type=Path, default=HERE / "time_mean_skill_comparison")
    parser.add_argument("--csv-output", type=Path, default=HERE / "time_mean_skill_comparison.csv")
    parser.add_argument("--summary-output", type=Path, default=HERE / "time_mean_skill_comparison_summary.txt")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=[1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 240, 365, 540, 730, 1095, 1460],
        help="Averaging windows in snapshots. Defaults assume one saved test snapshot per day.",
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


def load_fluxnet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SubmesoFluxNet predictions: {path}")
    with np.load(path) as data:
        required = {"sample_index", "prediction", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected SubmesoFluxNet prediction shape (N,1,H,W), got {prediction.shape}")
    return sample, prediction[:, 0], target[:, 0]


def load_fk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Missing FK predictions: {path}")
    with np.load(path) as data:
        required = {"sample_index", "prediction"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        time_index = np.asarray(data["time_index"], dtype=np.int64) if "time_index" in data.files else None
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected FK prediction shape (N,1,H,W), got {prediction.shape}")
    return sample, prediction[:, 0], time_index


def align_to_fluxnet(
    flux_sample: np.ndarray,
    flux_prediction: np.ndarray,
    target: np.ndarray,
    fk_sample: np.ndarray,
    fk_prediction: np.ndarray,
    fk_time_index: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    fk_lookup = {int(sample): pos for pos, sample in enumerate(fk_sample)}
    fk_positions: list[int] = []
    flux_positions: list[int] = []
    for flux_pos, sample in enumerate(flux_sample):
        fk_pos = fk_lookup.get(int(sample))
        if fk_pos is not None:
            flux_positions.append(flux_pos)
            fk_positions.append(fk_pos)
    if not flux_positions:
        raise ValueError("No overlapping sample_index values between SubmesoFluxNet and FK predictions.")

    flux_pos_array = np.asarray(flux_positions, dtype=np.int64)
    fk_pos_array = np.asarray(fk_positions, dtype=np.int64)
    aligned_time = fk_time_index[fk_pos_array] if fk_time_index is not None else None
    return (
        flux_prediction[flux_pos_array],
        target[flux_pos_array],
        fk_prediction[fk_pos_array],
        aligned_time,
    )


def block_average(values: np.ndarray, window: int) -> np.ndarray:
    n_blocks = values.shape[0] // window
    if n_blocks < 1:
        raise ValueError(f"Window {window} is longer than the time series length {values.shape[0]}")
    trimmed = values[: n_blocks * window]
    reshaped = trimmed.reshape(n_blocks, window, *values.shape[1:])
    return np.nanmean(reshaped, axis=1)


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((prediction - target) ** 2)))


def compute_skill_by_window(
    target: np.ndarray,
    flux_prediction: np.ndarray,
    fk_prediction: np.ndarray,
    windows: list[int],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    n_time = int(target.shape[0])
    for window in sorted(set(w for w in windows if w >= 1)):
        if window > n_time:
            continue
        target_avg = block_average(target, window)
        flux_avg = block_average(flux_prediction, window)
        fk_avg = block_average(fk_prediction, window)
        rows.append(
            {
                "window_snapshots": int(window),
                "n_blocks": int(target_avg.shape[0]),
                "used_snapshots": int(target_avg.shape[0] * window),
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


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png", ".tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(Path(f"{output_prefix}{ext}"), **kwargs)


def draw_figure(rows: list[dict[str, float | int]], output_prefix: Path) -> None:
    apply_style()
    x = np.asarray([row["window_snapshots"] for row in rows], dtype=np.float64)
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
    )
    ax.plot(
        x,
        fk_r2,
        color=PALETTE["fk"],
        marker="s",
        markersize=3.0,
        linewidth=1.35,
        label="FK",
    )

    markers = ((30, "Monthly"), (90, "Seasonal"), (365, "Annual"))
    ymin = min(float(np.nanmin(flux_r2)), float(np.nanmin(fk_r2))) - 0.04
    ymax = min(1.0, max(float(np.nanmax(flux_r2)), float(np.nanmax(fk_r2))) + 0.08)
    ax.set_ylim(ymin, ymax)
    for window, label in markers:
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
    save_figure(fig, output_prefix)
    plt.close(fig)


def write_summary(
    path: Path,
    rows: list[dict[str, float | int]],
    n_time: int,
    time_index: np.ndarray | None,
) -> None:
    best_flux = max(rows, key=lambda row: float(row["submesofluxnet_r2"]))
    best_fk = max(rows, key=lambda row: float(row["fk_r2"]))
    with path.open("w") as handle:
        handle.write("Time-mean skill comparison\n")
        handle.write("metric: global R2 after block averaging prediction and truth over consecutive test snapshots\n")
        handle.write("window_units: model days, assuming one saved test snapshot per model day\n")
        handle.write(f"n_aligned_snapshots: {n_time}\n")
        if time_index is not None and time_index.size > 1:
            diffs = np.diff(time_index)
            handle.write(f"time_index_median_spacing: {float(np.median(diffs)):.10g}\n")
            handle.write(f"time_index_min_spacing: {int(np.min(diffs))}\n")
            handle.write(f"time_index_max_spacing: {int(np.max(diffs))}\n")
        handle.write(f"best_submesofluxnet_window: {best_flux['window_snapshots']}\n")
        handle.write(f"best_submesofluxnet_r2: {float(best_flux['submesofluxnet_r2']):.10g}\n")
        handle.write(f"best_fk_window: {best_fk['window_snapshots']}\n")
        handle.write(f"best_fk_r2: {float(best_fk['fk_r2']):.10g}\n")
        for mark in (30, 90, 365):
            match = next((row for row in rows if int(row["window_snapshots"]) == mark), None)
            if match is not None:
                handle.write(f"\nwindow_{mark}\n")
                handle.write(f"n_blocks: {match['n_blocks']}\n")
                handle.write(f"submesofluxnet_r2: {float(match['submesofluxnet_r2']):.10g}\n")
                handle.write(f"fk_r2: {float(match['fk_r2']):.10g}\n")


def main() -> None:
    args = parse_args()
    flux_sample, flux_prediction, target = load_fluxnet(args.fluxnet_predictions)
    fk_sample, fk_prediction, fk_time_index = load_fk(args.fk_predictions)
    flux_prediction, target, fk_prediction, aligned_time = align_to_fluxnet(
        flux_sample, flux_prediction, target, fk_sample, fk_prediction, fk_time_index
    )
    rows = compute_skill_by_window(target, flux_prediction, fk_prediction, args.windows)
    write_csv(args.csv_output, rows)
    write_summary(args.summary_output, rows, int(target.shape[0]), aligned_time)
    draw_figure(rows, args.output_prefix)

    for ext in (".svg", ".pdf", ".png", ".tiff"):
        print(f"Saved {args.output_prefix}{ext}", flush=True)
    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    for row in rows:
        if int(row["window_snapshots"]) in {30, 90, 365}:
            print(
                f"window={row['window_snapshots']}: "
                f"SubmesoFluxNet R2={float(row['submesofluxnet_r2']):.6f}, "
                f"FK R2={float(row['fk_r2']):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
