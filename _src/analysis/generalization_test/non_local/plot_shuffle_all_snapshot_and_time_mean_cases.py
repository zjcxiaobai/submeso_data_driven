#!/usr/bin/env python3
"""Plot shuffle-all non-local best/worst snapshots and best monthly/seasonal/yearly mean cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Case:
    label: str
    truth: np.ndarray
    prediction: np.ndarray
    r2: float
    sample_start: int
    sample_stop: int
    n_snapshots: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare truth/prediction maps for selected shuffle-all non-local cases."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_predictions_denorm.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_shuffle_all_selected_case_maps",
    )
    parser.add_argument("--snapshots-per-day", type=int, default=2)
    return parser.parse_args()


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def best_mean_case(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_index: np.ndarray,
    window: int,
    label: str,
) -> Case:
    if prediction.shape[0] < window:
        raise ValueError(f"{label} window={window} exceeds N={prediction.shape[0]}")
    best_r2 = -np.inf
    best_start = 0
    best_truth = None
    best_prediction = None
    # Sliding windows are fine here: 11181 samples x small 20x40 maps.
    for start in range(0, prediction.shape[0] - window + 1):
        stop = start + window
        pred_avg = np.nanmean(prediction[start:stop], axis=0)
        target_avg = np.nanmean(target[start:stop], axis=0)
        score = r2_score(pred_avg, target_avg)
        if np.isfinite(score) and score > best_r2:
            best_r2 = score
            best_start = start
            best_truth = target_avg
            best_prediction = pred_avg
    assert best_truth is not None and best_prediction is not None
    return Case(
        label=label,
        truth=np.asarray(best_truth),
        prediction=np.asarray(best_prediction),
        r2=float(best_r2),
        sample_start=int(sample_index[best_start]),
        sample_stop=int(sample_index[best_start + window - 1]),
        n_snapshots=int(window),
    )


def symmetric_limits(*arrays: np.ndarray, pct: float = 99.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    vmax = float(np.nanpercentile(np.abs(values), pct))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(np.abs(values)))
    if vmax <= 0.0:
        vmax = 1.0
    return -vmax, vmax


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def draw_figure(cases: list[Case], output_prefix: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(
        len(cases),
        2,
        figsize=(5.7, 5.95),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if axes.ndim == 1:
        axes = axes[None, :]

    for row, case in enumerate(cases):
        vmin, vmax = symmetric_limits(case.truth, case.prediction)
        for col, (title, field) in enumerate((("Truth", case.truth), ("Prediction", case.prediction))):
            ax = axes[row, col]
            im = ax.imshow(field, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")
            if row == 0:
                ax.set_title(title, fontsize=10, pad=4)
            if col == 0:
                if case.n_snapshots == 1:
                    ylabel = f"{case.label}\n$R^2$={case.r2:.3f}"
                else:
                    days = case.n_snapshots / 2.0
                    ylabel = f"{case.label}\n$R^2$={case.r2:.3f}\n{days:g} model days"
                ax.set_ylabel(ylabel, rotation=0, ha="right", va="center", labelpad=54, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        cbar = fig.colorbar(im, ax=axes[row, :], fraction=0.032, pad=0.012, shrink=0.86)
        cbar.set_label(r"$w'T'$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, cases: list[Case], predictions_path: Path) -> None:
    with path.open("w") as handle:
        handle.write("Shuffle-all non-local selected truth/prediction cases\n")
        handle.write(f"Predictions: {predictions_path}\n")
        for case in cases:
            handle.write("\n")
            handle.write(f"case: {case.label}\n")
            handle.write(f"r2: {case.r2:.10g}\n")
            handle.write(f"sample_start: {case.sample_start}\n")
            handle.write(f"sample_stop: {case.sample_stop}\n")
            handle.write(f"n_snapshots: {case.n_snapshots}\n")
            handle.write(f"n_days: {case.n_snapshots / 2.0:.10g}\n")


def main() -> None:
    args = parse_args()
    with np.load(args.predictions) as data:
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)[:, 0]
        target = np.asarray(data["target"], dtype=np.float64)[:, 0]
        r2 = np.asarray(data["r2"], dtype=np.float64)

    best_pos = int(np.nanargmax(r2))

    yearly_window = 365 * args.snapshots_per_day
    valid = np.isfinite(r2).astype(np.float64)
    values = np.where(np.isfinite(r2), r2, 0.0)
    sums = np.convolve(values, np.ones(yearly_window, dtype=np.float64), mode="valid")
    counts = np.convolve(valid, np.ones(yearly_window, dtype=np.float64), mode="valid")
    means = np.full_like(sums, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0.0)
    best_year_start = int(np.nanargmax(means))
    best_year_stop = best_year_start + yearly_window
    worst_pos = best_year_start + int(np.nanargmin(r2[best_year_start:best_year_stop]))

    cases = [
        Case(
            label="Best snapshot",
            truth=target[best_pos],
            prediction=prediction[best_pos],
            r2=float(r2[best_pos]),
            sample_start=int(sample_index[best_pos]),
            sample_stop=int(sample_index[best_pos]),
            n_snapshots=1,
        ),
        Case(
            label="Worst snapshot",
            truth=target[worst_pos],
            prediction=prediction[worst_pos],
            r2=float(r2[worst_pos]),
            sample_start=int(sample_index[worst_pos]),
            sample_stop=int(sample_index[worst_pos]),
            n_snapshots=1,
        ),
        best_mean_case(prediction, target, sample_index, 30 * args.snapshots_per_day, "Monthly mean"),
        best_mean_case(prediction, target, sample_index, 90 * args.snapshots_per_day, "Seasonal mean"),
        best_mean_case(prediction, target, sample_index, 365 * args.snapshots_per_day, "Annual mean"),
    ]

    draw_figure(cases, args.output_prefix)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.txt")
    write_summary(summary_path, cases, args.predictions)

    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    for case in cases:
        print(
            f"{case.label}: R2={case.r2:.6f}, samples={case.sample_start}-{case.sample_stop}, n={case.n_snapshots}",
            flush=True,
        )


if __name__ == "__main__":
    main()
