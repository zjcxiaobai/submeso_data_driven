#!/usr/bin/env python3
"""Plot non-local prediction skill versus temporal averaging window."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NON_LOCAL = PROJECT_ROOT / "analysis/generalization/non_local"
MODEL_PREDICTIONS = NON_LOCAL / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz"
MLE_PREDICTIONS = NON_LOCAL / "evaluation_subdomain_mle_prediction.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/generalization/non_local_time_mean/time_mean_skill_comparison.png"

YEAR = 730
SNAPSHOTS_PER_DAY = 2
WINDOW_DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 240, 365]
BLUE = "#1F5A85"
RED = "#B64A3B"
GRAY = "#777777"
LIGHT_GRAY = "#D6D6D6"


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


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction.reshape(len(prediction), -1)
    target = target.reshape(len(target), -1)
    sse = np.sum((prediction - target) ** 2, axis=1)
    sst = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


def load_selected_year() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(MODEL_PREDICTIONS) as data:
        model_time = data["time_index"].astype(np.int64)
        model_prediction = data["prediction"][:, 0].astype(np.float64)
        target = data["target"][:, 0].astype(np.float64)

    with np.load(MLE_PREDICTIONS) as data:
        mle_time = data["time_index"].astype(np.int64)
        mle_prediction = data["prediction"][:, 0].astype(np.float64)

    r2 = sample_r2(model_prediction, target)
    means = [np.nanmean(r2[start : start + YEAR]) for start in range(len(r2) - YEAR + 1)]
    start = int(np.nanargmax(means))
    stop = start + YEAR

    mle_lookup = {int(time): position for position, time in enumerate(mle_time)}
    mle_position = [mle_lookup[int(time)] for time in model_time[start:stop]]
    return model_prediction[start:stop], target[start:stop], mle_prediction[mle_position]


def block_mean(values: np.ndarray, window: int) -> np.ndarray:
    blocks = len(values) // window
    values = values[: blocks * window]
    return np.nanmean(values.reshape(blocks, window, *values.shape[1:]), axis=1)


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    sse = np.nansum((prediction - target) ** 2)
    sst = np.nansum((target - np.nanmean(target)) ** 2)
    return float(1.0 - sse / sst)


def calculate_skill(
    model_prediction: np.ndarray,
    target: np.ndarray,
    mle_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_r2 = []
    mle_r2 = []
    for days in WINDOW_DAYS:
        window = days * SNAPSHOTS_PER_DAY
        target_mean = block_mean(target, window)
        model_r2.append(r2_score(block_mean(model_prediction, window), target_mean))
        mle_r2.append(r2_score(block_mean(mle_prediction, window), target_mean))
    return np.asarray(WINDOW_DAYS), np.asarray(model_r2), np.asarray(mle_r2)


def draw_figure(days: np.ndarray, model_r2: np.ndarray, mle_r2: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 2.75), constrained_layout=True)
    ax.plot(days, model_r2, color=BLUE, marker="o", markersize=3, linewidth=1.35, label="SubmesoFluxNet")
    ax.plot(days, mle_r2, color=RED, marker="s", markersize=3, linewidth=1.35, label="MLE parameterization")

    ymin = min(model_r2.min(), mle_r2.min()) - 0.04
    ymax = min(1.0, max(model_r2.max(), mle_r2.max()) + 0.08)
    ax.set_ylim(ymin, ymax)
    for window, label in ((30, "Monthly"), (90, "Seasonal"), (365, "Annual")):
        ax.axvline(window, color=GRAY, linewidth=0.65, linestyle=(0, (3, 2)))
        ax.text(window, ymax - 0.015, label, rotation=90, ha="right", va="top", fontsize=6.2, color=GRAY)

    ax.set_xscale("log")
    ax.set_xlim(days.min(), days.max())
    ax.set_xlabel("Averaging window (model days)")
    ax.set_ylabel(r"Time-mean global $R^2$")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.55)
    ax.legend(loc="lower right", fontsize=6.7, handlelength=1.7)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    model_prediction, target, mle_prediction = load_selected_year()
    days, model_r2, mle_r2 = calculate_skill(model_prediction, target, mle_prediction)
    draw_figure(days, model_r2, mle_r2)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
