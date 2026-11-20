#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PREDICTIONS = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
MLE_PREDICTIONS = PROJECT_ROOT / "analysis/fk_mle/tuned_mle_parameterization/mle_prediction_physical.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/model_skill/time_mean_skill/time_mean_skill_comparison.png"
WINDOWS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 240, 365, 540, 730, 1095]

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


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(MODEL_PREDICTIONS) as data:
        model_sample = data["sample_index"]
        model_prediction = data["prediction"][:, 0]
        target = data["target"][:, 0]

    with np.load(MLE_PREDICTIONS) as data:
        mle_sample = data["sample_index"]
        mle_prediction = data["prediction"][:, 0]

    if not np.array_equal(model_sample, mle_sample):
        raise ValueError("The SubmesoFluxNet and MLE test samples do not match.")

    return target, model_prediction, mle_prediction


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


def calculate_skill(
    target: np.ndarray,
    model_prediction: np.ndarray,
    mle_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = [window for window in WINDOWS if window <= len(target)]
    model_r2 = []
    mle_r2 = []

    for window in windows:
        target_avg = block_average(target, window)
        model_avg = block_average(model_prediction, window)
        mle_avg = block_average(mle_prediction, window)
        model_r2.append(r2_score(model_avg, target_avg))
        mle_r2.append(r2_score(mle_avg, target_avg))

    return np.asarray(windows), np.asarray(model_r2), np.asarray(mle_r2)


def draw_figure(windows: np.ndarray, model_r2: np.ndarray, mle_r2: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 2.75), constrained_layout=True)
    ax.plot(
        windows,
        model_r2,
        color=BLUE,
        marker="o",
        markersize=3.0,
        linewidth=1.35,
        label="SubmesoFluxNet",
    )
    ax.plot(
        windows,
        mle_r2,
        color=RED,
        marker="s",
        markersize=3.0,
        linewidth=1.35,
        label="MLE parameterization",
    )

    markers = ((30, "Monthly"), (90, "Seasonal"), (365, "Annual"))
    ymin = min(float(np.nanmin(model_r2)), float(np.nanmin(mle_r2))) - 0.04
    ymax = min(1.0, max(float(np.nanmax(model_r2)), float(np.nanmax(mle_r2))) + 0.08)
    ax.set_ylim(ymin, ymax)
    for window, label in markers:
        if windows.min() <= window <= windows.max():
            ax.axvline(window, color=GRAY, linewidth=0.65, linestyle=(0, (3, 2)), zorder=0)
            ax.text(
                window,
                ymax - 0.015,
                label,
                rotation=90,
                ha="right",
                va="top",
                fontsize=6.2,
                color=GRAY,
            )

    ax.set_xscale("log")
    ax.set_xlim(float(windows.min()), float(windows.max()))
    ax.set_xlabel("Averaging window (model days)")
    ax.set_ylabel(r"Time-mean global $R^2$")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.55)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.legend(loc="lower right", fontsize=6.7, handlelength=1.7)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    target, model_prediction, mle_prediction = load_data()
    windows, model_r2, mle_r2 = calculate_skill(target, model_prediction, mle_prediction)
    draw_figure(windows, model_r2, mle_r2)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
