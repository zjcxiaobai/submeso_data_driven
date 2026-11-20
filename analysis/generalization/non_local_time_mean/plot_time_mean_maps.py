#!/usr/bin/env python3
"""Plot non-local snapshots and time-mean prediction maps."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NON_LOCAL = PROJECT_ROOT / "analysis/generalization/non_local"
MODEL_PREDICTIONS = NON_LOCAL / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz"
MLE_PREDICTIONS = NON_LOCAL / "evaluation_subdomain_mle_prediction.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/generalization/non_local_time_mean/time_mean_maps.png"

YEAR = 730
MONTH = 60
SEASON = 180
FIELD_PERCENTILE = 99
GRAY = "#343434"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.65,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
    }
)


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction.reshape(len(prediction), -1)
    target = target.reshape(len(target), -1)
    sse = np.sum((prediction - target) ** 2, axis=1)
    sst = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    sse = np.nansum((prediction - target) ** 2)
    sst = np.nansum((target - np.nanmean(target)) ** 2)
    return float(1.0 - sse / sst)


def load_selected_year() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(MODEL_PREDICTIONS) as data:
        model_time = data["time_index"].astype(np.int64)
        model_prediction = data["prediction"][:, 0].astype(np.float64)
        target = data["target"][:, 0].astype(np.float64)

    with np.load(MLE_PREDICTIONS) as data:
        mle_time = data["time_index"].astype(np.int64)
        mle_prediction = data["prediction"][:, 0].astype(np.float64)

    r2 = sample_r2(model_prediction, target)
    year_r2 = [np.nanmean(r2[start : start + YEAR]) for start in range(len(r2) - YEAR + 1)]
    start = int(np.nanargmax(year_r2))
    stop = start + YEAR

    mle_lookup = {int(time): position for position, time in enumerate(mle_time)}
    mle_position = [mle_lookup[int(time)] for time in model_time[start:stop]]
    return model_prediction[start:stop], target[start:stop], mle_prediction[mle_position]


def best_block(prediction: np.ndarray, target: np.ndarray, length: int) -> np.ndarray:
    starts = np.arange(0, len(prediction) - length + 1, length)
    scores = [
        r2_score(prediction[start : start + length].mean(0), target[start : start + length].mean(0))
        for start in starts
    ]
    start = int(starts[np.nanargmax(scores)])
    return np.arange(start, start + length)


def make_rows(
    model_prediction: np.ndarray,
    target: np.ndarray,
    mle_prediction: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    model_r2 = sample_r2(model_prediction, target)
    mle_r2 = sample_r2(mle_prediction, target)
    annual = np.arange(YEAR)
    return [
        ("Best snapshot", np.array([np.nanargmax(model_r2)]), np.array([np.nanargmax(mle_r2)]), ""),
        ("Worst snapshot", np.array([np.nanargmin(model_r2)]), np.array([np.nanargmin(mle_r2)]), ""),
        ("Monthly mean", best_block(model_prediction, target, MONTH), best_block(mle_prediction, target, MONTH), "30 model days"),
        ("Seasonal mean", best_block(model_prediction, target, SEASON), best_block(mle_prediction, target, SEASON), "90 model days"),
        ("Annual mean", annual, annual, "365 model days"),
    ]


def hide_axes(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_figure(
    rows: list[tuple[str, np.ndarray, np.ndarray, str]],
    model_prediction: np.ndarray,
    target: np.ndarray,
    mle_prediction: np.ndarray,
) -> None:
    maps = []
    for _, model_indices, mle_indices, _ in rows:
        maps.append(
            (
                target[model_indices].mean(0),
                model_prediction[model_indices].mean(0),
                target[mle_indices].mean(0),
                mle_prediction[mle_indices].mean(0),
            )
        )

    values = np.concatenate([array.ravel() for row_maps in maps for array in row_maps])
    limit = float(np.nanpercentile(np.abs(values), FIELD_PERCENTILE))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig = plt.figure(figsize=(11, 7.2))
    grid = fig.add_gridspec(
        5,
        6,
        width_ratios=[0.90, 1.12, 1.12, 1.12, 1.12, 0.045],
        left=0.045,
        right=0.955,
        bottom=0.04,
        top=0.955,
        wspace=0.045,
        hspace=0.36,
    )
    titles = ["Truth-SubmesoFluxNet", "SubmesoFluxNet", "Truth-MLE", "MLE parameterization"]

    for row_number, ((label, _, _, detail), row_maps) in enumerate(zip(rows, maps)):
        label_axis = fig.add_subplot(grid[row_number, 0])
        label_axis.text(
            0.98,
            0.5,
            label + (f"\n{detail}" if detail else ""),
            ha="right",
            va="center",
            fontsize=8.2,
            transform=label_axis.transAxes,
        )
        label_axis.set_axis_off()

        image = None
        for column, (title, field) in enumerate(zip(titles, row_maps), start=1):
            axis = fig.add_subplot(grid[row_number, column])
            image = axis.imshow(field, origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
            hide_axes(axis)
            if row_number == 0:
                axis.set_title(title, fontsize=9.2, pad=4)

        color_axis = fig.add_subplot(grid[row_number, 5])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.ax.tick_params(labelsize=7, length=2.2, width=0.55, colors=GRAY)
        colorbar.outline.set_linewidth(0.55)
        colorbar.set_label(r"$w'T'$", fontsize=7.2, color=GRAY, labelpad=3)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    model_prediction, target, mle_prediction = load_selected_year()
    rows = make_rows(model_prediction, target, mle_prediction)
    draw_figure(rows, model_prediction, target, mle_prediction)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
