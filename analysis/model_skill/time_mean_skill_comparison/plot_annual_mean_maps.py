#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PREDICTIONS = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
MLE_PREDICTIONS = PROJECT_ROOT / "analysis/fk_mle/tuned_mle_parameterization/mle_prediction_physical.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/model_skill/time_mean_skill/annual_mean_maps.png"
WINDOW = 365


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
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


def annual_mean(values: np.ndarray) -> np.ndarray:
    return np.nanmean(values[:WINDOW], axis=0)


def normalized_squared_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ((prediction - target) / np.nanstd(target)) ** 2


def format_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_cbar(fig: plt.Figure, image, cax: plt.Axes, label: str) -> None:
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=6.5, length=2, width=0.55)
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(label, fontsize=8.5, labelpad=1.5)


def draw_figure(
    truth: np.ndarray,
    model: np.ndarray,
    mle: np.ndarray,
) -> None:
    model_error = normalized_squared_error(model, truth)
    mle_error = normalized_squared_error(mle, truth)

    fields = np.concatenate([truth.ravel(), model.ravel(), mle.ravel()])
    field_limit = np.nanpercentile(np.abs(fields), 99)
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)

    errors = np.concatenate([model_error.ravel(), mle_error.ravel()])
    error_norm = Normalize(vmin=0.0, vmax=np.nanpercentile(errors, 99))

    fig = plt.figure(figsize=(7.25, 2.65), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.035], height_ratios=[1.0, 1.0], wspace=0.04)
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])
    cbar_axes = [fig.add_subplot(grid[row, 3]) for row in range(2)]

    field_image = None
    for ax, title, values in (
        (axes[0, 0], "Truth", truth),
        (axes[0, 1], "SubmesoFluxNet", model),
        (axes[0, 2], "MLE parameterization", mle),
    ):
        field_image = ax.imshow(values, origin="lower", cmap="RdBu_r", norm=field_norm, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, pad=4)
        format_map_axis(ax)
    add_cbar(fig, field_image, cbar_axes[0], "Annual-mean\nheat flux (W m$^{-2}$)")

    axes[1, 0].set_axis_off()
    error_image = None
    for ax, title, values in (
        (axes[1, 1], "SubmesoFluxNet", model_error),
        (axes[1, 2], "MLE parameterization", mle_error),
    ):
        error_image = ax.imshow(values, origin="lower", cmap="magma", norm=error_norm, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, pad=4)
        format_map_axis(ax)
    add_cbar(fig, error_image, cbar_axes[1], r"Error $(\Delta/\sigma_y)^2$")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    target, model_prediction, mle_prediction = load_data()
    truth_mean = annual_mean(target)
    model_mean = annual_mean(model_prediction)
    mle_mean = annual_mean(mle_prediction)
    draw_figure(truth_mean, model_mean, mle_mean)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
