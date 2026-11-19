#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLUXNET_PREDICTIONS = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
MLE_PREDICTIONS = PROJECT_ROOT / "analysis/fk_mle/tuned_mle_parameterization/mle_prediction_physical.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/model_skill/mle_submesofluxnet_comparison/best_snapshot_comparison.png"
FIELD_PERCENTILE = 99.0
ERROR_PERCENTILE = 99.0

PALETTE = {
    "neutral_dark": "#343434",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.65,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_fluxnet(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing SubmesoFluxNet prediction file: {path}")
    with np.load(path) as data:
        required = {"sample_index", "prediction", "target", "r2"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected SubmesoFluxNet prediction shape (N,1,H,W), got {prediction.shape}")
    return sample, prediction[:, 0], target[:, 0], r2


def load_mle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing MLE prediction file: {path}")
    with np.load(path) as data:
        required = {"sample_index", "prediction"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected MLE prediction shape (N,1,H,W), got {prediction.shape}")
    return sample, prediction[:, 0]


def align_mle_to_fluxnet(
    flux_sample: np.ndarray,
    mle_sample: np.ndarray,
    mle_prediction: np.ndarray,
) -> np.ndarray:
    if flux_sample.shape != mle_sample.shape or not np.array_equal(flux_sample, mle_sample):
        flux_set = set(map(int, flux_sample))
        mle_set = set(map(int, mle_sample))
        raise ValueError(
            "SubmesoFluxNet and MLE sample_index values do not match exactly: "
            f"fluxnet_rows={flux_sample.size}, mle_rows={mle_sample.size}, common={len(flux_set & mle_set)}, "
            f"first_missing_from_mle={sorted(flux_set - mle_set)[:5]}, "
            f"first_missing_from_fluxnet={sorted(mle_set - flux_set)[:5]}"
        )
    return mle_prediction


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def temporal_corr_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_anom = prediction - np.nanmean(prediction, axis=0, keepdims=True)
    target_anom = target - np.nanmean(target, axis=0, keepdims=True)
    numerator = np.nansum(pred_anom * target_anom, axis=0, dtype=np.float64)
    pred_ss = np.nansum(pred_anom**2, axis=0, dtype=np.float64)
    target_ss = np.nansum(target_anom**2, axis=0, dtype=np.float64)
    denom = np.sqrt(pred_ss * target_ss)
    out = np.full(target.shape[1:], np.nan, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    return out


def normalized_squared_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_std = float(np.nanstd(target))
    if not np.isfinite(target_std) or target_std <= 0.0:
        target_std = 1.0
    return ((prediction - target) / target_std) ** 2


def robust_symmetric_limit(values: np.ndarray, percentile: float, min_limit: float = 1.0) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return min_limit
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(np.abs(finite)))
    return max(limit, min_limit)


def format_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_main_figure(
    *,
    left_truth: np.ndarray,
    left_fluxnet_prediction: np.ndarray,
    left_fluxnet_error: np.ndarray,
    right_truth: np.ndarray,
    right_mle_prediction: np.ndarray,
    right_mle_error: np.ndarray,
    fluxnet_temporal_corr: np.ndarray,
    mle_temporal_corr: np.ndarray,
) -> None:
    field_limit = robust_symmetric_limit(
        np.concatenate(
            [
                left_truth.ravel(),
                left_fluxnet_prediction.ravel(),
                right_truth.ravel(),
                right_mle_prediction.ravel(),
            ]
        ),
        percentile=FIELD_PERCENTILE,
        min_limit=1.0,
    )
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)

    error_values = np.concatenate([left_fluxnet_error.ravel(), right_mle_error.ravel()])
    error_limit = float(np.nanpercentile(error_values, ERROR_PERCENTILE))
    if not np.isfinite(error_limit) or error_limit <= 0:
        error_limit = float(np.nanmax(error_values))
    error_norm = Normalize(vmin=0.0, vmax=max(error_limit, 1.0e-6))
    corr_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    panel_title_fs = 8.5
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 2.75), constrained_layout=False)
    panels = (
        (axes[0, 0], left_truth, "RdBu_r", field_norm),
        (axes[0, 1], left_fluxnet_prediction, "RdBu_r", field_norm),
        (axes[0, 2], left_fluxnet_error, "magma", error_norm),
        (axes[0, 3], fluxnet_temporal_corr, "RdBu_r", corr_norm),
        (axes[1, 0], right_truth, "RdBu_r", field_norm),
        (axes[1, 1], right_mle_prediction, "RdBu_r", field_norm),
        (axes[1, 2], right_mle_error, "magma", error_norm),
        (axes[1, 3], mle_temporal_corr, "RdBu_r", corr_norm),
    )
    images = []
    for ax, values, cmap, norm in panels:
        images.append(ax.imshow(values, origin="lower", cmap=cmap, norm=norm, interpolation="nearest"))
        format_map_axis(ax)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.895, bottom=0.205, wspace=0.035, hspace=0.24)

    for ax, title in zip(axes[0], ("Truth", "Prediction", "Error", "Temporal correlation")):
        ax.set_title(title, fontsize=panel_title_fs, pad=3)

    axes[0, 0].set_ylabel("SubmesoFluxNet", fontsize=panel_title_fs, color=PALETTE["neutral_dark"], labelpad=4)
    axes[1, 0].set_ylabel("MLE\nparameterization", fontsize=panel_title_fs, color=PALETTE["neutral_dark"], labelpad=4)

    # Place compact manual colorbars without changing the equal-sized map panels.
    fig.canvas.draw()
    heat_left = axes[1, 0].get_position()
    heat_right = axes[1, 1].get_position()
    heat_cax = fig.add_axes([heat_left.x0, heat_left.y0 - 0.055, heat_right.x1 - heat_left.x0, 0.018])
    single_col_items = (
        (images[2], axes[1, 2], r"Error $(\Delta/\sigma_y)^2$"),
        (images[3], axes[1, 3], r"Pearson $\rho$"),
    )
    cbar = fig.colorbar(images[1], cax=heat_cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=5.4, length=2, width=0.5, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.5)
    cbar.set_label(r"Heat flux (W m$^{-2}$)", fontsize=panel_title_fs, color=PALETTE["neutral_dark"], labelpad=0.5)
    for image, anchor_ax, label in single_col_items:
        pos = anchor_ax.get_position()
        cax = fig.add_axes([pos.x0, pos.y0 - 0.055, pos.width, 0.018])
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=5.4, length=2, width=0.5, colors=PALETTE["neutral_dark"])
        cbar.outline.set_linewidth(0.5)
        cbar.set_label(label, fontsize=panel_title_fs, color=PALETTE["neutral_dark"], labelpad=0.5)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    flux_sample, flux_prediction, target, flux_r2 = load_fluxnet(FLUXNET_PREDICTIONS)
    mle_sample, mle_prediction = load_mle(MLE_PREDICTIONS)
    mle_aligned = align_mle_to_fluxnet(flux_sample, mle_sample, mle_prediction)

    mle_r2 = np.asarray(
        [r2_score(mle_aligned[n], target[n]) for n in range(target.shape[0])],
        dtype=np.float64,
    )
    best_fluxnet_pos = int(np.nanargmax(flux_r2))
    best_mle_pos = int(np.nanargmax(mle_r2))

    fluxnet_temporal_corr = temporal_corr_map(flux_prediction, target)
    mle_temporal_corr = temporal_corr_map(mle_aligned, target)

    left_truth = target[best_fluxnet_pos]
    left_fluxnet = flux_prediction[best_fluxnet_pos]
    left_fluxnet_error = normalized_squared_error(left_fluxnet, left_truth)

    right_truth = target[best_mle_pos]
    right_mle = mle_aligned[best_mle_pos]
    right_mle_error = normalized_squared_error(right_mle, right_truth)

    draw_main_figure(
        left_truth=left_truth,
        left_fluxnet_prediction=left_fluxnet,
        left_fluxnet_error=left_fluxnet_error,
        right_truth=right_truth,
        right_mle_prediction=right_mle,
        right_mle_error=right_mle_error,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        mle_temporal_corr=mle_temporal_corr,
    )

    print(f"Saved {OUTPUT}")
    print(f"Best SubmesoFluxNet sample: {int(flux_sample[best_fluxnet_pos])}")
    print(f"Best MLE sample: {int(flux_sample[best_mle_pos])}")


if __name__ == "__main__":
    main()
