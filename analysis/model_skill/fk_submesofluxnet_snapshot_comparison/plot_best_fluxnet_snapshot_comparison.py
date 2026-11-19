#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm


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
        description=(
            "Compare a selected SubmesoFluxNet/FK test snapshot using the current model_analysis outputs, "
            "including joint histograms, normalized squared errors, and temporal correlation maps."
        )
    )
    parser.add_argument(
        "--fluxnet-predictions",
        type=Path,
        default=PROJECT_DIR / "results/final_model/test_predictions_denorm.npz",
    )
    parser.add_argument(
        "--fk-predictions",
        type=Path,
        default=PROJECT_DIR / "analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_prediction_physical.npz",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "best_mle_snapshot_comparison",
    )
    parser.add_argument("--summary-output", type=Path, default=HERE / "best_mle_snapshot_comparison_summary.txt")
    parser.add_argument("--data-output", type=Path, default=HERE / "best_mle_snapshot_comparison_data.npz")
    parser.add_argument("--joint-output-prefix", type=Path, default=HERE / "best_mle_snapshot_joint_histograms")
    parser.add_argument("--field-percentile", type=float, default=99.0)
    parser.add_argument("--error-percentile", type=float, default=99.0)
    parser.add_argument("--hist-bins", type=int, default=80)
    parser.add_argument("--selection", choices=("best-fk", "best-fluxnet"), default="best-fk", help="Snapshot to plot: best FK/MLE R2 or best SubmesoFluxNet R2.")
    parser.add_argument("--fk-display-label", default="MLE parameterization")
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
            "axes.linewidth": 0.65,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
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


def load_fk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"Missing FK prediction file: {path}")
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


def align_fk_to_fluxnet(
    flux_sample: np.ndarray,
    fk_sample: np.ndarray,
    fk_prediction: np.ndarray,
    fk_time_index: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if flux_sample.shape != fk_sample.shape or not np.array_equal(flux_sample, fk_sample):
        flux_set = set(map(int, flux_sample))
        fk_set = set(map(int, fk_sample))
        raise ValueError(
            "SubmesoFluxNet and MLE sample_index values do not match exactly: "
            f"fluxnet_rows={flux_sample.size}, mle_rows={fk_sample.size}, common={len(flux_set & fk_set)}, "
            f"first_missing_from_mle={sorted(flux_set - fk_set)[:5]}, "
            f"first_missing_from_fluxnet={sorted(fk_set - flux_set)[:5]}"
        )
    return fk_prediction, fk_time_index


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def spatial_corr(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.ravel()
    truth = target.ravel()
    pred = pred - np.nanmean(pred)
    truth = truth - np.nanmean(truth)
    denom = float(np.sqrt(np.nansum(pred**2) * np.nansum(truth**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.nansum(pred * truth) / denom)


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


def add_cbar(fig: plt.Figure, image, cax: plt.Axes, label: str) -> None:
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=5.8, length=2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(label, fontsize=5.9, color=PALETTE["neutral_dark"], labelpad=1.5)


def log_magnitude_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.abs(np.asarray(array).ravel()) for array in arrays])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0e-6, 1.0
    low = float(np.nanpercentile(values, 1.0))
    high = float(np.nanpercentile(values, 99.5))
    if not np.isfinite(low) or low <= 0.0:
        low = float(np.nanmin(values[values > 0.0]))
    if not np.isfinite(high) or high <= low:
        high = float(np.nanmax(values))
    if high <= low:
        high = low * 10.0
    return low, high


def draw_signed_joint_hist(
    fig: plt.Figure,
    ax: plt.Axes,
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str,
    method_label: str,
    sign: str,
    bins: int,
    limits: tuple[float, float],
) -> int:
    if sign == "negative":
        mask = (truth < 0.0) & (prediction < 0.0)
    elif sign == "positive":
        mask = (truth > 0.0) & (prediction > 0.0)
    else:
        raise ValueError(f"Unknown sign: {sign}")

    x = np.abs(truth[mask].ravel())
    y = np.abs(prediction[mask].ravel())
    low, high = limits
    edges = np.logspace(np.log10(low), np.log10(high), bins + 1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.plot([low, high], [low, high], color="#808080", linewidth=0.85, linestyle=(0, (4, 2)), zorder=3)
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.set_xlabel(r"Truth $|q|$")
    ax.set_ylabel(fr"{method_label} $|q|$")
    ax.grid(False)

    if x.size == 0:
        ax.text(0.5, 0.5, "No same-sign\npoints", transform=ax.transAxes, ha="center", va="center", fontsize=6.5)
        return 0

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[edges, edges])
    hist = hist.T
    hist[hist <= 0] = np.nan
    vmax = float(np.nanmax(hist)) if np.isfinite(np.nanmax(hist)) else 1.0
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        hist,
        cmap="Blues",
        norm=LogNorm(vmin=1.0, vmax=max(vmax, 1.0)),
        shading="auto",
        zorder=1,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.018)
    cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Count", fontsize=5.8, color=PALETTE["neutral_dark"], labelpad=1)
    return int(x.size)


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_prefix}.png"), dpi=600, bbox_inches="tight")


def add_vertical_divider(fig: plt.Figure, left_axes: list[plt.Axes], right_axes: list[plt.Axes]) -> None:
    fig.canvas.draw()
    left_right = max(ax.get_position().x1 for ax in left_axes)
    right_left = min(ax.get_position().x0 for ax in right_axes)
    x = 0.5 * (left_right + right_left)
    y0 = min(ax.get_position().y0 for ax in left_axes + right_axes)
    y1 = max(ax.get_position().y1 for ax in left_axes + right_axes)
    line = mpl.lines.Line2D([x, x], [y0, y1], transform=fig.transFigure, color="black", linewidth=0.85)
    fig.add_artist(line)


def draw_main_figure(
    *,
    left_sample_index: int,
    left_time_index: int | None,
    left_fluxnet_r2: float,
    left_mle_r2: float,
    left_truth: np.ndarray,
    left_fluxnet_prediction: np.ndarray,
    left_fluxnet_error: np.ndarray,
    right_sample_index: int,
    right_time_index: int | None,
    right_fluxnet_r2: float,
    right_mle_r2: float,
    right_truth: np.ndarray,
    right_mle_prediction: np.ndarray,
    right_mle_error: np.ndarray,
    fluxnet_temporal_corr: np.ndarray,
    mle_temporal_corr: np.ndarray,
    field_percentile: float,
    error_percentile: float,
    output_prefix: Path,
    fk_label: str = "MLE parameterization",
) -> None:
    apply_style()
    field_limit = robust_symmetric_limit(
        np.concatenate(
            [
                left_truth.ravel(),
                left_fluxnet_prediction.ravel(),
                right_truth.ravel(),
                right_mle_prediction.ravel(),
            ]
        ),
        percentile=field_percentile,
        min_limit=1.0,
    )
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)

    error_values = np.concatenate([left_fluxnet_error.ravel(), right_mle_error.ravel()])
    error_limit = float(np.nanpercentile(error_values, error_percentile))
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

    save_figure(fig, output_prefix)
    plt.close(fig)

def draw_joint_hist_figure(
    *,
    truth_best: np.ndarray,
    fluxnet_best: np.ndarray,
    fk_best: np.ndarray,
    hist_bins: int,
    output_prefix: Path,
    fk_label: str = "MLE parameterization",
) -> None:
    apply_style()
    hist_limits = log_magnitude_limits(truth_best, fluxnet_best, fk_best)
    fig, axes = plt.subplots(2, 2, figsize=(5.15, 4.65), constrained_layout=True)
    panels = (
        (axes[0, 0], truth_best, fluxnet_best, "SubmesoFluxNet negative", "SubmesoFluxNet", "negative"),
        (axes[0, 1], truth_best, fluxnet_best, "SubmesoFluxNet positive", "SubmesoFluxNet", "positive"),
        (axes[1, 0], truth_best, fk_best, f"{fk_label} negative", fk_label, "negative"),
        (axes[1, 1], truth_best, fk_best, f"{fk_label} positive", fk_label, "positive"),
    )
    for ax, truth, prediction, title, method_label, sign in panels:
        draw_signed_joint_hist(
            fig,
            ax,
            truth,
            prediction,
            title=title,
            method_label=method_label,
            sign=sign,
            bins=hist_bins,
            limits=hist_limits,
        )
    fig.suptitle("Signed log-magnitude joint histograms", fontsize=8.4, y=1.004)
    save_figure(fig, output_prefix)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    best_fluxnet_sample_index: int,
    best_fluxnet_test_position: int,
    best_fluxnet_time_index: int | None,
    best_fluxnet_r2: float,
    best_fluxnet_mle_r2: float,
    best_fluxnet_spatial_corr: float,
    best_fluxnet_mle_spatial_corr: float,
    best_fluxnet_error: np.ndarray,
    best_mle_sample_index: int,
    best_mle_test_position: int,
    best_mle_time_index: int | None,
    best_mle_fluxnet_r2: float,
    best_mle_r2: float,
    best_mle_fluxnet_spatial_corr: float,
    best_mle_spatial_corr: float,
    best_mle_error: np.ndarray,
    fluxnet_temporal_corr: np.ndarray,
    mle_temporal_corr: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Two-column best-snapshot comparison: SubmesoFluxNet versus MLE parameterization\n")
        handle.write("layout: left column is maximum SubmesoFluxNet R2; right column is maximum MLE parameterization R2\n")
        handle.write("rows: truth, prediction, normalized squared error, temporal correlation\n")
        handle.write("normalized_squared_error: ((prediction - truth) / std(truth_selected_snapshot))^2\n")
        handle.write("temporal_correlation: pixelwise Pearson correlation over the aligned final test time series\n")

        handle.write("\nBest SubmesoFluxNet column\n")
        handle.write(f"sample_index: {best_fluxnet_sample_index}\n")
        handle.write(f"test_position: {best_fluxnet_test_position}\n")
        if best_fluxnet_time_index is not None:
            handle.write(f"time_index: {best_fluxnet_time_index}\n")
        handle.write(f"submesofluxnet_snapshot_r2: {best_fluxnet_r2:.10g}\n")
        handle.write(f"mle_snapshot_r2_same_sample: {best_fluxnet_mle_r2:.10g}\n")
        handle.write(f"submesofluxnet_snapshot_spatial_corr: {best_fluxnet_spatial_corr:.10g}\n")
        handle.write(f"mle_snapshot_spatial_corr_same_sample: {best_fluxnet_mle_spatial_corr:.10g}\n")
        handle.write(f"submesofluxnet_nse_mean: {float(np.nanmean(best_fluxnet_error)):.10g}\n")

        handle.write("\nBest MLE parameterization column\n")
        handle.write(f"sample_index: {best_mle_sample_index}\n")
        handle.write(f"test_position: {best_mle_test_position}\n")
        if best_mle_time_index is not None:
            handle.write(f"time_index: {best_mle_time_index}\n")
        handle.write(f"submesofluxnet_snapshot_r2_same_sample: {best_mle_fluxnet_r2:.10g}\n")
        handle.write(f"mle_snapshot_r2: {best_mle_r2:.10g}\n")
        handle.write(f"submesofluxnet_snapshot_spatial_corr_same_sample: {best_mle_fluxnet_spatial_corr:.10g}\n")
        handle.write(f"mle_snapshot_spatial_corr: {best_mle_spatial_corr:.10g}\n")
        handle.write(f"mle_nse_mean: {float(np.nanmean(best_mle_error)):.10g}\n")

        handle.write("\nTemporal correlation maps\n")
        handle.write(f"submesofluxnet_temporal_corr_mean: {float(np.nanmean(fluxnet_temporal_corr)):.10g}\n")
        handle.write(f"mle_temporal_corr_mean: {float(np.nanmean(mle_temporal_corr)):.10g}\n")

def main() -> None:
    args = parse_args()
    flux_sample, flux_prediction, target, flux_r2 = load_fluxnet(args.fluxnet_predictions)
    fk_sample, fk_prediction, fk_time_index = load_fk(args.fk_predictions)
    fk_aligned, fk_time_aligned = align_fk_to_fluxnet(flux_sample, fk_sample, fk_prediction, fk_time_index)

    fk_r2_series = np.asarray([r2_score(fk_aligned[n], target[n]) for n in range(target.shape[0])], dtype=np.float64)
    best_fluxnet_pos = int(np.nanargmax(flux_r2))
    best_mle_pos = int(np.nanargmax(fk_r2_series))

    fluxnet_temporal_corr = temporal_corr_map(flux_prediction, target)
    mle_temporal_corr = temporal_corr_map(fk_aligned, target)

    left_truth = target[best_fluxnet_pos]
    left_fluxnet = flux_prediction[best_fluxnet_pos]
    left_mle = fk_aligned[best_fluxnet_pos]
    left_fluxnet_error = normalized_squared_error(left_fluxnet, left_truth)

    right_truth = target[best_mle_pos]
    right_fluxnet = flux_prediction[best_mle_pos]
    right_mle = fk_aligned[best_mle_pos]
    right_mle_error = normalized_squared_error(right_mle, right_truth)

    left_time = int(fk_time_aligned[best_fluxnet_pos]) if fk_time_aligned is not None else None
    right_time = int(fk_time_aligned[best_mle_pos]) if fk_time_aligned is not None else None

    draw_main_figure(
        left_sample_index=int(flux_sample[best_fluxnet_pos]),
        left_time_index=left_time,
        left_fluxnet_r2=float(flux_r2[best_fluxnet_pos]),
        left_mle_r2=float(fk_r2_series[best_fluxnet_pos]),
        left_truth=left_truth,
        left_fluxnet_prediction=left_fluxnet,
        left_fluxnet_error=left_fluxnet_error,
        right_sample_index=int(flux_sample[best_mle_pos]),
        right_time_index=right_time,
        right_fluxnet_r2=float(flux_r2[best_mle_pos]),
        right_mle_r2=float(fk_r2_series[best_mle_pos]),
        right_truth=right_truth,
        right_mle_prediction=right_mle,
        right_mle_error=right_mle_error,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        mle_temporal_corr=mle_temporal_corr,
        field_percentile=args.field_percentile,
        error_percentile=args.error_percentile,
        output_prefix=args.output_prefix,
        fk_label=args.fk_display_label,
    )

    draw_joint_hist_figure(
        truth_best=right_truth,
        fluxnet_best=right_fluxnet,
        fk_best=right_mle,
        hist_bins=args.hist_bins,
        output_prefix=args.joint_output_prefix,
        fk_label=args.fk_display_label,
    )

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data_output,
        best_submesofluxnet_sample_index=np.asarray([int(flux_sample[best_fluxnet_pos])], dtype=np.int64),
        best_submesofluxnet_test_position=np.asarray([best_fluxnet_pos], dtype=np.int64),
        best_submesofluxnet_time_index=np.asarray([left_time], dtype=np.int64) if left_time is not None else np.asarray([], dtype=np.int64),
        best_submesofluxnet_truth=left_truth.astype(np.float32),
        best_submesofluxnet_prediction=left_fluxnet.astype(np.float32),
        best_submesofluxnet_mle_prediction_same_sample=left_mle.astype(np.float32),
        best_submesofluxnet_normalized_squared_error=left_fluxnet_error.astype(np.float32),
        best_mle_sample_index=np.asarray([int(flux_sample[best_mle_pos])], dtype=np.int64),
        best_mle_test_position=np.asarray([best_mle_pos], dtype=np.int64),
        best_mle_time_index=np.asarray([right_time], dtype=np.int64) if right_time is not None else np.asarray([], dtype=np.int64),
        best_mle_truth=right_truth.astype(np.float32),
        best_mle_prediction=right_mle.astype(np.float32),
        best_mle_submesofluxnet_prediction_same_sample=right_fluxnet.astype(np.float32),
        best_mle_normalized_squared_error=right_mle_error.astype(np.float32),
        submesofluxnet_temporal_corr=fluxnet_temporal_corr.astype(np.float32),
        mle_temporal_corr=mle_temporal_corr.astype(np.float32),
    )

    write_summary(
        args.summary_output,
        best_fluxnet_sample_index=int(flux_sample[best_fluxnet_pos]),
        best_fluxnet_test_position=best_fluxnet_pos,
        best_fluxnet_time_index=left_time,
        best_fluxnet_r2=float(flux_r2[best_fluxnet_pos]),
        best_fluxnet_mle_r2=float(fk_r2_series[best_fluxnet_pos]),
        best_fluxnet_spatial_corr=spatial_corr(left_fluxnet, left_truth),
        best_fluxnet_mle_spatial_corr=spatial_corr(left_mle, left_truth),
        best_fluxnet_error=left_fluxnet_error,
        best_mle_sample_index=int(flux_sample[best_mle_pos]),
        best_mle_test_position=best_mle_pos,
        best_mle_time_index=right_time,
        best_mle_fluxnet_r2=float(flux_r2[best_mle_pos]),
        best_mle_r2=float(fk_r2_series[best_mle_pos]),
        best_mle_fluxnet_spatial_corr=spatial_corr(right_fluxnet, right_truth),
        best_mle_spatial_corr=spatial_corr(right_mle, right_truth),
        best_mle_error=right_mle_error,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        mle_temporal_corr=mle_temporal_corr,
    )

    print(f"Saved {args.output_prefix}.png", flush=True)
    print(f"Saved {args.joint_output_prefix}.png", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.data_output}", flush=True)
    print(f"Best SubmesoFluxNet sample: {int(flux_sample[best_fluxnet_pos])}", flush=True)
    print(f"  SubmesoFluxNet R2: {float(flux_r2[best_fluxnet_pos]):.6f}", flush=True)
    print(f"  {args.fk_display_label} R2 on same sample: {float(fk_r2_series[best_fluxnet_pos]):.6f}", flush=True)
    print(f"Best {args.fk_display_label} sample: {int(flux_sample[best_mle_pos])}", flush=True)
    print(f"  SubmesoFluxNet R2 on same sample: {float(flux_r2[best_mle_pos]):.6f}", flush=True)
    print(f"  {args.fk_display_label} R2: {float(fk_r2_series[best_mle_pos]):.6f}", flush=True)


if __name__ == "__main__":
    main()
