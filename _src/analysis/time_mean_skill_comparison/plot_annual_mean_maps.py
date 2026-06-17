#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

PALETTE = {
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot annual-mean truth, model predictions, and normalized errors.")
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
    parser.add_argument("--window", type=int, default=365, help="Averaging window in model days/snapshots.")
    parser.add_argument("--block-index", type=int, default=0, help="Annual block to plot after trimming to full blocks.")
    parser.add_argument("--field-percentile", type=float, default=99.0)
    parser.add_argument("--error-percentile", type=float, default=99.0)
    parser.add_argument("--output-prefix", type=Path, default=HERE / "annual_mean_maps")
    parser.add_argument("--summary-output", type=Path, default=HERE / "annual_mean_maps_summary.txt")
    parser.add_argument("--data-output", type=Path, default=HERE / "annual_mean_maps_data.npz")
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
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
        flux_sample[flux_pos_array],
        flux_prediction[flux_pos_array],
        target[flux_pos_array],
        fk_prediction[fk_pos_array],
        aligned_time,
    )


def annual_block_mean(values: np.ndarray, window: int, block_index: int) -> np.ndarray:
    n_blocks = values.shape[0] // window
    if n_blocks < 1:
        raise ValueError(f"Window {window} is longer than the aligned time series length {values.shape[0]}")
    if block_index < 0 or block_index >= n_blocks:
        raise ValueError(f"block_index must be in [0, {n_blocks - 1}], got {block_index}")
    start = block_index * window
    stop = start + window
    return np.nanmean(values[start:stop], axis=0)


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


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


def add_cbar(fig: plt.Figure, image, axes, label: str) -> None:
    cbar = fig.colorbar(image, ax=axes, fraction=0.04, pad=0.018)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(label, fontsize=6.3, color=PALETTE["neutral_dark"], labelpad=2)


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png", ".tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(Path(f"{output_prefix}{ext}"), **kwargs)


def draw_figure(
    *,
    truth: np.ndarray,
    fluxnet: np.ndarray,
    fk: np.ndarray,
    fluxnet_error: np.ndarray,
    fk_error: np.ndarray,
    field_percentile: float,
    error_percentile: float,
    output_prefix: Path,
) -> None:
    apply_style()
    field_limit = robust_symmetric_limit(
        np.concatenate([truth.ravel(), fluxnet.ravel(), fk.ravel()]),
        percentile=field_percentile,
        min_limit=1.0,
    )
    field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)

    error_values = np.concatenate([fluxnet_error.ravel(), fk_error.ravel()])
    error_limit = float(np.nanpercentile(error_values, error_percentile))
    if not np.isfinite(error_limit) or error_limit <= 0:
        error_limit = float(np.nanmax(error_values))
    error_norm = Normalize(vmin=0.0, vmax=max(error_limit, 1.0e-6))

    fig = plt.figure(figsize=(6.1, 3.65), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0])
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])

    field_image = None
    for ax, title, values in (
        (axes[0, 0], "Truth", truth),
        (axes[0, 1], "SubmesoFluxNet", fluxnet),
        (axes[0, 2], "FK", fk),
    ):
        field_image = ax.imshow(values, origin="lower", cmap="RdBu_r", norm=field_norm, interpolation="nearest")
        ax.set_title(title, fontsize=7.5, pad=4)
        format_map_axis(ax)
    add_cbar(fig, field_image, [axes[0, 0], axes[0, 1], axes[0, 2]], r"Annual-mean heat flux (W m$^{-2}$)")

    axes[1, 0].set_axis_off()
    error_image = None
    for ax, title, values in (
        (axes[1, 1], "SubmesoFluxNet", fluxnet_error),
        (axes[1, 2], "FK", fk_error),
    ):
        error_image = ax.imshow(values, origin="lower", cmap="magma", norm=error_norm, interpolation="nearest")
        ax.set_title(title, fontsize=7.5, pad=4)
        format_map_axis(ax)
    add_cbar(fig, error_image, [axes[1, 1], axes[1, 2]], r"Normalized squared error $(\Delta/\sigma_y)^2$")

    save_figure(fig, output_prefix)
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    window: int,
    block_index: int,
    start_sample: int,
    end_sample: int,
    start_time: int | None,
    end_time: int | None,
    fluxnet_r2: float,
    fk_r2: float,
    fluxnet_error: np.ndarray,
    fk_error: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Annual-mean map comparison\n")
        handle.write("selection: first complete annual block by default\n")
        handle.write("normalized_squared_error: ((prediction - truth) / std(annual_mean_truth))^2\n")
        handle.write(f"window_model_days: {window}\n")
        handle.write(f"block_index: {block_index}\n")
        handle.write(f"start_sample_index: {start_sample}\n")
        handle.write(f"end_sample_index: {end_sample}\n")
        if start_time is not None and end_time is not None:
            handle.write(f"start_time_index: {start_time}\n")
            handle.write(f"end_time_index: {end_time}\n")
        handle.write(f"submesofluxnet_annual_r2: {fluxnet_r2:.10g}\n")
        handle.write(f"fk_annual_r2: {fk_r2:.10g}\n")
        handle.write(f"submesofluxnet_nse_mean: {float(np.nanmean(fluxnet_error)):.10g}\n")
        handle.write(f"fk_nse_mean: {float(np.nanmean(fk_error)):.10g}\n")


def main() -> None:
    args = parse_args()
    flux_sample, flux_prediction, target = load_fluxnet(args.fluxnet_predictions)
    fk_sample, fk_prediction, fk_time_index = load_fk(args.fk_predictions)
    sample, flux_prediction, target, fk_prediction, time_index = align_to_fluxnet(
        flux_sample, flux_prediction, target, fk_sample, fk_prediction, fk_time_index
    )

    start = args.block_index * args.window
    stop = start + args.window
    truth_mean = annual_block_mean(target, args.window, args.block_index)
    fluxnet_mean = annual_block_mean(flux_prediction, args.window, args.block_index)
    fk_mean = annual_block_mean(fk_prediction, args.window, args.block_index)
    fluxnet_error = normalized_squared_error(fluxnet_mean, truth_mean)
    fk_error = normalized_squared_error(fk_mean, truth_mean)
    fluxnet_r2 = r2_score(fluxnet_mean, truth_mean)
    fk_r2 = r2_score(fk_mean, truth_mean)

    draw_figure(
        truth=truth_mean,
        fluxnet=fluxnet_mean,
        fk=fk_mean,
        fluxnet_error=fluxnet_error,
        fk_error=fk_error,
        field_percentile=args.field_percentile,
        error_percentile=args.error_percentile,
        output_prefix=args.output_prefix,
    )

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data_output,
        window_model_days=np.asarray([args.window], dtype=np.int64),
        block_index=np.asarray([args.block_index], dtype=np.int64),
        sample_index=sample[start:stop],
        time_index=time_index[start:stop] if time_index is not None else np.asarray([], dtype=np.int64),
        truth=truth_mean.astype(np.float32),
        submesofluxnet_prediction=fluxnet_mean.astype(np.float32),
        fk_prediction=fk_mean.astype(np.float32),
        submesofluxnet_normalized_squared_error=fluxnet_error.astype(np.float32),
        fk_normalized_squared_error=fk_error.astype(np.float32),
    )
    write_summary(
        args.summary_output,
        window=args.window,
        block_index=args.block_index,
        start_sample=int(sample[start]),
        end_sample=int(sample[stop - 1]),
        start_time=int(time_index[start]) if time_index is not None else None,
        end_time=int(time_index[stop - 1]) if time_index is not None else None,
        fluxnet_r2=fluxnet_r2,
        fk_r2=fk_r2,
        fluxnet_error=fluxnet_error,
        fk_error=fk_error,
    )

    for ext in (".svg", ".pdf", ".png", ".tiff"):
        print(f"Saved {args.output_prefix}{ext}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.data_output}", flush=True)
    print(f"Annual block {args.block_index}: SubmesoFluxNet R2={fluxnet_r2:.6f}, FK R2={fk_r2:.6f}", flush=True)


if __name__ == "__main__":
    main()
