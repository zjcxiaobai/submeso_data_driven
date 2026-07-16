#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_best_fluxnet_snapshot_comparison import draw_joint_hist_figure, draw_main_figure, load_fk, load_fluxnet
from plot_best_fluxnet_snapshot_comparison import normalized_squared_error, r2_score, spatial_corr, temporal_corr_map


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FK_DISPLAY_LABEL = "MLE"


FLUXNET_PREDICTIONS = PROJECT_DIR / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_predictions_denorm.npz"
FK_PREDICTIONS = PROJECT_DIR / "analysis/fk_mle/upper_40x20/fk_shuffle_all_test_Ce2p13062237_prediction_physical.npz"


def align_common_subset(
    flux_sample: np.ndarray,
    flux_prediction: np.ndarray,
    target: np.ndarray,
    flux_r2: np.ndarray,
    fk_sample: np.ndarray,
    fk_prediction: np.ndarray,
    fk_time_index: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    flux_lookup = {int(sample): pos for pos, sample in enumerate(flux_sample)}
    fk_lookup = {int(sample): pos for pos, sample in enumerate(fk_sample)}
    common = np.asarray(sorted(set(flux_lookup).intersection(fk_lookup)), dtype=np.int64)
    if common.size == 0:
        raise ValueError("No common sample_index values between shuffle-all SubmesoFluxNet and FK predictions.")

    flux_pos = np.asarray([flux_lookup[int(sample)] for sample in common], dtype=np.int64)
    fk_pos = np.asarray([fk_lookup[int(sample)] for sample in common], dtype=np.int64)
    aligned_time = fk_time_index[fk_pos] if fk_time_index is not None else None
    return (
        common,
        flux_prediction[flux_pos],
        target[flux_pos],
        flux_r2[flux_pos],
        fk_prediction[fk_pos],
        aligned_time,
    )


def write_summary(
    path: Path,
    *,
    selection: str,
    n_common: int,
    sample_index: int,
    common_position: int,
    time_index: int | None,
    fluxnet_r2: float,
    fk_r2: float,
    fluxnet_spatial_corr: float,
    fk_spatial_corr: float,
    fluxnet_error: np.ndarray,
    fk_error: np.ndarray,
    fluxnet_temporal_corr: np.ndarray,
    fk_temporal_corr: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"Shuffle-all SubmesoFluxNet snapshot comparison against {FK_DISPLAY_LABEL}\n")
        handle.write(f"comparison_scope: common sample_index subset where {FK_DISPLAY_LABEL} prediction is available\n")
        handle.write(f"selection: {selection}\n")
        handle.write(f"n_common_snapshots: {n_common}\n")
        handle.write(f"sample_index: {sample_index}\n")
        handle.write(f"common_subset_position: {common_position}\n")
        if time_index is not None:
            handle.write(f"time_index: {time_index}\n")
        handle.write(f"submesofluxnet_snapshot_r2: {fluxnet_r2:.10g}\n")
        handle.write(f"fk_snapshot_r2: {fk_r2:.10g}\n")
        handle.write(f"submesofluxnet_snapshot_spatial_corr: {fluxnet_spatial_corr:.10g}\n")
        handle.write(f"fk_snapshot_spatial_corr: {fk_spatial_corr:.10g}\n")
        handle.write(f"submesofluxnet_nse_mean: {float(np.nanmean(fluxnet_error)):.10g}\n")
        handle.write(f"fk_nse_mean: {float(np.nanmean(fk_error)):.10g}\n")
        handle.write(f"submesofluxnet_temporal_corr_mean: {float(np.nanmean(fluxnet_temporal_corr)):.10g}\n")
        handle.write(f"fk_temporal_corr_mean: {float(np.nanmean(fk_temporal_corr)):.10g}\n")


def save_case(
    *,
    selection: str,
    pos: int,
    samples: np.ndarray,
    flux_prediction: np.ndarray,
    target: np.ndarray,
    flux_r2: np.ndarray,
    fk_prediction: np.ndarray,
    fk_time_index: np.ndarray | None,
    fluxnet_temporal_corr: np.ndarray,
    fk_temporal_corr: np.ndarray,
    prefix: str,
) -> None:
    sample_index = int(samples[pos])
    time_index = int(fk_time_index[pos]) if fk_time_index is not None else None
    truth_best = target[pos]
    fluxnet_best = flux_prediction[pos]
    fk_best = fk_prediction[pos]
    fluxnet_error = normalized_squared_error(fluxnet_best, truth_best)
    fk_error = normalized_squared_error(fk_best, truth_best)
    fk_r2 = r2_score(fk_best, truth_best)
    fluxnet_spcorr = spatial_corr(fluxnet_best, truth_best)
    fk_spcorr = spatial_corr(fk_best, truth_best)

    output_prefix = HERE / f"{prefix}_fk_comparison"
    joint_prefix = HERE / f"{prefix}_joint_histograms"
    data_output = HERE / f"{prefix}_fk_comparison_data.npz"
    summary_output = HERE / f"{prefix}_fk_comparison_summary.txt"

    draw_main_figure(
        sample_index=sample_index,
        fluxnet_r2=float(flux_r2[pos]),
        fk_r2=fk_r2,
        truth_best=truth_best,
        fluxnet_best=fluxnet_best,
        fk_best=fk_best,
        fluxnet_error=fluxnet_error,
        fk_error=fk_error,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        fk_temporal_corr=fk_temporal_corr,
        field_percentile=99.0,
        error_percentile=99.0,
        output_prefix=output_prefix,
        fk_label=FK_DISPLAY_LABEL,
    )
    draw_joint_hist_figure(
        truth_best=truth_best,
        fluxnet_best=fluxnet_best,
        fk_best=fk_best,
        hist_bins=80,
        output_prefix=joint_prefix,
        fk_label=FK_DISPLAY_LABEL,
    )
    np.savez_compressed(
        data_output,
        sample_index=np.asarray([sample_index], dtype=np.int64),
        common_subset_position=np.asarray([pos], dtype=np.int64),
        time_index=np.asarray([time_index], dtype=np.int64) if time_index is not None else np.asarray([], dtype=np.int64),
        truth=truth_best.astype(np.float32),
        submesofluxnet_prediction=fluxnet_best.astype(np.float32),
        fk_prediction=fk_best.astype(np.float32),
        submesofluxnet_normalized_squared_error=fluxnet_error.astype(np.float32),
        fk_normalized_squared_error=fk_error.astype(np.float32),
        submesofluxnet_temporal_corr=fluxnet_temporal_corr.astype(np.float32),
        fk_temporal_corr=fk_temporal_corr.astype(np.float32),
    )
    write_summary(
        summary_output,
        selection=selection,
        n_common=int(samples.size),
        sample_index=sample_index,
        common_position=pos,
        time_index=time_index,
        fluxnet_r2=float(flux_r2[pos]),
        fk_r2=fk_r2,
        fluxnet_spatial_corr=fluxnet_spcorr,
        fk_spatial_corr=fk_spcorr,
        fluxnet_error=fluxnet_error,
        fk_error=fk_error,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        fk_temporal_corr=fk_temporal_corr,
    )

    print(f"{selection}: sample={sample_index}, SubmesoFluxNet R2={float(flux_r2[pos]):.6f}, {FK_DISPLAY_LABEL} R2={fk_r2:.6f}", flush=True)
    print(f"Saved {output_prefix}.png", flush=True)
    print(f"Saved {joint_prefix}.png", flush=True)
    print(f"Saved {summary_output}", flush=True)
    print(f"Saved {data_output}", flush=True)


def main() -> None:
    flux_sample, flux_prediction, target, flux_r2 = load_fluxnet(FLUXNET_PREDICTIONS)
    fk_sample, fk_prediction, fk_time_index = load_fk(FK_PREDICTIONS)
    samples, flux_prediction, target, flux_r2, fk_prediction, fk_time_index = align_common_subset(
        flux_sample,
        flux_prediction,
        target,
        flux_r2,
        fk_sample,
        fk_prediction,
        fk_time_index,
    )

    fk_r2_series = np.asarray([r2_score(fk_prediction[n], target[n]) for n in range(samples.size)], dtype=np.float64)
    best_fluxnet_pos = int(np.nanargmax(flux_r2))
    best_fk_pos = int(np.nanargmax(fk_r2_series))
    fluxnet_temporal_corr = temporal_corr_map(flux_prediction, target)
    fk_temporal_corr = temporal_corr_map(fk_prediction, target)

    save_case(
        selection=f"maximum SubmesoFluxNet R2 on common {FK_DISPLAY_LABEL}-available shuffle-all test subset",
        pos=best_fluxnet_pos,
        samples=samples,
        flux_prediction=flux_prediction,
        target=target,
        flux_r2=flux_r2,
        fk_prediction=fk_prediction,
        fk_time_index=fk_time_index,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        fk_temporal_corr=fk_temporal_corr,
        prefix="shuffle_all_best_fluxnet_snapshot",
    )
    save_case(
        selection=f"maximum {FK_DISPLAY_LABEL} R2 on common {FK_DISPLAY_LABEL}-available shuffle-all test subset",
        pos=best_fk_pos,
        samples=samples,
        flux_prediction=flux_prediction,
        target=target,
        flux_r2=flux_r2,
        fk_prediction=fk_prediction,
        fk_time_index=fk_time_index,
        fluxnet_temporal_corr=fluxnet_temporal_corr,
        fk_temporal_corr=fk_temporal_corr,
        prefix="shuffle_all_best_fk_snapshot",
    )


if __name__ == "__main__":
    main()
