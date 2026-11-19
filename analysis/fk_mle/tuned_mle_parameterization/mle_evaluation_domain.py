#!/usr/bin/env python3
"""Evaluate the tuned MLE parameterization on the SubmesoFluxNet test split."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import xarray as xr

from mle_tuning.mle_parameterization import compute_mle_submeso_heat_flux_one_time
from submeso_data_driven.utils import build_coarse_grid, read_mitgcm

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "results/analysis_outputs/mle_parameterization/tuned_mle_parameterization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-e", type=float, default=2.25895966)
    parser.add_argument("--lf-min", type=float, default=1374.0)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "results/model/test_predictions_denorm.npz",
    )
    parser.add_argument(
        "--time-source",
        type=Path,
        default=PROJECT_DIR / "data/processed/upper/time_index.npy",
    )
    parser.add_argument(
        "--grid-run-dir",
        type=Path,
        default=Path("/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"),
    )
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument(
        "--nc-dir",
        type=Path,
        default=PROJECT_DIR / "data/interim/temporal_spatial_avg",
    )
    parser.add_argument("--engine", default="h5netcdf", choices=("h5netcdf", "netcdf4", "scipy"))
    parser.add_argument("--crop", type=int, nargs=4, default=(92, 112, 12, 52))
    parser.add_argument("--submeso-lmax", type=float, default=2751.0)
    parser.add_argument("--submeso-inv-tau", type=float, default=2.0e-6)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "mle_prediction_physical.npz")
    parser.add_argument("--r2-csv", type=Path, default=OUT_DIR / "mle_r2_series.csv")
    return parser.parse_args()


def load_test_times(predictions_path: Path, time_source_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(predictions_path)
    time_source = np.load(time_source_path)
    sample_index = pred["sample_index"].astype(np.int64)
    if isinstance(time_source, np.lib.npyio.NpzFile):
        key = "time_index" if "time_index" in time_source.files else "time"
        all_time = time_source[key].astype(np.int64)
    else:
        all_time = np.asarray(time_source, dtype=np.int64)
    time_index = all_time[sample_index]
    return sample_index, time_index


def build_grid(grid_run_dir: Path, box_j: int, box_i: int) -> xr.Dataset:
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_run_dir)
    return build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=box_j, box_i=box_i)


def load_mle_prediction(path: Path):
    if not path.exists():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), None
    with np.load(path) as data:
        return (
            data["sample_index"].astype(np.int64),
            data["time_index"].astype(np.int64),
            data["prediction"].astype(np.float32),
        )


def save_mle_prediction(
    path: Path,
    sample_index: np.ndarray,
    time_index: np.ndarray,
    prediction: np.ndarray,
    args: argparse.Namespace,
) -> None:
    order = np.argsort(sample_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            sample_index=sample_index[order],
            time_index=time_index[order],
            prediction=prediction[order],
            units=np.asarray(["W m^-2"]),
            c_e=np.asarray([args.c_e]),
            lf_min=np.asarray([args.lf_min]),
        )
    os.replace(temporary, path)


def calculate_mle(args: argparse.Namespace, sample_index: np.ndarray, target_time: np.ndarray):
    if args.output.exists():
        with np.load(args.output) as saved:
            if not np.isclose(float(saved["c_e"][0]), args.c_e) or not np.isclose(float(saved["lf_min"][0]), args.lf_min):
                raise ValueError("Existing MLE output uses different C_e or LfMin values.")

    ds_grid = build_grid(args.grid_run_dir, args.box_j, args.box_i)
    ds_kpphbl = xr.open_dataset(args.nc_dir / "kpphbl_12hmean_coarse.nc", engine=args.engine, decode_times=False)
    ds_t = xr.open_dataset(args.nc_dir / "T_12hmean_coarse.nc", engine=args.engine, decode_times=False)
    ds_s = xr.open_dataset(args.nc_dir / "S_12hmean_coarse.nc", engine=args.engine, decode_times=False)
    ds_eta = xr.open_dataset(args.nc_dir / "Eta_12hmean_coarse.nc", engine=args.engine, decode_times=False)

    kpphbl_da = ds_kpphbl["kpphbl"]
    t_da = ds_t["T"]
    s_da = ds_s["S"]
    eta_da = ds_eta["Eta"]

    available = set(np.asarray(kpphbl_da["time"].values, dtype=np.int64))
    available &= set(np.asarray(t_da["time"].values, dtype=np.int64))
    available &= set(np.asarray(s_da["time"].values, dtype=np.int64))
    available &= set(np.asarray(eta_da["time"].values, dtype=np.int64))
    missing_from_nc = [int(t) for t in target_time if int(t) not in available]
    if missing_from_nc:
        raise RuntimeError(
            f"MLE input files are missing {len(missing_from_nc)} test times. "
            f"First missing: {missing_from_nc[:5]}"
        )

    saved_sample, saved_time, saved_prediction = load_mle_prediction(args.output)
    saved_set = set(map(int, saved_time))
    todo = [
        (int(s), int(t))
        for s, t in zip(sample_index, target_time)
        if int(t) in available and int(t) not in saved_set
    ]
    if args.max_snapshots is not None:
        todo = todo[: args.max_snapshots]

    print(f"Existing MLE predictions: {len(saved_set)}", flush=True)
    print(f"Need to compute in this run: {len(todo)}", flush=True)
    j0, j1, i0, i1 = args.crop
    for n, (sample, time) in enumerate(todo, start=1):
        print(f"[{n}/{len(todo)}] MLE time_index={time}", flush=True)
        full_map = compute_mle_submeso_heat_flux_one_time(
            time_index=time,
            ds_grid=ds_grid,
            kpphbl_da=kpphbl_da,
            t_da=t_da,
            s_da=s_da,
            eta_da=eta_da,
            c_e=args.c_e,
            submeso_lmax=args.submeso_lmax,
            submeso_inv_tau=args.submeso_inv_tau,
            lf_min=args.lf_min,
            f0=args.f0,
            beta=args.beta,
        )
        cropped = np.asarray(full_map[j0:j1, i0:i1], dtype=np.float32)[None, None]
        saved_sample = np.append(saved_sample, sample)
        saved_time = np.append(saved_time, time)
        saved_prediction = cropped if saved_prediction is None else np.concatenate([saved_prediction, cropped])
        save_mle_prediction(args.output, saved_sample, saved_time, saved_prediction, args)

    return load_mle_prediction(args.output)


def load_prediction_and_truth(output_path: Path, predictions_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_mle = np.load(output_path)
    pred_nn = np.load(predictions_path)
    sample_index = pred_mle["sample_index"].astype(np.int64)
    time_index = pred_mle["time_index"].astype(np.int64)
    mle = pred_mle["prediction"].astype(np.float64)
    all_sample = pred_nn["sample_index"].astype(np.int64)
    all_target = pred_nn["target"].astype(np.float64)
    lookup = {int(s): n for n, s in enumerate(all_sample)}
    target = np.stack([all_target[lookup[int(s)]] for s in sample_index], axis=0)
    if mle.shape != target.shape:
        raise ValueError(f"MLE prediction and target shapes differ: {mle.shape} != {target.shape}")
    return sample_index, time_index, mle, target


def per_snapshot_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    r2 = np.full(pred.shape[0], np.nan, dtype=np.float64)
    mse = np.full(pred.shape[0], np.nan, dtype=np.float64)
    mae = np.full(pred.shape[0], np.nan, dtype=np.float64)
    for n in range(pred.shape[0]):
        valid = np.isfinite(flat_pred[n]) & np.isfinite(flat_target[n])
        if not np.any(valid):
            continue
        p = flat_pred[n, valid]
        y = flat_target[n, valid]
        diff = p - y
        sse = float(np.sum(diff**2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst > 0.0:
            r2[n] = 1.0 - sse / sst
        mse[n] = float(np.mean(diff**2))
        mae[n] = float(np.mean(np.abs(diff)))
    return r2, mse, mae


def write_r2_csv(path: Path, sample_index: np.ndarray, time_index: np.ndarray, r2: np.ndarray, mse: np.ndarray, mae: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "time_index", "r2", "mse", "mae"])
        writer.writeheader()
        for n, row in enumerate(zip(sample_index, time_index, r2, mse, mae)):
            sample, time, r2_n, mse_n, mae_n = row
            writer.writerow(
                {
                    "test_position": n,
                    "sample_index": int(sample),
                    "time_index": int(time),
                    "r2": float(r2_n),
                    "mse": float(mse_n),
                    "mae": float(mae_n),
                }
            )


def main() -> None:
    args = parse_args()
    test_samples, test_times = load_test_times(args.predictions, args.time_source)
    print(f"Test snapshots: {test_samples.size}")
    print(f"Using C_e={args.c_e} and LfMin={args.lf_min} m")

    sample_index, time_index, prediction = calculate_mle(args, test_samples, test_times)
    if prediction is None:
        raise RuntimeError("No MLE predictions were calculated.")

    sample_index, time_index, prediction, target = load_prediction_and_truth(
        args.output,
        args.predictions,
    )
    r2, mse, mae = per_snapshot_metrics(prediction, target)
    write_r2_csv(args.r2_csv, sample_index, time_index, r2, mse, mae)

    print(f"Completed snapshots: {sample_index.size}/{test_samples.size}")
    print(f"Mean snapshot R2: {np.nanmean(r2):.6f}")
    print(f"Saved: {args.output}")
    print(f"Saved: {args.r2_csv}")


if __name__ == "__main__":
    main()
