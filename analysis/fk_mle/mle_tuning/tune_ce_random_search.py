#!/usr/bin/env python3
"""Tune MLE C_e on the SubmesoFluxNet training samples.

The MLE heat flux is linear in C_e when LfMin, invTau, and Lmax are fixed. This
script therefore computes a reusable C_e=1 basis on the training samples
specified in config.yaml, then random-searches C_e by
scaling that basis and maximizing mean per-snapshot R2.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from submeso_data_driven.utils import build_coarse_grid, read_mitgcm
import xarray as xr
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parent
CASE_DIR = PROJECT_DIR / "results/analysis_outputs/fk_mle/mle_tuning"
MLE_INPUT_DIR = PROJECT_DIR / "data/interim/fk_mle"

from mle_parameterization import compute_mle_submeso_heat_flux_one_time
from submeso_data_driven.utils import make_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ANALYSIS_DIR / "config.yaml")
    parser.add_argument("--fk-inputs", type=Path, default=MLE_INPUT_DIR / "fk_input_trimmed.npz")
    parser.add_argument("--truth", type=Path, default=PROJECT_DIR / "data/processed/upper/y.npy")
    parser.add_argument("--time-source", type=Path, default=PROJECT_DIR / "data/processed/upper/time_index.npy")
    parser.add_argument(
        "--grid-run-dir",
        type=Path,
        default=Path("/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"),
    )
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument("--crop", type=int, nargs=4, default=(92, 112, 12, 52), metavar=("J0", "J1", "I0", "I1"))
    parser.add_argument("--lf-min", type=float, default=1000.0)
    parser.add_argument("--submeso-lmax", type=float, default=2751.0)
    parser.add_argument("--submeso-inv-tau", type=float, default=2.0e-6)
    parser.add_argument("--ce-min", type=float, default=0.1)
    parser.add_argument("--ce-max", type=float, default=4.0)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-snapshots",
        type=int,
        default=None,
        help="Optional quick-test limit after alignment. Default uses all overlapping train snapshots.",
    )
    parser.add_argument(
        "--basis-batch-size",
        type=int,
        default=None,
        help="Compute only the next N missing C_e=1 basis snapshots in this run, then continue/exit depending on mode.",
    )
    parser.add_argument(
        "--build-basis-only",
        action="store_true",
        help="Only extend the C_e=1 basis file, then exit before random search.",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Skip basis building and tune C_e using the snapshots already present in the basis file.",
    )
    parser.add_argument(
        "--basis-output",
        type=Path,
        default=CASE_DIR / "ce_basis.npz",
        help="Reusable C_e=1 MLE basis. Existing snapshots are reused.",
    )
    parser.add_argument("--history-csv", type=Path, default=CASE_DIR / "ce_tuning.csv")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def valid_center_indices(n_samples: int, temporal_offsets: list[int]) -> np.ndarray:
    min_offset = min(temporal_offsets)
    max_offset = max(temporal_offsets)
    start = max(0, -min_offset)
    stop = min(n_samples, n_samples - max_offset)
    if stop <= start:
        raise ValueError(f"No valid centers for n_samples={n_samples}, temporal_offsets={temporal_offsets}")
    return np.arange(start, stop, dtype=np.int64)


def split_indices(n_samples: int, split: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    shuffle = bool(split.get("shuffle", False))
    shuffle_train_val = bool(split.get("shuffle_train_val", False))
    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    indices = np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if shuffle:
        indices = rng.permutation(indices)
        return indices[:n_train], indices[n_train : n_train + n_val], indices[n_train + n_val :]
    train_val_idx = indices[: n_train + n_val]
    test_idx = indices[n_train + n_val :]
    if shuffle_train_val:
        train_val_idx = rng.permutation(train_val_idx)
    return train_val_idx[:n_train], train_val_idx[n_train : n_train + n_val], test_idx


def submesofluxnet_train_indices(config: dict, project_dir: Path) -> np.ndarray:
    data_cfg = config["data"]
    y_path = Path(data_cfg["y_path"])
    if not y_path.is_absolute():
        y_path = project_dir / y_path
    y = np.load(y_path, mmap_mode="r")
    offsets = [int(v) for v in data_cfg.get("temporal_offsets", [0])]
    centers = valid_center_indices(y.shape[0], offsets)
    train_pos, _, _ = split_indices(centers.shape[0], data_cfg["split"], int(config.get("seed", 42)))
    return centers[train_pos]


def load_time_index(path: Path) -> np.ndarray:
    z = np.load(path)
    if isinstance(z, np.lib.npyio.NpzFile):
        time_key = "time_index" if "time_index" in z.files else "time"
        return z[time_key].astype(np.int64)
    return np.asarray(z, dtype=np.int64)


def load_truth(path: Path, time_source: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
        data = np.asarray(data[:, 0] if data.ndim == 4 and data.shape[1] == 1 else data, dtype=np.float32)
        return load_time_index(time_source), data
    z = np.load(path)
    time_key = "time_index" if "time_index" in z.files else "time"
    return z[time_key].astype(np.int64), z["data"].astype(np.float32)


def load_fk_input_times(path: Path) -> np.ndarray:
    z = np.load(path)
    time_key = "time" if "time" in z.files else "time_index"
    return z[time_key].astype(np.int64)


def align_train_samples(
    train_sample_index: np.ndarray,
    truth_time: np.ndarray,
    fk_input_time: np.ndarray,
    max_snapshots: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    train_time = truth_time[train_sample_index]
    fk_times = set(map(int, fk_input_time))
    keep = np.asarray([n for n, t in enumerate(train_time) if int(t) in fk_times], dtype=np.int64)
    if keep.size == 0:
        raise ValueError("No SubmesoFluxNet train samples overlap the MLE input cache.")
    if max_snapshots is not None:
        keep = keep[:max_snapshots]
    return train_sample_index[keep], train_time[keep]


def build_grid(grid_run_dir: Path, box_j: int, box_i: int) -> tuple[xr.Dataset, object]:
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_run_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=box_j, box_i=box_i)
    grid = make_grid.make_grid_periodic_x(ds_coarse)
    return ds_coarse, grid


def make_dataarrays(fk_inputs: np.lib.npyio.NpzFile, ds_coarse: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, np.ndarray]:
    time_key = "time" if "time" in fk_inputs.files else "time_index"
    times = fk_inputs[time_key].astype(np.int64)
    kpphbl_da = xr.DataArray(
        fk_inputs["kpphbl"].astype(np.float32),
        dims=("time", "j", "i"),
        coords={"time": times, "j": ds_coarse["YC"].coords["j"], "i": ds_coarse["YC"].coords["i"]},
        name="kpphbl",
    )
    t_da = xr.DataArray(
        fk_inputs["T"].astype(np.float32),
        dims=("time", "k", "j", "i"),
        coords={
            "time": times,
            "k": ds_coarse["RC"].coords["k"],
            "j": ds_coarse["YC"].coords["j"],
            "i": ds_coarse["YC"].coords["i"],
        },
        name="T",
    )
    s_da = xr.DataArray(
        fk_inputs["S"].astype(np.float32),
        dims=("time", "k", "j", "i"),
        coords={
            "time": times,
            "k": ds_coarse["RC"].coords["k"],
            "j": ds_coarse["YC"].coords["j"],
            "i": ds_coarse["YC"].coords["i"],
        },
        name="S",
    )
    eta_da = xr.DataArray(
        fk_inputs["Eta"].astype(np.float32),
        dims=("time", "j", "i"),
        coords={"time": times, "j": ds_coarse["YC"].coords["j"], "i": ds_coarse["YC"].coords["i"]},
        name="Eta",
    )
    return kpphbl_da, t_da, s_da, eta_da, times


def load_basis(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return z["sample_index"].astype(np.int64), z["time_index"].astype(np.int64), z["basis_flux"].astype(np.float32)


def save_basis(path: Path, sample_index: np.ndarray, time_index: np.ndarray, basis_flux: np.ndarray) -> None:
    order = np.argsort(time_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_index=sample_index[order].astype(np.int64),
            time_index=time_index[order].astype(np.int64),
            basis_flux=basis_flux[order].astype(np.float32),
            base_ce=np.asarray([1.0], dtype=np.float64),
        )
    os.replace(tmp_path, path)


def append_basis(path: Path, sample: int, time: int, flux: np.ndarray) -> None:
    current = load_basis(path)
    new_sample = np.asarray([sample], dtype=np.int64)
    new_time = np.asarray([time], dtype=np.int64)
    new_flux = np.asarray([flux], dtype=np.float32)
    if current is None:
        save_basis(path, new_sample, new_time, new_flux)
        return
    old_sample, old_time, old_flux = current
    sample_cat = np.concatenate([old_sample, new_sample])
    time_cat = np.concatenate([old_time, new_time])
    flux_cat = np.concatenate([old_flux, new_flux], axis=0)
    reverse = np.arange(time_cat.size - 1, -1, -1)
    _, first = np.unique(time_cat[reverse], return_index=True)
    keep = np.sort(reverse[first])
    save_basis(path, sample_cat[keep], time_cat[keep], flux_cat[keep])


def build_ce1_basis(args: argparse.Namespace, sample_index: np.ndarray, time_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = load_basis(args.basis_output)
    existing_times: set[int] = set()
    if current is not None:
        existing_sample, existing_time, existing_flux = current
        existing_times = set(map(int, existing_time))
        print(f"Existing C_e=1 basis snapshots: {existing_time.size}", flush=True)

    missing_all = [(int(s), int(t)) for s, t in zip(sample_index, time_index) if int(t) not in existing_times]
    missing = missing_all
    if args.basis_batch_size is not None:
        missing = missing[: args.basis_batch_size]
    print(f"Missing C_e=1 basis snapshots total: {len(missing_all)}", flush=True)
    print(f"Will compute in this run: {len(missing)}", flush=True)
    if missing:
        ds_coarse, _ = build_grid(args.grid_run_dir, args.box_j, args.box_i)
        fk_inputs = np.load(args.fk_inputs)
        kpphbl_da, t_da, s_da, eta_da, _ = make_dataarrays(fk_inputs, ds_coarse)
        j0, j1, i0, i1 = args.crop

        for n, (sample, time) in enumerate(missing, start=1):
            print(f"[basis {n}/{len(missing)}] sample={sample} time={time}", flush=True)
            flux_full = compute_fk_submeso_heat_flux_one_time(
                time_index=time,
                ds_grid=ds_coarse,
                kpphbl_da=kpphbl_da,
                t_da=t_da,
                s_da=s_da,
                eta_da=eta_da,
                c_e=1.0,
                submeso_lmax=args.submeso_lmax,
                submeso_inv_tau=args.submeso_inv_tau,
                lf_min=args.lf_min,
            )
            append_basis(args.basis_output, sample, time, flux_full[j0:j1, i0:i1])

    final = load_basis(args.basis_output)
    if final is None:
        raise RuntimeError("Basis file was not created.")
    basis_sample, basis_time, basis_flux = final
    wanted = {int(t): int(s) for s, t in zip(sample_index, time_index)}
    keep = np.asarray([n for n, t in enumerate(basis_time) if int(t) in wanted], dtype=np.int64)
    return basis_sample[keep], basis_time[keep], basis_flux[keep]


def per_snapshot_r2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_truth = truth.reshape(truth.shape[0], -1)
    out = np.full(pred.shape[0], np.nan, dtype=np.float64)
    for n in range(pred.shape[0]):
        valid = np.isfinite(flat_pred[n]) & np.isfinite(flat_truth[n])
        if not np.any(valid):
            continue
        y = flat_truth[n, valid]
        p = flat_pred[n, valid]
        sse = float(np.sum((p - y) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst > 0.0:
            out[n] = 1.0 - sse / sst
    return out


def random_search_ce(args: argparse.Namespace, basis_flux: np.ndarray, truth: np.ndarray) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(args.seed)
    ce_values = rng.uniform(args.ce_min, args.ce_max, size=args.n_trials)
    # Include endpoints and commonly used values.
    ce_values = np.concatenate([np.asarray([args.ce_min, args.ce_max, 0.1477, 0.15, 1.0]), ce_values])

    rows: list[dict[str, float | int]] = []
    for trial, ce in enumerate(ce_values):
        pred = float(ce) * basis_flux
        r2_series = per_snapshot_r2(pred, truth)
        row = {
            "trial": int(trial),
            "C_e": float(ce),
            "mean_per_snapshot_r2": float(np.nanmean(r2_series)),
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["mean_per_snapshot_r2"]), reverse=True)
    return rows


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "trial",
                "C_e",
                "mean_per_snapshot_r2",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)


def report_basis_status(path: Path, target_time: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    basis = load_basis(path)
    if basis is None:
        print(f"Basis status: no readable basis file at {path}", flush=True)
        return None
    basis_sample, basis_time, basis_flux = basis
    target_set = set(map(int, target_time))
    covered = sum(1 for t in basis_time if int(t) in target_set)
    missing = len(target_set) - covered
    print(f"Basis status: {covered}/{len(target_set)} aligned snapshots available; missing {missing}", flush=True)
    if basis_time.size:
        print(f"Basis time range: {int(basis_time.min())} ... {int(basis_time.max())}", flush=True)
    return basis_sample, basis_time, basis_flux


def main() -> None:
    args = parse_args()
    if args.build_basis_only and args.tune_only:
        raise ValueError("Use only one of --build-basis-only or --tune-only, not both.")

    config = load_config(args.config)
    train_sample_all = submesofluxnet_train_indices(config, PROJECT_DIR)
    truth_time, truth_data = load_truth(args.truth, args.time_source)
    fk_input_time = load_fk_input_times(args.fk_inputs)
    sample_index, time_index = align_train_samples(train_sample_all, truth_time, fk_input_time, args.max_train_snapshots)

    print(f"SubmesoFluxNet train samples: {train_sample_all.size}", flush=True)
    print(f"Overlapping MLE-input train samples: {sample_index.size}", flush=True)
    print(f"sample_index range: {int(sample_index.min())} ... {int(sample_index.max())}", flush=True)
    print(f"time_index range: {int(time_index.min())} ... {int(time_index.max())}", flush=True)

    if args.tune_only:
        current = report_basis_status(args.basis_output, time_index)
        if current is None:
            raise RuntimeError("Cannot use --tune-only because no readable basis file exists yet.")
        basis_sample_all, basis_time_all, basis_flux_all = current
        wanted = set(map(int, time_index))
        keep = np.asarray([n for n, t in enumerate(basis_time_all) if int(t) in wanted], dtype=np.int64)
        if keep.size == 0:
            raise RuntimeError("Basis file has no snapshots overlapping the requested training set.")
        basis_sample = basis_sample_all[keep]
        basis_time = basis_time_all[keep]
        basis_flux = basis_flux_all[keep]
    else:
        basis_sample, basis_time, basis_flux = build_ce1_basis(args, sample_index, time_index)
        report_basis_status(args.basis_output, time_index)

    if args.build_basis_only:
        print("Build-basis-only mode complete. Rerun the same command to add more snapshots, or use --tune-only to tune with the current basis.", flush=True)
        return

    truth_lookup = {int(t): n for n, t in enumerate(truth_time)}
    truth = np.stack([truth_data[truth_lookup[int(t)]] for t in basis_time], axis=0)

    rows = random_search_ce(args, basis_flux.astype(np.float64), truth.astype(np.float64))
    write_history(args.history_csv, rows)

    best = rows[0]
    print("Best C_e tuning result:", flush=True)
    print(f"  C_e = {float(best['C_e']):.8f}", flush=True)
    print(f"  mean per-snapshot R2 = {float(best['mean_per_snapshot_r2']):.8f}", flush=True)
    print(f"Saved {args.history_csv}", flush=True)


if __name__ == "__main__":
    main()
