#!/usr/bin/env python3
"""Compute FK predictions on the lower_40x20 evaluation subdomain."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from submeso_data_driven.utils import read_mitgcm
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_DIR.parent
FK_DIR = PROJECT_DIR / "analysis/fk_mle/upper_40x20"
HERE = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FK_DIR) not in sys.path:
    sys.path.insert(0, str(FK_DIR))

from fk_physics import compute_fk_submeso_heat_flux_one_time  # noqa: E402
from submeso_data_driven.utils import build_coarse_grid  # noqa: E402


FEATURES = ["U", "V", "W", "zeta", "strain", "T", "rho", "kpphbl"]
TEMPORAL_OFFSETS = [-8, -7, -6, -5, -4, -3, -2, -1, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-e", type=float, default=2.25895966)
    parser.add_argument("--lf-min", type=float, default=1374.0)
    parser.add_argument("--submeso-lmax", type=float, default=2751.0)
    parser.add_argument("--submeso-inv-tau", type=float, default=2.0e-6)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--eval-data-dir", type=Path, default=HERE / "evaluation_subdomain")
    parser.add_argument("--time-source", type=Path, default=HERE / "evaluation_subdomain/time_index.npy")
    parser.add_argument(
        "--grid-run-dir",
        type=Path,
        default=Path("/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"),
    )
    parser.add_argument(
        "--nc-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("/scratch/jczhang/data_5km/save_coarse/5500080_13827312_nc"),
            Path("/scratch/jczhang/data_5km/save_coarse/out_nc"),
            Path("/scratch/jczhang/data_5km/save_coarse/13901616_nc"),
        ],
        help="FK-input NetCDF folders searched in order for each time_index.",
    )
    parser.add_argument("--engine", default="h5netcdf", choices=("h5netcdf", "netcdf4", "scipy"))
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=(72, 92, 12, 52),
        metavar=("J0", "J1", "I0", "I1"),
        help="Lower 40x20 crop on the 120x60 FK grid. Matches 56x56 y=10:30, x=10:50.",
    )
    parser.add_argument("--max-snapshots", type=int, default=None, help="Compute only the next N missing snapshots.")
    parser.add_argument("--assemble-only", action="store_true", help="Only assemble output from existing progress cache.")
    parser.add_argument("--allow-partial-output", action="store_true", help="Save overlapping computed snapshots even if incomplete.")
    parser.add_argument("--progress", type=Path, default=HERE / "evaluation_subdomain_fk_progress_fullgrid.npz")
    parser.add_argument("--output", type=Path, default=HERE / "evaluation_subdomain_fk_prediction.npz")
    return parser.parse_args()


def valid_center_indices(n_samples: int, temporal_offsets: list[int]) -> np.ndarray:
    start = max(0, -min(temporal_offsets))
    stop = n_samples - max(0, max(temporal_offsets))
    if stop <= start:
        raise ValueError(f"No valid samples for n_samples={n_samples} and offsets={temporal_offsets}")
    return np.arange(start, stop, dtype=np.int64)


def load_target_samples(eval_data_dir: Path, time_source: Path) -> tuple[np.ndarray, np.ndarray]:
    x_path = eval_data_dir / "X.npy"
    y_path = eval_data_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing evaluation data in {eval_data_dir}")
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    time_index = np.load(time_source, mmap_mode="r")
    if x.shape[0] != y.shape[0] or x.shape[0] != time_index.shape[0]:
        raise ValueError(f"Sample mismatch: X={x.shape}, y={y.shape}, time={time_index.shape}")
    if x.shape[1] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} channels, got {x.shape[1]}")
    sample_index = valid_center_indices(x.shape[0], TEMPORAL_OFFSETS)
    target_time = np.asarray(time_index[sample_index], dtype=np.int64)
    return sample_index, target_time


def load_progress(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        return np.asarray([], dtype=np.int64), None
    try:
        with np.load(path) as data:
            return data["time_index"].astype(np.int64), data["data"].astype(np.float32)
    except (zipfile.BadZipFile, EOFError, KeyError, ValueError) as exc:
        bad_path = path.with_suffix(path.suffix + ".bad")
        counter = 1
        while bad_path.exists():
            bad_path = path.with_suffix(path.suffix + f".bad{counter}")
            counter += 1
        path.rename(bad_path)
        print(f"Warning: moved unreadable progress file to {bad_path}: {exc}", flush=True)
        return np.asarray([], dtype=np.int64), None


def save_progress(path: Path, time_index: np.ndarray, data: np.ndarray) -> None:
    order = np.argsort(time_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, time_index=time_index[order].astype(np.int64), data=data[order].astype(np.float32))
    os.replace(tmp_path, path)


def append_progress(path: Path, new_time: int, new_map: np.ndarray) -> None:
    old_time, old_data = load_progress(path)
    time_cat = np.concatenate([old_time, np.asarray([new_time], dtype=np.int64)])
    data_new = np.asarray([new_map], dtype=np.float32)
    data_cat = data_new if old_data is None else np.concatenate([old_data, data_new], axis=0)
    reverse = np.arange(time_cat.size - 1, -1, -1)
    _, first = np.unique(time_cat[reverse], return_index=True)
    keep = np.sort(reverse[first])
    save_progress(path, time_cat[keep], data_cat[keep])


def build_grid(grid_run_dir: Path, box_j: int, box_i: int) -> xr.Dataset:
    print(f"Reading MITgcm grid from {grid_run_dir}", flush=True)
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_run_dir)
    print(f"Building coarse grid with box_j={box_j}, box_i={box_i}", flush=True)
    return build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=box_j, box_i=box_i)


def open_fk_inputs(nc_dir: Path, engine: str) -> dict[str, xr.DataArray | set[int] | Path]:
    ds_kpphbl = xr.open_dataset(nc_dir / "kpphbl_12hmean_coarse.nc", engine=engine, decode_times=False)
    ds_t = xr.open_dataset(nc_dir / "T_12hmean_coarse.nc", engine=engine, decode_times=False)
    ds_s = xr.open_dataset(nc_dir / "S_12hmean_coarse.nc", engine=engine, decode_times=False)
    ds_eta = xr.open_dataset(nc_dir / "Eta_12hmean_coarse.nc", engine=engine, decode_times=False)
    kpphbl_da = ds_kpphbl["kpphbl"]
    t_da = ds_t["T"]
    s_da = ds_s["S"]
    eta_da = ds_eta["Eta"]
    common = set(np.asarray(kpphbl_da["time"].values, dtype=np.int64).tolist())
    common &= set(np.asarray(t_da["time"].values, dtype=np.int64).tolist())
    common &= set(np.asarray(s_da["time"].values, dtype=np.int64).tolist())
    common &= set(np.asarray(eta_da["time"].values, dtype=np.int64).tolist())
    return {
        "dir": nc_dir,
        "times": common,
        "kpphbl": kpphbl_da,
        "T": t_da,
        "S": s_da,
        "Eta": eta_da,
    }


def choose_inputs(time: int, inputs: list[dict[str, object]]) -> dict[str, object] | None:
    for item in inputs:
        if time in item["times"]:
            return item
    return None


def calculate_fk(args: argparse.Namespace, sample_index: np.ndarray, target_time: np.ndarray) -> None:
    ds_grid = build_grid(args.grid_run_dir, args.box_j, args.box_i)
    inputs = [open_fk_inputs(path, args.engine) for path in args.nc_dirs]
    available_all = set().union(*(item["times"] for item in inputs))
    missing_from_nc = [int(t) for t in target_time if int(t) not in available_all]
    if missing_from_nc:
        raise RuntimeError(f"FK input NC files miss {len(missing_from_nc)} target times. First missing: {missing_from_nc[:8]}")

    saved_time, _ = load_progress(args.progress)
    saved_set = set(map(int, saved_time))
    todo = [(int(s), int(t)) for s, t in zip(sample_index, target_time) if int(t) not in saved_set]
    if args.max_snapshots is not None:
        todo = todo[: args.max_snapshots]

    print(f"Evaluation samples: {sample_index.size}", flush=True)
    print(f"Existing progress snapshots: {len(saved_set)}", flush=True)
    print(f"Need to compute in this run: {len(todo)}", flush=True)
    for n, (sample, time) in enumerate(todo, start=1):
        item = choose_inputs(time, inputs)
        if item is None:
            raise RuntimeError(f"No FK input dataset found for time_index={time}")
        print(f"[{n}/{len(todo)}] sample={sample} time_index={time} C_e={args.c_e} LfMin={args.lf_min} nc={item['dir']}", flush=True)
        fk_map = compute_fk_submeso_heat_flux_one_time(
            time_index=time,
            ds_grid=ds_grid,
            kpphbl_da=item["kpphbl"],
            t_da=item["T"],
            s_da=item["S"],
            eta_da=item["Eta"],
            c_e=args.c_e,
            submeso_lmax=args.submeso_lmax,
            submeso_inv_tau=args.submeso_inv_tau,
            lf_min=args.lf_min,
            f0=args.f0,
            beta=args.beta,
        )
        append_progress(args.progress, time, fk_map)


def save_prediction(args: argparse.Namespace, sample_index: np.ndarray, target_time: np.ndarray) -> bool:
    saved_time, saved_data = load_progress(args.progress)
    if saved_data is None:
        raise FileNotFoundError(f"No progress data found at {args.progress}")
    lookup = {int(t): n for n, t in enumerate(saved_time)}
    if args.allow_partial_output:
        keep = np.asarray([n for n, t in enumerate(target_time) if int(t) in lookup], dtype=np.int64)
    else:
        missing = [int(t) for t in target_time if int(t) not in lookup]
        if missing:
            print(f"Final output not saved yet; missing {len(missing)} snapshots. First missing: {missing[:8]}", flush=True)
            return False
        keep = np.arange(target_time.size, dtype=np.int64)

    if keep.size == 0:
        raise RuntimeError("No progress snapshots overlap target evaluation times.")

    j0, j1, i0, i1 = args.crop
    full = np.stack([saved_data[lookup[int(target_time[n])]] for n in keep], axis=0)
    cropped = full[:, None, j0:j1, i0:i1]
    if cropped.shape[-2:] != (20, 40):
        raise ValueError(f"Expected cropped lower_40x20 maps, got {cropped.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_index=sample_index[keep].astype(np.int64),
        time_index=target_time[keep].astype(np.int64),
        prediction=cropped.astype(np.float32),
        fk_heat_flux=cropped.astype(np.float32),
        units=np.asarray(["W m^-2"]),
        c_e=np.asarray([args.c_e], dtype=np.float64),
        lf_min=np.asarray([args.lf_min], dtype=np.float64),
        submeso_lmax=np.asarray([args.submeso_lmax], dtype=np.float64),
        submeso_inv_tau=np.asarray([args.submeso_inv_tau], dtype=np.float64),
        crop_j=np.asarray([j0, j1], dtype=np.int64),
        crop_i=np.asarray([i0, i1], dtype=np.int64),
        source_eval_data_dir=np.asarray([str(args.eval_data_dir)]),
    )
    print(f"Saved FK prediction: {args.output} {cropped.shape}", flush=True)
    return True


def main() -> None:
    args = parse_args()
    sample_index, target_time = load_target_samples(args.eval_data_dir, args.time_source)
    print(f"Target evaluation samples: {sample_index.size}", flush=True)
    print(f"Sample range: {int(sample_index[0])} ... {int(sample_index[-1])}", flush=True)
    print(f"Time range: {int(target_time[0])} ... {int(target_time[-1])}", flush=True)
    print(f"Lower crop on FK grid: j={args.crop[0]}:{args.crop[1]}, i={args.crop[2]}:{args.crop[3]}", flush=True)

    if not args.assemble_only:
        calculate_fk(args, sample_index, target_time)
    save_prediction(args, sample_index, target_time)


if __name__ == "__main__":
    main()
