#!/usr/bin/env python3
"""Save resumable 12-snapshot means for U, V, W, T, Eta, and kpphbl."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from MITgcmutils import mds


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_mitgcm  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_1km/verification/tutorial_reentrant_channel/run"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "twelve_snapshot_means"

VAR_DIMS = {
    "U": ("k", "j", "i_g"),
    "V": ("k", "j_g", "i"),
    "W": ("k_l", "j", "i"),
    "T": ("k", "j", "i"),
    "Eta": ("j", "i"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save 12-snapshot means from MITgcm MDS output.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, required=True, help="First snapshot iteration.")
    parser.add_argument("--end", type=int, required=True, help="Last snapshot iteration to consider.")
    parser.add_argument("--interval", type=int, default=90, help="Iteration interval between snapshots.")
    parser.add_argument("--block-size", type=int, default=12, help="Number of snapshots per mean.")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["U", "V", "W", "T", "Eta", "kpphbl"],
        choices=["U", "V", "W", "T", "Eta", "kpphbl"],
    )
    parser.add_argument(
        "--kpphbl-source",
        default="Diags/DiagMXL_2d",
        help="MDS base path for mixed-layer diagnostics, relative to run-dir.",
    )
    parser.add_argument(
        "--kpphbl-index",
        type=int,
        default=1,
        help="Variable index inside kpphbl-source. Existing scripts use index 1 for kpphbl.",
    )
    return parser.parse_args()


def snapshot_iterations(start: int, end: int, interval: int) -> np.ndarray:
    iterations = np.arange(start, end + interval, interval, dtype=np.int64)
    return iterations[iterations <= end]


def complete_blocks(iterations: np.ndarray, block_size: int) -> list[np.ndarray]:
    n_blocks = iterations.size // block_size
    return [iterations[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]


def read_snapshot(run_dir: Path, varname: str, iteration: int, kpphbl_source: str, kpphbl_index: int) -> xr.DataArray:
    if varname == "kpphbl":
        arr = np.asarray(mds.rdmds(str(run_dir / kpphbl_source), int(iteration)))
        if arr.ndim == 3:
            arr = arr[kpphbl_index, :, :]
        return xr.DataArray(arr, dims=("j", "i"), name="kpphbl")
    return read_mitgcm._ocean_var(run_dir, varname, int(iteration), VAR_DIMS[varname])


def mean_block(
    run_dir: Path,
    varname: str,
    block_iterations: np.ndarray,
    kpphbl_source: str,
    kpphbl_index: int,
) -> xr.DataArray:
    snapshots = [
        read_snapshot(run_dir, varname, int(iteration), kpphbl_source, kpphbl_index)
        for iteration in block_iterations
    ]
    stacked = xr.concat(snapshots, dim=xr.DataArray(block_iterations, dims="snapshot", name="snapshot"))
    return stacked.mean("snapshot", keep_attrs=True).astype("float32")


def load_existing_variable(output_path: Path, varname: str) -> tuple[list[xr.DataArray], list[int], list[int]]:
    if not output_path.exists():
        return [], [], []

    with xr.open_dataset(output_path, decode_times=False) as existing:
        existing = existing.load()

    if varname not in existing:
        raise KeyError(f"{output_path} exists but does not contain variable {varname!r}")
    if "time_start" not in existing or "time_end" not in existing:
        raise KeyError(f"{output_path} exists but is missing time_start/time_end coordinates")

    starts = [int(value) for value in existing["time_start"].values]
    ends = [int(value) for value in existing["time_end"].values]
    means = [existing[varname].isel(time=index, drop=True) for index in range(existing.sizes["time"])]
    return means, starts, ends


def dataset_from_blocks(
    varname: str,
    means: list[xr.DataArray],
    starts: list[int],
    ends: list[int],
    run_dir: Path,
    block_size: int,
) -> xr.Dataset:
    order = np.argsort(np.asarray(starts, dtype=np.int64))
    starts_sorted = [starts[int(index)] for index in order]
    ends_sorted = [ends[int(index)] for index in order]
    means_sorted = [means[int(index)] for index in order]
    centers = [0.5 * (start + end) for start, end in zip(starts_sorted, ends_sorted)]

    time = xr.DataArray(np.asarray(centers, dtype=np.float64), dims="time", name="time")
    data = xr.concat(means_sorted, dim=time).rename(varname)
    return xr.Dataset(
        {varname: data},
        coords={
            "time_start": ("time", np.asarray(starts_sorted, dtype=np.int64)),
            "time_end": ("time", np.asarray(ends_sorted, dtype=np.int64)),
        },
        attrs={
            "source_run_dir": str(run_dir),
            "mean_definition": "Mean over complete consecutive snapshot blocks.",
            "block_size_snapshots": int(block_size),
            "variable": varname,
        },
    )


def write_dataset_atomic(ds: xr.Dataset, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ds.to_netcdf(tmp_path)
    os.replace(tmp_path, output_path)


def save_variable_means(
    run_dir: Path,
    output_dir: Path,
    varname: str,
    blocks: list[np.ndarray],
    kpphbl_source: str,
    kpphbl_index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{varname}_12snap_mean.nc"
    means, starts, ends = load_existing_variable(output_path, varname)
    saved_blocks = set(zip(starts, ends))

    if starts:
        print(f"{varname}: found {len(starts)} existing averaged blocks in {output_path}", flush=True)

    for block_number, block in enumerate(blocks, start=1):
        block_start = int(block[0])
        block_end = int(block[-1])
        if (block_start, block_end) in saved_blocks:
            print(f"{varname}: skip existing block {block_number}/{len(blocks)} {block_start}-{block_end}", flush=True)
            continue

        print(f"{varname}: averaging block {block_number}/{len(blocks)} {block_start}-{block_end}", flush=True)
        mean = mean_block(run_dir, varname, block, kpphbl_source, kpphbl_index)
        means.append(mean)
        starts.append(block_start)
        ends.append(block_end)
        saved_blocks.add((block_start, block_end))

        ds = dataset_from_blocks(varname, means, starts, ends, run_dir, len(block))
        write_dataset_atomic(ds, output_path)
        print(f"{varname}: saved block {block_number}/{len(blocks)} to {output_path}", flush=True)

    if means:
        print(f"{varname}: complete, {len(means)} averaged blocks saved in {output_path}", flush=True)
    else:
        print(f"{varname}: no blocks available to save", flush=True)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    iterations = snapshot_iterations(args.start, args.end, args.interval)
    blocks = complete_blocks(iterations, args.block_size)
    if not blocks:
        raise ValueError("No complete averaging blocks selected.")

    print(
        f"Selected {iterations.size} snapshots and {len(blocks)} complete blocks "
        f"of {args.block_size} snapshots.",
        flush=True,
    )
    for varname in args.variables:
        save_variable_means(
            run_dir,
            output_dir,
            varname,
            blocks,
            args.kpphbl_source,
            args.kpphbl_index,
        )


if __name__ == "__main__":
    main()
