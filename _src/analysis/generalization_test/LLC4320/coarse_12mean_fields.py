#!/usr/bin/env python3
"""Spatially coarse 12-snapshot mean fields and submesoscale heat flux."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import build_coarse_grid, coarsen_UVWTS, read_mitgcm  # noqa: E402
from src.utils.wT_submeso import submeso_vertical_heat_flux  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_1km/verification/tutorial_reentrant_channel/run"
)
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "twelve_snapshot_means"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "twelve_snapshot_means" / "coarse_12mean_50box.nc"

VARIABLES = ("U", "V", "W", "T", "Eta", "kpphbl")
COARSEN_NAMES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "T": "T",
    "Eta": "Eta",
    "kpphbl": "kpphbl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read 12-snapshot mean U/V/W/T/Eta/kpphbl, coarse them in space, "
            "compute submesoscale heat flux on the same coarse grid, and save NetCDF."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--box-size",
        type=int,
        default=50,
        help="Coarse box size in fine-grid points. Use 50 for 1 km -> 50 km; use 100 for 500 m -> 50 km.",
    )
    parser.add_argument("--box-j", type=int, default=None, help="Override meridional coarse box size.")
    parser.add_argument("--box-i", type=int, default=None, help="Override zonal coarse box size.")
    parser.add_argument("--boundary-z", default="fill")
    parser.add_argument("--to-wm2", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def open_mean_files(input_dir: Path) -> dict[str, xr.Dataset]:
    datasets = {}
    for varname in VARIABLES:
        path = input_dir / f"{varname}_12snap_mean.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        datasets[varname] = xr.open_dataset(path, decode_times=False)
    return datasets


def common_time_starts(datasets: dict[str, xr.Dataset]) -> np.ndarray:
    common = None
    for varname, ds in datasets.items():
        if "time_start" not in ds or "time_end" not in ds:
            raise KeyError(f"{varname} file is missing time_start/time_end")
        starts = ds["time_start"].values.astype(np.int64)
        common = starts if common is None else np.intersect1d(common, starts)
    if common is None or common.size == 0:
        raise ValueError("No common 12-snapshot blocks across all input variables.")
    return common


def select_var_at_start(ds: xr.Dataset, varname: str, start: int) -> xr.DataArray:
    matches = np.where(ds["time_start"].values.astype(np.int64) == int(start))[0]
    if matches.size != 1:
        raise ValueError(f"{varname}: expected one block with time_start={start}, got {matches.size}")
    return ds[varname].isel(time=int(matches[0]), drop=True).reset_coords(drop=True)


def end_at_start(ds: xr.Dataset, start: int) -> int:
    matches = np.where(ds["time_start"].values.astype(np.int64) == int(start))[0]
    if matches.size != 1:
        raise ValueError(f"Expected one block with time_start={start}, got {matches.size}")
    return int(ds["time_end"].isel(time=int(matches[0])).item())


def load_existing(output_path: Path) -> tuple[list[xr.Dataset], list[int], list[int]]:
    if not output_path.exists():
        return [], [], []

    with xr.open_dataset(output_path, decode_times=False) as existing:
        existing = existing.load()

    starts = [int(value) for value in existing["time_start"].values]
    ends = [int(value) for value in existing["time_end"].values]
    blocks = [existing.isel(time=index, drop=True).reset_coords(drop=True) for index in range(existing.sizes["time"])]
    return blocks, starts, ends


def dataset_from_blocks(
    blocks: list[xr.Dataset],
    starts: list[int],
    ends: list[int],
    run_dir: Path,
    input_dir: Path,
    box_j: int,
    box_i: int,
    to_wm2: bool,
) -> xr.Dataset:
    order = np.argsort(np.asarray(starts, dtype=np.int64))
    blocks = [blocks[int(index)] for index in order]
    starts = [starts[int(index)] for index in order]
    ends = [ends[int(index)] for index in order]
    centers = np.asarray([0.5 * (start + end) for start, end in zip(starts, ends)], dtype=np.float64)

    ds = xr.concat(blocks, dim=xr.DataArray(centers, dims="time", name="time"))
    ds = ds.assign_coords(
        time_start=("time", np.asarray(starts, dtype=np.int64)),
        time_end=("time", np.asarray(ends, dtype=np.int64)),
    )
    ds.attrs.update(
        source_run_dir=str(run_dir),
        source_12_snapshot_mean_dir=str(input_dir),
        spatial_coarse_box_j=int(box_j),
        spatial_coarse_box_i=int(box_i),
        submeso_wT_units="W m^-2" if to_wm2 else "K m s^-1",
        note="All fields are 12-snapshot means spatially coarsened to the requested box.",
    )
    return ds


def write_dataset_atomic(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ds.to_netcdf(tmp_path)
    os.replace(tmp_path, output_path)


def coarse_one_block(
    datasets: dict[str, xr.Dataset],
    ds_fine: xr.Dataset,
    ds_coarse: xr.Dataset,
    start: int,
    box_j: int,
    box_i: int,
    boundary_z: str,
    to_wm2: bool,
) -> tuple[xr.Dataset, int]:
    fields = {varname: select_var_at_start(datasets[varname], varname, start) for varname in VARIABLES}
    ends = {varname: end_at_start(datasets[varname], start) for varname in VARIABLES}
    unique_ends = sorted(set(ends.values()))
    if len(unique_ends) != 1:
        raise ValueError(f"time_end mismatch for time_start={start}: {ends}")

    coarse_fields = {
        varname: coarsen_UVWTS.coarse_field(
            COARSEN_NAMES[varname],
            fields[varname],
            ds_fine,
            box_j,
            box_i,
        )
        .astype("float32")
        .rename(varname)
        for varname in VARIABLES
    }

    submeso_wT = submeso_vertical_heat_flux(
        ds_fine,
        ds_coarse,
        xr.Dataset({"W": fields["W"], "T": fields["T"]}),
        xr.Dataset({"W": coarse_fields["W"], "T": coarse_fields["T"]}),
        boundary_Z=boundary_z,
        box_j=box_j,
        box_i=box_i,
        to_wm2=to_wm2,
        output="submeso_center",
    )
    coarse_fields["wT_submeso"] = submeso_wT.astype("float32").rename("wT_submeso")

    return xr.Dataset(coarse_fields), unique_ends[0]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    box_j = int(args.box_j if args.box_j is not None else args.box_size)
    box_i = int(args.box_i if args.box_i is not None else args.box_size)

    datasets = open_mean_files(input_dir)
    ds_fine = read_mitgcm.read_mitgrid_staggered(run_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=box_j, box_i=box_i)

    blocks, starts, ends = load_existing(output_path)
    saved = set(starts)
    common_starts = common_time_starts(datasets)

    print(
        f"Coarsening {len(common_starts)} common 12-snapshot blocks with box_j={box_j}, box_i={box_i}.",
        flush=True,
    )

    for block_number, start in enumerate(common_starts, start=1):
        start = int(start)
        if start in saved:
            print(f"skip existing block {block_number}/{len(common_starts)} time_start={start}", flush=True)
            continue

        print(f"coarsen block {block_number}/{len(common_starts)} time_start={start}", flush=True)
        block_ds, end = coarse_one_block(
            datasets,
            ds_fine,
            ds_coarse,
            start,
            box_j,
            box_i,
            args.boundary_z,
            args.to_wm2,
        )
        blocks.append(block_ds)
        starts.append(start)
        ends.append(end)
        saved.add(start)

        out = dataset_from_blocks(blocks, starts, ends, run_dir, input_dir, box_j, box_i, args.to_wm2)
        write_dataset_atomic(out, output_path)
        print(f"saved block {block_number}/{len(common_starts)} {start}-{end} to {output_path}", flush=True)

    for ds in datasets.values():
        ds.close()

    print(f"done: {len(blocks)} blocks saved in {output_path}", flush=True)


if __name__ == "__main__":
    main()
