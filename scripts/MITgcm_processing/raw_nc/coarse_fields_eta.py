#!/usr/bin/env python3
"""Create resumable 12-hour coarse Eta fields from MITgcm snapshots."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import h5netcdf
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_mitgcm  # noqa: E402
from src.utils.coarsen_UVWTS import coarse_field  # noqa: E402


DEFAULT_GRID_DIR = Path(
    "/scratch/jczhang/"
    "MITgcm_baroclinic_instability_old/"
    "MITgcm/verification/tutorial_reentrant_channel/run"
)
DEFAULT_OUT_NC = PROJECT_ROOT / "13901616_nc" / "Eta_12hmean_coarse.nc"


DIMS_MAP = {
    "Eta": ("j", "i"),
}


def make_mitgcm_loader_onevar(
    grid_dir: Path,
    varname: str,
    dims: tuple[str, ...],
) -> Callable[[int], xr.DataArray]:
    """Return a loader for one MITgcm variable at one iteration."""

    def load_snapshot(iteration: int) -> xr.DataArray:
        da = read_mitgcm._ocean_var(str(grid_dir), varname, int(iteration), dims)
        da = da.drop_vars("time", errors="ignore")
        da.name = varname
        return da

    return load_snapshot


def _read_existing_state(
    out_nc: Path,
    time_name: str,
    varname: str,
    box_j: int,
    box_i: int,
    start_time_index: int,
) -> tuple[set[int], int]:
    """Read saved block starts and validate that an existing file is compatible."""
    saved: set[int] = set()
    origin_iter = start_time_index

    if not out_nc.exists():
        return saved, origin_iter

    with h5netcdf.File(out_nc, "r") as nc:
        if time_name in nc.variables:
            saved = set(map(int, nc.variables[time_name][:].tolist()))
        if "origin_iter" in nc.attrs:
            origin_iter = int(nc.attrs["origin_iter"])

        if "varname" in nc.attrs and str(nc.attrs["varname"]) != str(varname):
            raise ValueError(
                f"{out_nc} already stores varname={nc.attrs['varname']}, "
                f"but requested {varname}"
            )
        if "box_j" in nc.attrs and int(nc.attrs["box_j"]) != int(box_j):
            raise ValueError(f"{out_nc} box_j differs from requested box_j={box_j}")
        if "box_i" in nc.attrs and int(nc.attrs["box_i"]) != int(box_i):
            raise ValueError(f"{out_nc} box_i differs from requested box_i={box_i}")
        if time_name in nc.variables and varname in nc.variables:
            nt = int(nc.variables[time_name].shape[0])
            nv = int(nc.variables[varname].shape[0])
            if nt != nv:
                raise ValueError(
                    f"{out_nc} has inconsistent lengths: {time_name}={nt}, "
                    f"{varname}={nv}"
                )

    return saved, origin_iter


def _init_output_file(
    out_nc: Path,
    coarse_da: xr.DataArray,
    time_name: str,
    varname: str,
    origin_iter: int,
    step: int,
    block_size: int,
    box_j: int,
    box_i: int,
) -> None:
    """Initialize an extendable one-variable NetCDF file."""
    spatial_dims = tuple(coarse_da.dims)
    spatial_sizes = {dim: int(coarse_da.sizes[dim]) for dim in spatial_dims}

    out_nc.parent.mkdir(parents=True, exist_ok=True)
    with h5netcdf.File(out_nc, "w") as nc:
        nc.dimensions[time_name] = None
        for dim, size in spatial_sizes.items():
            nc.dimensions[dim] = size

        nc.attrs["origin_iter"] = int(origin_iter)
        nc.attrs["hourly_step"] = int(step)
        nc.attrs["block_size"] = int(block_size)
        nc.attrs["time_is_block_start"] = 1
        nc.attrs["box_j"] = int(box_j)
        nc.attrs["box_i"] = int(box_i)
        nc.attrs["varname"] = str(varname)

        time_var = nc.create_variable(
            time_name,
            dimensions=(time_name,),
            dtype="i8",
            chunks=(1,),
            maxshape=(None,),
        )
        time_var.attrs["long_name"] = (
            f"start index of each {block_size}-snapshot averaging block"
        )
        time_var.attrs["units"] = "model_iteration_index"

        data_dims = (time_name,) + spatial_dims
        data_chunks = (1,) + tuple(spatial_sizes[dim] for dim in spatial_dims)
        data_shape = (None,) + tuple(spatial_sizes[dim] for dim in spatial_dims)
        data_var = nc.create_variable(
            varname,
            dimensions=data_dims,
            dtype="f4",
            chunks=data_chunks,
            maxshape=data_shape,
        )
        data_var.attrs["long_name"] = varname


def append_averaged_coarse_field(
    *,
    start_time_index: int,
    end_time_index: int,
    load_snapshot: Callable[[int], xr.DataArray],
    out_nc: Path,
    ds_fine: xr.Dataset,
    box_j: int,
    box_i: int,
    varname: str = "Eta",
    block_size: int = 12,
    step: int = 144,
    time_name: str = "time",
    dtype: np.dtype = np.float32,
    skip_missing: bool = False,
    print_every: int = 20,
) -> None:
    """Average snapshots, coarsen the mean field, and append missing blocks."""
    start_time_index = int(start_time_index)
    end_time_index = int(end_time_index)
    step = int(step)
    block_size = int(block_size)

    if end_time_index < start_time_index:
        raise ValueError("end_time_index must be >= start_time_index")
    if step <= 0:
        raise ValueError("step must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    saved, origin_iter = _read_existing_state(
        out_nc, time_name, varname, box_j, box_i, start_time_index
    )

    block_span = block_size * step
    max_n = int(
        np.floor((end_time_index - origin_iter - (block_size - 1) * step) / block_span)
    )
    if max_n < 0:
        print("Not enough data to form one complete averaging block.", flush=True)
        return

    appended = 0
    skipped_saved = 0
    skipped_missing = 0
    nc_app = None

    try:
        for n in range(max_n + 1):
            block_start = origin_iter + n * block_span
            if print_every and n % print_every == 0:
                print(f"[try] n={n}/{max_n} block_start={block_start}", flush=True)

            if block_start in saved:
                skipped_saved += 1
                continue

            block_iters = [block_start + offset * step for offset in range(block_size)]
            snap0 = None
            sum_arr = None

            try:
                for iteration in block_iters:
                    da = load_snapshot(iteration)
                    if snap0 is None:
                        snap0 = da
                        sum_arr = da.values.astype(dtype, copy=True)
                    else:
                        sum_arr += da.values.astype(dtype, copy=False)
            except Exception as exc:
                if skip_missing:
                    skipped_missing += 1
                    print(
                        f"  -> skip block_start={block_start}; unreadable snapshot: {exc}",
                        flush=True,
                    )
                    continue
                raise

            if snap0 is None or sum_arr is None:
                raise RuntimeError(f"No snapshots loaded for block_start={block_start}")

            mean_da = snap0.copy(data=sum_arr / block_size)
            mean_da.name = varname

            coarse_da = coarse_field(varname, mean_da, ds_fine, box_j=box_j, box_i=box_i)
            coarse_da = coarse_da.astype("float32")
            coarse_da.name = varname

            if not out_nc.exists():
                _init_output_file(
                    out_nc,
                    coarse_da,
                    time_name,
                    varname,
                    origin_iter,
                    step,
                    block_size,
                    box_j,
                    box_i,
                )

            if nc_app is None:
                nc_app = h5netcdf.File(out_nc, "a")

            time_var = nc_app.variables[time_name]
            write_idx = int(time_var.shape[0])

            data_var = nc_app.variables[varname]
            spatial_shape = data_var.shape[1:]
            data_var._h5ds.resize((write_idx + 1,) + spatial_shape)
            data_var[write_idx, ...] = coarse_da.values.astype(np.float32, copy=False)

            time_var._h5ds.resize((write_idx + 1,))
            time_var[write_idx] = block_start

            saved.add(block_start)
            appended += 1

    finally:
        if nc_app is not None:
            nc_app.close()

    print(
        f"\nUpdated NetCDF: {out_nc}\n"
        f"  var: {varname}\n"
        f"  appended: {appended}\n"
        f"  skipped already saved: {skipped_saved}\n"
        f"  skipped missing: {skipped_missing}\n"
        f"  origin_iter={origin_iter}, step={step}, block_size={block_size}\n"
        f"  coarse box: box_j={box_j}, box_i={box_i}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resumable coarse 12-hour mean Eta NetCDF output."
    )
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--out-nc", type=Path, default=DEFAULT_OUT_NC)
    parser.add_argument("--start", type=int, default=13901616)
    parser.add_argument("--end", type=int, default=26707984)
    parser.add_argument("--step", type=int, default=144)
    parser.add_argument("--block-size", type=int, default=12)
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()

    varname = "Eta"
    grid_dir = args.grid_dir.expanduser().resolve()
    out_nc = args.out_nc.expanduser().resolve()

    ds_fine = read_mitgcm.read_mitgrid_staggered(str(grid_dir))
    load_eta = make_mitgcm_loader_onevar(grid_dir, varname, DIMS_MAP[varname])

    append_averaged_coarse_field(
        start_time_index=args.start,
        end_time_index=args.end,
        load_snapshot=load_eta,
        out_nc=out_nc,
        ds_fine=ds_fine,
        box_j=args.box_j,
        box_i=args.box_i,
        varname=varname,
        block_size=args.block_size,
        step=args.step,
        skip_missing=args.skip_missing,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
