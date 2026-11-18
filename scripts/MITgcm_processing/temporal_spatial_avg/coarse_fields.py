"""Create resumable 12-hour coarse fields from MITgcm output."""

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5netcdf
import numpy as np
import xarray as xr
from MITgcmutils import mds


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import read_mitgcm
from utils.coarsen_UVWTS import coarse_field


GRID_DIR = Path(
    "/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"
)
OUTPUT_DIR = PROJECT_ROOT / "data/interim/temporal_spatial_avg"
DEFAULT_START = 26_400_096
DEFAULT_END = 26_401_680
DEFAULT_STEP = 144

FIELD_DIMS = {
    "T": ("k", "j", "i"),
    "S": ("k", "j", "i"),
    "U": ("k", "j", "i_g"),
    "V": ("k", "j_g", "i"),
    "W": ("k_l", "j", "i"),
    "Eta": ("j", "i"),
}
SUPPORTED_VARIABLES = (*FIELD_DIMS, "kpphbl")


def load_field(grid_dir: Path, variable: str, iteration: int) -> xr.DataArray:
    """Read one MITgcm field at one model iteration."""
    field = read_mitgcm._ocean_var(
        grid_dir,
        variable,
        iteration,
        FIELD_DIMS[variable],
    )
    return field.rename(variable)


def load_kpphbl(grid_dir: Path, iteration: int) -> xr.DataArray:
    """Read KPPhbl, the second field in DiagMXL_2d."""
    diagnostic = mds.rdmds(
        str(grid_dir / "Diags" / "DiagMXL_2d"),
        iteration,
    )
    return xr.DataArray(
        diagnostic[1],
        dims=("j", "i"),
        name="kpphbl",
    )


def write_time_blocks(
    compute_block: Callable[[list[int]], xr.DataArray],
    output_path: Path,
    variable: str,
    start: int,
    end: int,
    *,
    step: int = DEFAULT_STEP,
    block_size: int = 12,
    box_j: int = 10,
    box_i: int = 10,
    print_every: int = 1,
    attributes: dict | None = None,
) -> None:
    """Compute missing time blocks and append them to one NetCDF file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    block_span = block_size * step
    origin = start
    saved: set[int] = set()

    if output_path.exists():
        with h5netcdf.File(output_path, "r") as nc:
            if "time" in nc.variables:
                saved = set(map(int, nc.variables["time"][:].tolist()))
            origin = int(nc.attrs.get("origin_iter", start))

    last_block = int(
        np.floor((end - origin - (block_size - 1) * step) / block_span)
    )
    if last_block < 0:
        print("Not enough snapshots to form one complete time block.", flush=True)
        return

    appended = 0
    skipped = 0
    handle = None

    try:
        for block_number in range(last_block + 1):
            block_start = origin + block_number * block_span
            if block_start < start:
                continue
            if block_start in saved:
                skipped += 1
                continue
            if print_every and block_number % print_every == 0:
                print(
                    f"[block {block_number}/{last_block}] start={block_start}",
                    flush=True,
                )

            iterations = [
                block_start + offset * step
                for offset in range(block_size)
            ]
            result = compute_block(iterations).astype("float32").rename(variable)

            if not output_path.exists():
                _initialize_output(
                    output_path,
                    result,
                    variable,
                    origin,
                    step,
                    block_size,
                    box_j,
                    box_i,
                    attributes or {},
                )

            if handle is None:
                handle = h5netcdf.File(output_path, "a")

            time_var = handle.variables["time"]
            data_var = handle.variables[variable]
            position = int(time_var.shape[0])

            data_var._h5ds.resize((position + 1,) + data_var.shape[1:])
            data_var[position] = result.values
            time_var._h5ds.resize((position + 1,))
            time_var[position] = block_start

            saved.add(block_start)
            appended += 1
    finally:
        if handle is not None:
            handle.close()

    print(
        f"Saved: {output_path}\n"
        f"  variable: {variable}\n"
        f"  appended: {appended}\n"
        f"  already present: {skipped}",
        flush=True,
    )


def _initialize_output(
    path: Path,
    example: xr.DataArray,
    variable: str,
    origin: int,
    step: int,
    block_size: int,
    box_j: int,
    box_i: int,
    attributes: dict,
) -> None:
    """Create an extendable NetCDF file using one computed field as a template."""
    spatial_dims = tuple(example.dims)
    spatial_shape = tuple(example.sizes[dim] for dim in spatial_dims)

    with h5netcdf.File(path, "w") as nc:
        nc.dimensions["time"] = None
        for dim, size in zip(spatial_dims, spatial_shape):
            nc.dimensions[dim] = size

        nc.attrs.update(
            {
                "origin_iter": origin,
                "hourly_step": step,
                "block_size": block_size,
                "time_is_block_start": 1,
                "box_j": box_j,
                "box_i": box_i,
                "variable": variable,
                **attributes,
            }
        )

        time_var = nc.create_variable(
            "time",
            dimensions=("time",),
            dtype="i8",
            chunks=(1,),
            maxshape=(None,),
        )
        time_var.attrs["units"] = "model_iteration_index"

        data_var = nc.create_variable(
            variable,
            dimensions=("time",) + spatial_dims,
            dtype="f4",
            chunks=(1,) + spatial_shape,
            maxshape=(None,) + spatial_shape,
        )
        data_var.attrs.update(example.attrs)


def process_variable(
    variable: str,
    *,
    grid_dir: Path = GRID_DIR,
    output_path: Path | None = None,
    start: int = DEFAULT_START,
    end: int = DEFAULT_END,
    step: int = DEFAULT_STEP,
    block_size: int = 12,
    box_j: int = 10,
    box_i: int = 10,
) -> None:
    """Average and coarsen one supported MITgcm field."""
    if variable.lower() == "kpphbl":
        variable = "kpphbl"
    elif variable.lower() == "eta":
        variable = "Eta"
    else:
        variable = variable.upper()
    output_path = output_path or OUTPUT_DIR / f"{variable}_12hmean_coarse.nc"
    fine_grid = read_mitgcm.read_mitgrid_staggered(grid_dir)

    def compute_block(iterations: list[int]) -> xr.DataArray:
        def read(iteration: int) -> xr.DataArray:
            if variable == "kpphbl":
                return load_kpphbl(grid_dir, iteration)
            return load_field(grid_dir, variable, iteration)

        first = read(iterations[0])
        total = first.values.astype(np.float32, copy=True)
        for iteration in iterations[1:]:
            total += read(iteration).values.astype(np.float32, copy=False)
        mean = first.copy(data=total / len(iterations)).rename(variable)
        return coarse_field(variable, mean, fine_grid, box_j, box_i)

    write_time_blocks(
        compute_block,
        output_path,
        variable,
        start,
        end,
        step=step,
        block_size=block_size,
        box_j=box_j,
        box_i=box_i,
        attributes=(
            {"source_field": "DiagMXL_2d record 1: KPPhbl"}
            if variable == "kpphbl"
            else None
        ),
    )


def main(default_variable: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variable",
        nargs="?",
        default=default_variable,
        choices=SUPPORTED_VARIABLES,
    )
    parser.add_argument("--grid-dir", type=Path, default=GRID_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--block-size", type=int, default=12)
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    args = parser.parse_args()

    if args.variable is None:
        parser.error("variable is required: choose T, S, U, V, W, Eta, or kpphbl")

    process_variable(
        args.variable,
        grid_dir=args.grid_dir,
        output_path=args.output,
        start=args.start,
        end=args.end,
        step=args.step,
        block_size=args.block_size,
        box_j=args.box_j,
        box_i=args.box_i,
    )


if __name__ == "__main__":
    main()
