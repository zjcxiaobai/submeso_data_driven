#!/usr/bin/env python3
"""Calculate coarse-grid vorticity, strain, and speed from 12-hour means."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5netcdf
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import build_coarse_grid  # noqa: E402
from src.utils.speed import horizontal_speed  # noqa: E402
from src.utils.strain import total_strain  # noqa: E402
from src.utils.vorticity import vorticity  # noqa: E402

from save_12hour_means_and_coarse_llc4320 import read_static_grid  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR
    / "twelve_hour_means_llc4320"
    / "LLC4320_12hour_mean_coarse_50km.nc"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "coarse_velocity_diagnostics"
DEFAULT_OUTPUT_NAME = "LLC4320_12hour_coarse_vorticity_strain_speed.nc"
DIAGNOSTICS = ("vorticity", "strain", "speed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate 3-D vorticity, total strain magnitude, and horizontal "
            "speed from LLC4320 12-hour mean coarse U/V fields."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--k-chunk",
        type=int,
        default=8,
        help="Number of vertical levels processed together.",
    )
    parser.add_argument(
        "--max-new-blocks",
        type=int,
        default=None,
        help="Maximum unfinished time blocks to process in this run.",
    )
    parser.add_argument("--y-boundary", choices=("fill",), default="fill")
    return parser.parse_args()


def initialize_output(
    output_path: Path,
    sample: xr.Dataset,
    source_path: Path,
    box_j: int,
    box_i: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5netcdf.File(output_path, "w") as nc:
        nc.dimensions["time"] = None
        for dim in ("k", "j", "i"):
            nc.dimensions[dim] = int(sample.sizes[dim])

        nc.attrs["source_12hour_coarse_file"] = str(source_path)
        nc.attrs["box_j"] = int(box_j)
        nc.attrs["box_i"] = int(box_i)
        nc.attrs["diagnostic_grid"] = "same coarse tracer-center grid as source"
        nc.attrs["time_encoding"] = "YYYYMMDDHH"

        for name in ("time_start", "time_end"):
            variable = nc.create_variable(
                name,
                dimensions=("time",),
                dtype="i8",
                chunks=(1,),
                maxshape=(None,),
            )
            variable.attrs["units"] = "YYYYMMDDHH"

        for name in DIAGNOSTICS:
            field = sample[name]
            variable = nc.create_variable(
                name,
                dimensions=("time", "k", "j", "i"),
                dtype="f4",
                chunks=(
                    1,
                    min(8, int(field.sizes["k"])),
                    int(field.sizes["j"]),
                    int(field.sizes["i"]),
                ),
                maxshape=(
                    None,
                    int(field.sizes["k"]),
                    int(field.sizes["j"]),
                    int(field.sizes["i"]),
                ),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            for key, value in field.attrs.items():
                if value not in (None, ""):
                    variable.attrs[key] = value


def repair_and_saved_times(output_path: Path) -> set[int]:
    """Remove an interrupted partial append and return completed timestamps."""
    if not output_path.exists():
        return set()

    with h5netcdf.File(output_path, "a") as nc:
        if "time_start" not in nc.variables:
            raise ValueError(f"{output_path} is missing time_start")
        completed = int(nc.variables["time_start"].shape[0])
        for name in ("time_end",) + DIAGNOSTICS:
            if name not in nc.variables:
                raise ValueError(f"{output_path} is missing {name}")
            variable = nc.variables[name]
            if variable.shape[0] < completed:
                raise ValueError(f"{output_path}: {name} is shorter than time_start")
            if variable.shape[0] > completed:
                variable._h5ds.resize((completed,) + variable.shape[1:])
        return set(map(int, nc.variables["time_start"][:].tolist()))


def append_block(
    output_path: Path,
    diagnostics: xr.Dataset,
    time_start: int,
    time_end: int,
) -> None:
    """Append diagnostics and write time_start last as the completion marker."""
    with h5netcdf.File(output_path, "a") as nc:
        write_index = int(nc.variables["time_start"].shape[0])

        for name in DIAGNOSTICS:
            variable = nc.variables[name]
            variable._h5ds.resize((write_index + 1,) + variable.shape[1:])
            variable[write_index] = diagnostics[name].values.astype(
                np.float32, copy=False
            )

        end_variable = nc.variables["time_end"]
        end_variable._h5ds.resize((write_index + 1,))
        end_variable[write_index] = np.int64(time_end)

        start_variable = nc.variables["time_start"]
        start_variable._h5ds.resize((write_index + 1,))
        start_variable[write_index] = np.int64(time_start)
        nc.flush()


def grid_chunk(ds_coarse_grid: xr.Dataset, k_slice: slice) -> xr.Dataset:
    """Select one vertical slab while retaining all horizontal grid metrics."""
    vertical_names = {"DRF", "RC", "hFacC", "hFacW", "hFacS"}
    variables = {}
    for name, field in ds_coarse_grid.data_vars.items():
        if name in vertical_names and "k" in field.dims:
            variables[name] = field.isel(k=k_slice)
        elif "k" not in field.dims and "k_l" not in field.dims:
            variables[name] = field
    return xr.Dataset(variables)


def normalize_staggered_sizes(
    ds_coarse_grid: xr.Dataset,
    center_j: int,
    center_i: int,
) -> xr.Dataset:
    """Trim terminal coarse faces to the repo's equal-length C-grid convention."""
    if ds_coarse_grid.sizes["j_g"] < center_j:
        raise ValueError("Coarse grid has fewer j_g faces than j centers")
    if ds_coarse_grid.sizes["i_g"] < center_i:
        raise ValueError("Coarse grid has fewer i_g faces than i centers")

    normalized = ds_coarse_grid.isel(
        j=slice(0, center_j),
        j_g=slice(0, center_j),
        i=slice(0, center_i),
        i_g=slice(0, center_i),
    )
    return normalized.assign_coords(
        j=np.arange(center_j, dtype=np.int64),
        j_g=np.arange(center_j, dtype=np.int64),
        i=np.arange(center_i, dtype=np.int64),
        i_g=np.arange(center_i, dtype=np.int64),
    )


def calculate_slab(
    ds_grid: xr.Dataset,
    u: xr.DataArray,
    v: xr.DataArray,
    y_boundary: str,
) -> xr.Dataset:
    ds_uv = xr.Dataset({"U": u, "V": v})
    zeta = vorticity(
        ds_grid,
        ds_uv,
        y_boundary=y_boundary,
        output_name="vorticity",
    )
    strain = total_strain(
        ds_grid,
        ds_uv,
        y_boundary=y_boundary,
        output_name="strain",
    )
    speed = horizontal_speed(
        ds_grid,
        ds_uv,
        y_boundary=y_boundary,
        output_name="speed",
    )
    return xr.Dataset(
        {
            "vorticity": zeta.astype("float32"),
            "strain": strain.astype("float32"),
            "speed": speed.astype("float32"),
        }
    )


def calculate_time_block(
    source: xr.Dataset,
    ds_coarse_grid: xr.Dataset,
    time_index: int,
    k_chunk: int,
    y_boundary: str,
) -> xr.Dataset:
    nk = int(source.sizes["k"])
    slabs = []
    for k_start in range(0, nk, k_chunk):
        k_stop = min(k_start + k_chunk, nk)
        print(f"      vertical levels {k_start}:{k_stop}", flush=True)
        k_slice = slice(k_start, k_stop)
        u = source["U"].isel(
            time=time_index,
            k=k_slice,
            i_g=slice(0, source.sizes["i"]),
            drop=True,
        ).load()
        v = source["V"].isel(
            time=time_index,
            k=k_slice,
            j_g=slice(0, source.sizes["j"]),
            drop=True,
        ).load()
        slab_grid = grid_chunk(ds_coarse_grid, k_slice)
        slabs.append(calculate_slab(slab_grid, u, v, y_boundary))

    diagnostics = xr.concat(slabs, dim="k")
    diagnostics = diagnostics.assign_coords(
        k=np.arange(nk, dtype=np.int64),
        j=np.arange(source.sizes["j"], dtype=np.int64),
        i=np.arange(source.sizes["i"], dtype=np.int64),
    )
    return diagnostics.transpose("k", "j", "i")


def main() -> None:
    args = parse_args()
    if args.k_chunk <= 0:
        raise ValueError("--k-chunk must be positive")
    if args.max_new_blocks is not None and args.max_new_blocks <= 0:
        raise ValueError("--max-new-blocks must be positive")

    input_path = args.input.expanduser().resolve()
    output_path = args.output_dir.expanduser().resolve() / args.output_name

    with xr.open_dataset(
        input_path,
        engine="h5netcdf",
        decode_times=False,
        chunks=None,
    ) as source:
        required = {"time_start", "time_end", "U", "V"}
        missing = sorted(required - set(source.variables))
        if missing:
            raise KeyError(f"{input_path} is missing variables: {missing}")

        source_dir = Path(str(source.attrs["source_data_dir"]))
        source_file = source_dir / str(source.attrs["source_first_file"])
        box_j = int(source.attrs["box_j"])
        box_i = int(source.attrs["box_i"])

        print("Building the matching coarse grid metrics...", flush=True)
        ds_fine_grid = read_static_grid(source_file)
        ds_coarse_grid = build_coarse_grid.build_coarse_grid_staggered(
            ds_fine_grid,
            box_j=box_j,
            box_i=box_i,
        )
        ds_coarse_grid = normalize_staggered_sizes(
            ds_coarse_grid,
            center_j=int(source.sizes["j"]),
            center_i=int(source.sizes["i"]),
        )

        saved = repair_and_saved_times(output_path)
        completed_this_run = 0
        nt = int(source.sizes["time"])

        for time_index in range(nt):
            time_start = int(source["time_start"].isel(time=time_index).item())
            time_end = int(source["time_end"].isel(time=time_index).item())
            if time_start in saved:
                print(f"[{time_index + 1}/{nt}] skip saved {time_start}", flush=True)
                continue
            if (
                args.max_new_blocks is not None
                and completed_this_run >= args.max_new_blocks
            ):
                print(
                    f"Reached the limit of {args.max_new_blocks} new blocks.",
                    flush=True,
                )
                break

            print(
                f"[{time_index + 1}/{nt}] calculating {time_start}-{time_end}",
                flush=True,
            )
            diagnostics = calculate_time_block(
                source,
                ds_coarse_grid,
                time_index,
                args.k_chunk,
                args.y_boundary,
            )

            if not output_path.exists():
                initialize_output(
                    output_path,
                    diagnostics,
                    input_path,
                    box_j,
                    box_i,
                )
            append_block(output_path, diagnostics, time_start, time_end)
            saved.add(time_start)
            completed_this_run += 1
            print(f"    saved {time_start}-{time_end}", flush=True)

    print(f"New blocks completed in this run: {completed_this_run}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
