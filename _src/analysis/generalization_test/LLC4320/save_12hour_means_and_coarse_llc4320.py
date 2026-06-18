#!/usr/bin/env python3
"""Create serial, resumable LLC4320 12-hour mean and 50 km coarse files.

LLC4320 pre-SWOT input files are daily NetCDF4/HDF5 files containing 24 hourly
records. This script reads them directly with h5py, forms two 12-hour means per
day, and appends each result once to fine-grid and coarse-grid NetCDF files.
The script is ordinary single-process Python: no MPI and no Dask.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import h5netcdf
import h5py
import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import build_coarse_grid, coarsen_UVWTS  # noqa: E402
from src.utils.wT_submeso import submeso_vertical_heat_flux  # noqa: E402


DEFAULT_DATA_DIR = Path("/scratch/jczhang/LLC4320/Data")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "twelve_hour_means_llc4320"
DEFAULT_FINE_NAME = "LLC4320_12hour_mean.nc"
DEFAULT_COARSE_NAME = "LLC4320_12hour_mean_coarse_50km.nc"
FILE_PATTERN = "LLC4320_pre-SWOT_ACC_SMST_*.nc"
FILE_RE = re.compile(r"LLC4320_pre-SWOT_ACC_SMST_(\d{8})\.nc$")

VARIABLES = ("U", "V", "W", "T", "S", "Eta", "kpphbl")
SOURCE_NAMES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "T": "Theta",
    "S": "Salt",
    "Eta": "Eta",
    "kpphbl": "KPPhbl",
}
FINE_DIMS = {
    "U": ("k", "j", "i_g"),
    "V": ("k", "j_g", "i"),
    "W": ("k_l", "j", "i"),
    "T": ("k", "j", "i"),
    "S": ("k", "j", "i"),
    "Eta": ("j", "i"),
    "kpphbl": ("j", "i"),
}
COARSEN_NAMES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "T": "T",
    "S": "Salt",
    "Eta": "Eta",
    "kpphbl": "kpphbl",
}


@dataclass(frozen=True)
class DailyFile:
    path: Path
    day: date


@dataclass(frozen=True)
class TimeBlock:
    source: DailyFile
    start_hour: int
    stop_hour: int

    @property
    def time_start(self) -> int:
        return int(f"{self.source.day:%Y%m%d}{self.start_hour:02d}")

    @property
    def time_end(self) -> int:
        return int(f"{self.source.day:%Y%m%d}{self.stop_hour - 1:02d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build serial, resumable LLC4320 12-hour fine and coarse NetCDF files."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fine-name", default=DEFAULT_FINE_NAME)
    parser.add_argument("--coarse-name", default=DEFAULT_COARSE_NAME)
    parser.add_argument("--n-files", type=int, default=90)
    parser.add_argument(
        "--start-date",
        default=None,
        help="First daily file as YYYYMMDD. Default: first consecutive run.",
    )
    parser.add_argument("--hours-per-mean", type=int, default=12)
    parser.add_argument(
        "--max-new-blocks",
        "--max-blocks",
        dest="max_new_blocks",
        type=int,
        default=None,
        help="Maximum number of unfinished blocks to process in this run.",
    )
    parser.add_argument("--stage", choices=("all", "fine", "coarse"), default="all")
    parser.add_argument("--target-km", type=float, default=50.0)
    parser.add_argument("--box-j", type=int, default=None)
    parser.add_argument("--box-i", type=int, default=None)
    parser.add_argument("--boundary-z", default="fill")
    parser.add_argument("--to-wm2", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        choices=range(0, 10),
        metavar="0..9",
    )
    return parser.parse_args()


def list_daily_files(data_dir: Path) -> list[DailyFile]:
    files = []
    for path in sorted(data_dir.glob(FILE_PATTERN)):
        match = FILE_RE.fullmatch(path.name)
        if match:
            files.append(
                DailyFile(path, datetime.strptime(match.group(1), "%Y%m%d").date())
            )
    if not files:
        raise FileNotFoundError(f"No files matching {FILE_PATTERN} in {data_dir}")
    return files


def select_consecutive_files(
    files: list[DailyFile], n_files: int, start_date: str | None
) -> list[DailyFile]:
    if n_files <= 0:
        raise ValueError("--n-files must be positive")
    by_day = {item.day: item for item in files}
    if start_date is not None:
        first = datetime.strptime(start_date, "%Y%m%d").date()
        days = [first + timedelta(days=i) for i in range(n_files)]
        missing = [day for day in days if day not in by_day]
        if missing:
            text = ", ".join(f"{day:%Y%m%d}" for day in missing[:10])
            raise FileNotFoundError(f"Requested sequence has missing dates: {text}")
        return [by_day[day] for day in days]

    for first_index in range(len(files) - n_files + 1):
        candidate = files[first_index : first_index + n_files]
        first = candidate[0].day
        if all(item.day == first + timedelta(days=i) for i, item in enumerate(candidate)):
            return candidate
    raise ValueError(f"No run of {n_files} consecutive daily files was found")


def make_blocks(files: list[DailyFile], hours_per_mean: int) -> list[TimeBlock]:
    if hours_per_mean <= 0 or 24 % hours_per_mean:
        raise ValueError("--hours-per-mean must be a positive divisor of 24")
    return [
        TimeBlock(item, start, start + hours_per_mean)
        for item in files
        for start in range(0, 24, hours_per_mean)
    ]


def decode_attr(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_hdf5_time_mean(
    h5: h5py.File,
    source_name: str,
    start_hour: int,
    stop_hour: int,
    output_name: str,
) -> tuple[np.ndarray, dict[str, str]]:
    """Read one variable in temporal chunks and accumulate a float32 mean."""
    if source_name not in h5:
        raise KeyError(f"{h5.filename} does not contain {source_name!r}")
    dataset = h5[source_name]
    if dataset.ndim < 3 or dataset.shape[0] < stop_hour:
        raise ValueError(
            f"{source_name} shape {dataset.shape} cannot provide hours "
            f"{start_hour}:{stop_hour}"
        )

    accumulator = np.zeros(dataset.shape[1:], dtype=np.float32)
    valid_count = np.zeros(dataset.shape[1:], dtype=np.uint16)
    fill_value = dataset.attrs.get("_FillValue")
    fill_value = None if fill_value is None else float(np.asarray(fill_value).reshape(-1)[0])
    valid_min = dataset.attrs.get("valid_min")
    valid_min = None if valid_min is None else float(np.asarray(valid_min).reshape(-1)[0])
    valid_max = dataset.attrs.get("valid_max")
    valid_max = None if valid_max is None else float(np.asarray(valid_max).reshape(-1)[0])

    time_chunk = dataset.chunks[0] if dataset.chunks else 1
    read_step = max(1, min(int(time_chunk), stop_hour - start_hour))
    for read_start in range(start_hour, stop_hour, read_step):
        read_stop = min(read_start + read_step, stop_hour)
        values = np.asarray(dataset[read_start:read_stop], dtype=np.float32)
        valid = np.isfinite(values)
        if fill_value is not None:
            valid &= values != np.float32(fill_value)
        if valid_min is not None:
            valid &= values >= np.float32(valid_min)
        if valid_max is not None:
            valid &= values <= np.float32(valid_max)
        accumulator += np.where(valid, values, 0.0).sum(axis=0, dtype=np.float32)
        valid_count += valid.sum(axis=0, dtype=np.uint16)
        print(f"      {output_name}: read hours {read_start:02d}-{read_stop:02d}", flush=True)

    attrs = {
        "units": decode_attr(dataset.attrs.get("units")),
        "long_name": decode_attr(
            dataset.attrs.get("long_name"),
            decode_attr(dataset.attrs.get("standard_name"), output_name),
        ),
        "source_variable": source_name,
    }
    mean = np.full(accumulator.shape, np.nan, dtype=np.float32)
    np.divide(accumulator, valid_count, out=mean, where=valid_count > 0)
    attrs["missing_value_handling"] = "exclude _FillValue and values outside valid range"
    return mean, attrs


def as_data_array(data: np.ndarray, name: str) -> xr.DataArray:
    dims = FINE_DIMS[name]
    if data.ndim != len(dims):
        raise ValueError(f"{name} shape {data.shape} does not match dims {dims}")
    coords = {dim: np.arange(size, dtype=np.int64) for dim, size in zip(dims, data.shape)}
    return xr.DataArray(data, dims=dims, coords=coords, name=name)


def read_static_grid(sample_path: Path) -> xr.Dataset:
    """Construct the repo's staggered-grid dataset from LLC4320 HDF5 fields."""
    names = (
        "XC", "YC", "XG", "YG", "DXV", "DYU", "DXC", "DYC", "DXG",
        "DYG", "RAZ", "Depth", "Z", "Zp1",
    )
    with h5py.File(sample_path, "r") as h5:
        static = {name: np.asarray(h5[name][...], dtype=np.float32) for name in names}

    rf = static["Zp1"]
    rc = static["Z"]
    drf = np.abs(np.diff(rf)).astype(np.float32)
    drc = np.empty(rf.size, dtype=np.float32)
    drc[0] = drf[0]
    drc[-1] = drf[-1]
    drc[1:-1] = np.abs(np.diff(rc))

    depth = static["Depth"]
    upper = np.abs(rf[:-1])[:, None, None]
    lower = np.abs(rf[1:])[:, None, None]
    hfac_c = np.clip((depth[None] - upper) / (lower - upper), 0.0, 1.0)
    hfac_c = np.nan_to_num(hfac_c, nan=0.0).astype(np.float32)
    hfac_w = np.minimum(hfac_c, np.roll(hfac_c, 1, axis=2))
    hfac_s = np.minimum(
        hfac_c,
        np.pad(hfac_c[:, :-1], ((0, 0), (1, 0), (0, 0))),
    )

    rac = (static["DXV"] * static["DYU"]).astype(np.float32)
    raw = (static["DXC"] * static["DYG"]).astype(np.float32)
    ras = (static["DXG"] * static["DYC"]).astype(np.float32)
    nj, ni = depth.shape

    return xr.Dataset(
        {
            "XC": (("j", "i"), static["XC"]),
            "YC": (("j", "i"), static["YC"]),
            "XG": (("j_g", "i_g"), static["XG"]),
            "YG": (("j_g", "i_g"), static["YG"]),
            "DXC": (("j", "i_g"), static["DXC"]),
            "DYC": (("j_g", "i"), static["DYC"]),
            "DXG": (("j_g", "i"), static["DXG"]),
            "DYG": (("j", "i_g"), static["DYG"]),
            "RAC": (("j", "i"), rac),
            "RAW": (("j", "i_g"), raw),
            "RAS": (("j_g", "i"), ras),
            "RAZ": (("j_g", "i_g"), static["RAZ"]),
            "DRF": (("k",), drf),
            "DRC": (("k_p1",), drc),
            "RF": (("k_p1",), rf),
            "RC": (("k",), rc),
            "hFacC": (("k", "j", "i"), hfac_c),
            "hFacW": (("k", "j", "i_g"), hfac_w),
            "hFacS": (("k", "j_g", "i"), hfac_s),
            "k_l": (("k_l",), np.arange(rc.size, dtype=np.int64)),
        },
        coords={
            "k": np.arange(rc.size),
            "k_p1": np.arange(rf.size),
            "j": np.arange(nj),
            "j_g": np.arange(nj),
            "i": np.arange(ni),
            "i_g": np.arange(ni),
        },
    )


def infer_coarse_box(ds_fine: xr.Dataset, target_km: float) -> tuple[int, int]:
    if target_km <= 0:
        raise ValueError("--target-km must be positive")
    mean_dy = float(ds_fine["DYG"].where(ds_fine["DYG"] > 0).mean())
    mean_dx = float(ds_fine["DXG"].where(ds_fine["DXG"] > 0).mean())
    return (
        max(1, int(round(target_km * 1000.0 / mean_dy))),
        max(1, int(round(target_km * 1000.0 / mean_dx))),
    )


def spatial_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 3:
        return (min(shape[0], 8), min(shape[1], 100), min(shape[2], 120))
    return tuple(min(size, 200) for size in shape)


def initialize_output(
    output_path: Path,
    fields: dict[str, xr.DataArray],
    *,
    coarse: bool,
    source_dir: Path,
    first_file: DailyFile,
    last_file: DailyFile,
    hours_per_mean: int,
    box_j: int | None,
    box_i: int | None,
    compression_level: int,
    to_wm2: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5netcdf.File(output_path, "w") as nc:
        nc.dimensions["time"] = None
        for field in fields.values():
            for dim, size in field.sizes.items():
                if dim not in nc.dimensions:
                    nc.dimensions[dim] = int(size)

        nc.attrs["source_data_dir"] = str(source_dir)
        nc.attrs["source_first_file"] = first_file.path.name
        nc.attrs["source_last_file"] = last_file.path.name
        nc.attrs["hours_per_mean"] = int(hours_per_mean)
        nc.attrs["time_encoding"] = "YYYYMMDDHH"
        nc.attrs["processing"] = "serial h5py mean; weighted repo coarsening"
        nc.attrs["coarse_output"] = int(coarse)
        if coarse:
            nc.attrs["box_j"] = int(box_j)
            nc.attrs["box_i"] = int(box_i)
            nc.attrs["wT_units"] = "W m-2" if to_wm2 else "degree_C m s-1"

        for name in ("time_start", "time_end"):
            variable = nc.create_variable(
                name, dimensions=("time",), dtype="i8", chunks=(1,), maxshape=(None,)
            )
            variable.attrs["units"] = "YYYYMMDDHH"

        for name, field in fields.items():
            spatial_shape = tuple(int(field.sizes[dim]) for dim in field.dims)
            kwargs = {}
            if compression_level > 0:
                kwargs = {
                    "compression": "gzip",
                    "compression_opts": int(compression_level),
                    "shuffle": True,
                }
            variable = nc.create_variable(
                name,
                dimensions=("time",) + tuple(field.dims),
                dtype="f4",
                chunks=(1,) + spatial_chunks(spatial_shape),
                maxshape=(None,) + spatial_shape,
                **kwargs,
            )
            for key, value in field.attrs.items():
                if value not in (None, ""):
                    variable.attrs[key] = value


def repair_and_saved_times(output_path: Path, expected_names: tuple[str, ...]) -> set[int]:
    """Trim a partial last append and return completed block starts."""
    if not output_path.exists():
        return set()
    with h5netcdf.File(output_path, "a") as nc:
        if "time_start" not in nc.variables:
            raise ValueError(f"{output_path} is missing time_start")
        completed = int(nc.variables["time_start"].shape[0])
        for name in ("time_end",) + expected_names:
            if name not in nc.variables:
                raise ValueError(f"{output_path} is missing variable {name}")
            variable = nc.variables[name]
            if variable.shape[0] < completed:
                raise ValueError(f"{output_path}: {name} is shorter than time_start")
            if variable.shape[0] > completed:
                variable._h5ds.resize((completed,) + variable.shape[1:])
        return set(map(int, nc.variables["time_start"][:].tolist()))


def append_block(
    output_path: Path,
    fields: dict[str, xr.DataArray],
    time_start: int,
    time_end: int,
) -> None:
    """Append fields first and write time_start last as the completion marker."""
    with h5netcdf.File(output_path, "a") as nc:
        write_index = int(nc.variables["time_start"].shape[0])
        for name, field in fields.items():
            variable = nc.variables[name]
            variable._h5ds.resize((write_index + 1,) + variable.shape[1:])
            variable[write_index] = field.values.astype(np.float32, copy=False)

        end_variable = nc.variables["time_end"]
        end_variable._h5ds.resize((write_index + 1,))
        end_variable[write_index] = np.int64(time_end)

        start_variable = nc.variables["time_start"]
        start_variable._h5ds.resize((write_index + 1,))
        start_variable[write_index] = np.int64(time_start)
        nc.flush()


def compute_coarse_fields(
    fine_fields: dict[str, xr.DataArray],
    ds_fine: xr.Dataset,
    ds_coarse: xr.Dataset,
    box_j: int,
    box_i: int,
    boundary_z: str,
    to_wm2: bool,
) -> dict[str, xr.DataArray]:
    coarse = {}
    for name in VARIABLES:
        print(f"    coarsening {name}", flush=True)
        field = coarsen_UVWTS.coarse_field(
            COARSEN_NAMES[name], fine_fields[name], ds_fine, box_j, box_i
        )
        field = field.astype("float32").rename(name)
        field.attrs.update(fine_fields[name].attrs)
        coarse[name] = field

    print("    calculating wT_submeso", flush=True)
    wt = submeso_vertical_heat_flux(
        ds_fine,
        ds_coarse,
        xr.Dataset({"W": fine_fields["W"], "T": fine_fields["T"]}),
        xr.Dataset({"W": coarse["W"], "T": coarse["T"]}),
        boundary_Z=boundary_z,
        box_j=box_j,
        box_i=box_i,
        to_wm2=to_wm2,
        output="submeso_center",
    )
    coarse["wT_submeso"] = wt.astype("float32").rename("wT_submeso")
    return coarse


def validate_source_file(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        missing = [SOURCE_NAMES[name] for name in VARIABLES if SOURCE_NAMES[name] not in h5]
        if missing:
            raise KeyError(f"{path} is missing datasets: {missing}")
        for name in VARIABLES:
            source = SOURCE_NAMES[name]
            if h5[source].shape[0] != 24:
                raise ValueError(f"{path}: {source} has {h5[source].shape[0]} times, expected 24")


def main() -> None:
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    fine_path = output_dir / args.fine_name
    coarse_path = output_dir / args.coarse_name

    selected = select_consecutive_files(
        list_daily_files(data_dir), args.n_files, args.start_date
    )
    validate_source_file(selected[0].path)
    blocks = make_blocks(selected, args.hours_per_mean)
    if args.max_new_blocks is not None and args.max_new_blocks <= 0:
        raise ValueError("--max-new-blocks must be positive")
    print(
        f"Selected {len(selected)} consecutive daily files "
        f"{selected[0].day:%Y%m%d}-{selected[-1].day:%Y%m%d}.",
        flush=True,
    )
    print(f"Available {args.hours_per_mean}-hour blocks: {len(blocks)}", flush=True)
    if args.max_new_blocks is not None:
        print(
            f"This run will process at most {args.max_new_blocks} unfinished blocks.",
            flush=True,
        )
    print("Execution mode: serial Python (no MPI, no Dask)", flush=True)

    ds_fine = ds_coarse = None
    box_j = box_i = None
    if args.stage in ("all", "coarse"):
        print("Reading static grid from the first LLC4320 file...", flush=True)
        ds_fine = read_static_grid(selected[0].path)
        inferred_j, inferred_i = infer_coarse_box(ds_fine, args.target_km)
        box_j = int(args.box_j if args.box_j is not None else inferred_j)
        box_i = int(args.box_i if args.box_i is not None else inferred_i)
        print(f"Coarse target {args.target_km:g} km: box_j={box_j}, box_i={box_i}", flush=True)
        ds_coarse = build_coarse_grid.build_coarse_grid_staggered(
            ds_fine, box_j=box_j, box_i=box_i
        )

    fine_saved = (
        repair_and_saved_times(fine_path, VARIABLES)
        if args.stage in ("all", "fine") and fine_path.exists()
        else set()
    )
    coarse_names = VARIABLES + ("wT_submeso",)
    coarse_saved = (
        repair_and_saved_times(coarse_path, coarse_names)
        if args.stage in ("all", "coarse") and coarse_path.exists()
        else set()
    )

    completed_this_run = 0
    for block_number, block in enumerate(blocks, start=1):
        need_fine = args.stage in ("all", "fine") and block.time_start not in fine_saved
        need_coarse = args.stage in ("all", "coarse") and block.time_start not in coarse_saved
        if not need_fine and not need_coarse:
            print(f"[{block_number}/{len(blocks)}] skip saved {block.time_start}", flush=True)
            continue
        if args.max_new_blocks is not None and completed_this_run >= args.max_new_blocks:
            print(
                f"Reached the limit of {args.max_new_blocks} new blocks for this run.",
                flush=True,
            )
            break

        print(
            f"[{block_number}/{len(blocks)}] {block.source.path.name} "
            f"hours {block.start_hour:02d}-{block.stop_hour:02d}",
            flush=True,
        )
        fine_fields = {}
        with h5py.File(block.source.path, "r") as h5:
            for name in VARIABLES:
                print(f"    reading {name} ({SOURCE_NAMES[name]})", flush=True)
                mean, attrs = read_hdf5_time_mean(
                    h5, SOURCE_NAMES[name], block.start_hour, block.stop_hour, name
                )
                field = as_data_array(mean, name)
                field.attrs.update(attrs)
                field.attrs["mean_hours"] = int(args.hours_per_mean)
                fine_fields[name] = field

        if need_fine:
            if not fine_path.exists():
                initialize_output(
                    fine_path,
                    fine_fields,
                    coarse=False,
                    source_dir=data_dir,
                    first_file=selected[0],
                    last_file=selected[-1],
                    hours_per_mean=args.hours_per_mean,
                    box_j=None,
                    box_i=None,
                    compression_level=args.compression_level,
                    to_wm2=args.to_wm2,
                )
            print(f"    appending fine block to {fine_path}", flush=True)
            append_block(fine_path, fine_fields, block.time_start, block.time_end)
            fine_saved.add(block.time_start)

        if need_coarse:
            if ds_fine is None or ds_coarse is None or box_j is None or box_i is None:
                raise RuntimeError("Coarse grid was not initialized")
            coarse_fields = compute_coarse_fields(
                fine_fields,
                ds_fine,
                ds_coarse,
                box_j,
                box_i,
                args.boundary_z,
                args.to_wm2,
            )
            if not coarse_path.exists():
                initialize_output(
                    coarse_path,
                    coarse_fields,
                    coarse=True,
                    source_dir=data_dir,
                    first_file=selected[0],
                    last_file=selected[-1],
                    hours_per_mean=args.hours_per_mean,
                    box_j=box_j,
                    box_i=box_i,
                    compression_level=args.compression_level,
                    to_wm2=args.to_wm2,
                )
            print(f"    appending coarse block to {coarse_path}", flush=True)
            append_block(coarse_path, coarse_fields, block.time_start, block.time_end)
            coarse_saved.add(block.time_start)

        print(f"    completed block {block.time_start}-{block.time_end}", flush=True)
        completed_this_run += 1

    print(f"New blocks completed in this run: {completed_this_run}", flush=True)

    if args.stage in ("all", "fine"):
        print(f"Fine output: {fine_path}", flush=True)
    if args.stage in ("all", "coarse"):
        print(f"Coarse output: {coarse_path}", flush=True)


if __name__ == "__main__":
    main()
