#!/usr/bin/env python3
"""Calculate submesoscale heat flux from 12-snapshot mean W and T fields."""

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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "twelve_snapshot_means" / "wT_submeso_12snap_mean.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use 12-snapshot mean W/T to compute submesoscale vertical heat flux."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument("--boundary-z", default="fill")
    parser.add_argument("--to-wm2", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_mean_files(input_dir: Path) -> tuple[xr.Dataset, xr.Dataset]:
    w_path = input_dir / "W_12snap_mean.nc"
    t_path = input_dir / "T_12snap_mean.nc"
    if not w_path.exists():
        raise FileNotFoundError(f"Missing {w_path}")
    if not t_path.exists():
        raise FileNotFoundError(f"Missing {t_path}")
    return (
        xr.open_dataset(w_path, decode_times=False),
        xr.open_dataset(t_path, decode_times=False),
    )


def common_time_indices(ds_w: xr.Dataset, ds_t: xr.Dataset) -> np.ndarray:
    starts_w = ds_w["time_start"].values.astype(np.int64)
    starts_t = ds_t["time_start"].values.astype(np.int64)
    common = np.intersect1d(starts_w, starts_t)
    if common.size == 0:
        raise ValueError("No common 12-snapshot blocks between W and T files.")
    return common


def select_block(ds: xr.Dataset, start: int) -> xr.Dataset:
    matches = np.where(ds["time_start"].values.astype(np.int64) == int(start))[0]
    if matches.size != 1:
        raise ValueError(f"Expected exactly one block with time_start={start}, got {matches.size}")
    return ds.isel(time=int(matches[0]))


def load_existing(output_path: Path) -> tuple[list[xr.DataArray], list[int], list[int]]:
    if not output_path.exists():
        return [], [], []
    with xr.open_dataset(output_path, decode_times=False) as existing:
        existing = existing.load()
    if "wT_sub_kC" not in existing:
        raise KeyError(f"{output_path} exists but does not contain 'wT_sub_kC'")
    starts = [int(value) for value in existing["time_start"].values]
    ends = [int(value) for value in existing["time_end"].values]
    fields = [existing["wT_sub_kC"].isel(time=index, drop=True) for index in range(existing.sizes["time"])]
    return fields, starts, ends


def dataset_from_fields(fields: list[xr.DataArray], starts: list[int], ends: list[int], run_dir: Path) -> xr.Dataset:
    order = np.argsort(np.asarray(starts, dtype=np.int64))
    fields = [fields[int(index)] for index in order]
    starts = [starts[int(index)] for index in order]
    ends = [ends[int(index)] for index in order]
    centers = [0.5 * (start + end) for start, end in zip(starts, ends)]
    time = xr.DataArray(np.asarray(centers, dtype=np.float64), dims="time", name="time")
    data = xr.concat(fields, dim=time).rename("wT_sub_kC")
    data.attrs["long_name"] = "submesoscale vertical heat flux from 12-snapshot mean W and T"
    return xr.Dataset(
        {"wT_sub_kC": data},
        coords={
            "time_start": ("time", np.asarray(starts, dtype=np.int64)),
            "time_end": ("time", np.asarray(ends, dtype=np.int64)),
        },
        attrs={
            "source_run_dir": str(run_dir),
            "definition": "submeso_vertical_heat_flux applied to 12-snapshot mean W and T",
        },
    )


def write_dataset_atomic(ds: xr.Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ds.to_netcdf(tmp_path)
    os.replace(tmp_path, output_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    ds_w, ds_t = load_mean_files(input_dir)
    ds_fine = read_mitgcm.read_mitgrid_staggered(run_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=args.box_j, box_i=args.box_i)

    fields, starts, ends = load_existing(output_path)
    saved = set(starts)
    common_starts = common_time_indices(ds_w, ds_t)

    for block_number, start in enumerate(common_starts, start=1):
        start = int(start)
        if start in saved:
            print(f"skip existing block {block_number}/{len(common_starts)} time_start={start}", flush=True)
            continue

        w_block = select_block(ds_w, start)
        t_block = select_block(ds_t, start)
        end = int(w_block["time_end"].item())
        if end != int(t_block["time_end"].item()):
            raise ValueError(f"W/T time_end mismatch for block starting {start}")

        print(f"calculate block {block_number}/{len(common_starts)} {start}-{end}", flush=True)
        w = w_block["W"].drop_vars(["time_start", "time_end"], errors="ignore")
        t = t_block["T"].drop_vars(["time_start", "time_end"], errors="ignore")
        w_coarse = coarsen_UVWTS.coarse_field("W", w, ds_fine, args.box_j, args.box_i)
        t_coarse = coarsen_UVWTS.coarse_field("T", t, ds_fine, args.box_j, args.box_i)

        _, wT_sub_kC = submeso_vertical_heat_flux(
            ds_fine,
            ds_coarse,
            xr.Dataset({"W": w, "T": t}),
            xr.Dataset({"W": w_coarse, "T": t_coarse}),
            boundary_Z=args.boundary_z,
            box_j=args.box_j,
            box_i=args.box_i,
            to_wm2=args.to_wm2,
        )

        fields.append(wT_sub_kC.astype("float32"))
        starts.append(start)
        ends.append(end)
        saved.add(start)

        out = dataset_from_fields(fields, starts, ends, run_dir)
        write_dataset_atomic(out, output_path)
        print(f"saved block {block_number}/{len(common_starts)} to {output_path}", flush=True)

    ds_w.close()
    ds_t.close()
    print(f"done: {len(fields)} blocks saved in {output_path}", flush=True)


if __name__ == "__main__":
    main()
