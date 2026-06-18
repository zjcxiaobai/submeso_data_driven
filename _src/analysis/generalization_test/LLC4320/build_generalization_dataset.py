#!/usr/bin/env python3
"""Build SubmesoFluxNet X/y arrays from LLC4320 coarse 12-hour means."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import build_coarse_grid, make_grid  # noqa: E402
from src.utils.mld_avg import mixed_layer_average  # noqa: E402

from calculate_coarse_velocity_diagnostics import normalize_staggered_sizes  # noqa: E402
from save_12hour_means_and_coarse_llc4320 import read_static_grid  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIELDS = (
    SCRIPT_DIR
    / "twelve_hour_means_llc4320"
    / "LLC4320_12hour_mean_coarse_50km.nc"
)
DEFAULT_DIAGNOSTICS = (
    SCRIPT_DIR
    / "coarse_velocity_diagnostics"
    / "LLC4320_12hour_coarse_vorticity_strain_speed.nc"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data"

# Exact channel order used by data/X.npy in the no-shuffle overfitting test.
FEATURE_NAMES = (
    "U",
    "V",
    "W",
    "vorticity",
    "strain",
    "T",
    "speed",
    "kpphbl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create LLC4320 mixed-layer-averaged X/y arrays in the exact "
            "eight-channel SubmesoFluxNet input order."
        )
    )
    parser.add_argument("--fields", type=Path, default=DEFAULT_FIELDS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-nan",
        action="store_true",
        help="Save arrays even if NaN or infinite values remain.",
    )
    return parser.parse_args()


def time_lookup(ds: xr.Dataset) -> dict[int, int]:
    values = ds["time_start"].values.astype(np.int64)
    if np.unique(values).size != values.size:
        raise ValueError("time_start contains duplicate timestamps")
    return {int(value): index for index, value in enumerate(values)}


def common_times(fields: xr.Dataset, diagnostics: xr.Dataset) -> np.ndarray:
    times = np.intersect1d(
        fields["time_start"].values.astype(np.int64),
        diagnostics["time_start"].values.astype(np.int64),
    )
    if times.size == 0:
        raise ValueError("No common timestamps between fields and diagnostics")
    return np.sort(times)


def build_matching_coarse_grid(fields: xr.Dataset) -> xr.Dataset:
    source_dir = Path(str(fields.attrs["source_data_dir"]))
    source_file = source_dir / str(fields.attrs["source_first_file"])
    box_j = int(fields.attrs["box_j"])
    box_i = int(fields.attrs["box_i"])

    fine_grid = read_static_grid(source_file)
    coarse_grid = build_coarse_grid.build_coarse_grid_staggered(
        fine_grid,
        box_j=box_j,
        box_i=box_i,
    )
    return normalize_staggered_sizes(
        coarse_grid,
        center_j=int(fields.sizes["j"]),
        center_i=int(fields.sizes["i"]),
    )


def center_velocity_fields(
    fields: xr.Dataset,
    time_index: int,
    coarse_grid: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    grid = make_grid.make_grid_periodic_x(coarse_grid)
    u = fields["U"].isel(
        time=time_index,
        i_g=slice(0, fields.sizes["i"]),
        drop=True,
    ).load()
    v = fields["V"].isel(
        time=time_index,
        j_g=slice(0, fields.sizes["j"]),
        drop=True,
    ).load()
    w = fields["W"].isel(time=time_index, drop=True).load()

    u_center = grid.interp(u, "X", to="center").rename("U")
    v_center = grid.interp(v, "Y", to="center", boundary="fill").rename("V")
    w_center = grid.interp(w, "Z", to="center", boundary="fill").rename("W")
    return u_center, v_center, w_center


def mld_mean(
    field: xr.DataArray,
    kpphbl: xr.DataArray,
    coarse_grid: xr.Dataset,
    name: str,
) -> np.ndarray:
    averaged = mixed_layer_average(
        field,
        kpphbl,
        coarse_grid,
        output_name=name,
    )
    return averaged.values.astype(np.float32, copy=False)


def build_sample(
    fields: xr.Dataset,
    diagnostics: xr.Dataset,
    field_index: int,
    diagnostic_index: int,
    coarse_grid: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    kpphbl = fields["kpphbl"].isel(time=field_index, drop=True).load()
    u, v, w = center_velocity_fields(fields, field_index, coarse_grid)

    feature_fields = {
        "U": u,
        "V": v,
        "W": w,
        "vorticity": diagnostics["vorticity"]
        .isel(time=diagnostic_index, drop=True)
        .load(),
        "strain": diagnostics["strain"].isel(time=diagnostic_index, drop=True).load(),
        "T": fields["T"].isel(time=field_index, drop=True).load(),
        "speed": diagnostics["speed"].isel(time=diagnostic_index, drop=True).load(),
    }

    features = [
        mld_mean(feature_fields[name], kpphbl, coarse_grid, name)
        for name in FEATURE_NAMES[:-1]
    ]
    features.append(kpphbl.values.astype(np.float32, copy=False))

    heat_flux = fields["wT_submeso"].isel(time=field_index, drop=True).load()
    target = mld_mean(heat_flux, kpphbl, coarse_grid, "wT_submeso")

    x = np.stack(features, axis=0).astype(np.float32, copy=False)
    y = target[np.newaxis, :, :].astype(np.float32, copy=False)
    return x, y


def save_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    fields_path = args.fields.expanduser().resolve()
    diagnostics_path = args.diagnostics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    with (
        xr.open_dataset(fields_path, engine="h5netcdf", decode_times=False) as fields,
        xr.open_dataset(
            diagnostics_path,
            engine="h5netcdf",
            decode_times=False,
        ) as diagnostics,
    ):
        required_fields = {"time_start", "U", "V", "W", "T", "kpphbl", "wT_submeso"}
        required_diagnostics = {"time_start", "vorticity", "strain", "speed"}
        missing_fields = sorted(required_fields - set(fields.variables))
        missing_diagnostics = sorted(
            required_diagnostics - set(diagnostics.variables)
        )
        if missing_fields:
            raise KeyError(f"{fields_path} is missing variables: {missing_fields}")
        if missing_diagnostics:
            raise KeyError(
                f"{diagnostics_path} is missing variables: {missing_diagnostics}"
            )

        times = common_times(fields, diagnostics)
        fields_lookup = time_lookup(fields)
        diagnostics_lookup = time_lookup(diagnostics)
        coarse_grid = build_matching_coarse_grid(fields)

        x_samples = []
        y_samples = []
        for number, timestamp in enumerate(times, start=1):
            timestamp = int(timestamp)
            print(
                f"[{number}/{len(times)}] mixed-layer averages for {timestamp}",
                flush=True,
            )
            x, y = build_sample(
                fields,
                diagnostics,
                fields_lookup[timestamp],
                diagnostics_lookup[timestamp],
                coarse_grid,
            )
            x_samples.append(x)
            y_samples.append(y)

    X = np.stack(x_samples, axis=0).astype(np.float32, copy=False)
    y = np.stack(y_samples, axis=0).astype(np.float32, copy=False)

    finite_x = np.isfinite(X)
    finite_y = np.isfinite(y)
    print(f"Feature order: {list(FEATURE_NAMES)}", flush=True)
    print(f"X shape: {X.shape}; y shape: {y.shape}", flush=True)
    print(
        f"Non-finite values: X={X.size - int(finite_x.sum())}, "
        f"y={y.size - int(finite_y.sum())}",
        flush=True,
    )
    if not args.allow_nan and (not finite_x.all() or not finite_y.all()):
        raise ValueError(
            "Non-finite values remain in X or y. Inspect wet masks before "
            "using --allow-nan."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_atomic(output_dir / "X.npy", X)
    save_atomic(output_dir / "y.npy", y)
    save_atomic(output_dir / "time_index.npy", times.astype(np.int64))
    save_atomic(
        output_dir / "feature_names.npy",
        np.asarray(FEATURE_NAMES, dtype="<U16"),
    )
    print(f"Saved final generalization dataset to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
