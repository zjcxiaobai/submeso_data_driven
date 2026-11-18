"""Convert the coarse NetCDF fields into the NPZ files used by the model."""

from pathlib import Path
import sys

import gsw
import numpy as np
import xarray as xr


# Make ``submeso_data_driven.utils`` importable when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT.parent))

from submeso_data_driven.utils import (  # noqa: E402
    build_coarse_grid,
    make_grid,
    mld_avg,
    read_mitgcm,
    strain,
    vorticity,
)


# -----------------------------------------------------------------------------
# Settings for this project
# -----------------------------------------------------------------------------
GRID_DIR = Path(
    "/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"
)

SOURCE_DIR = REPO_ROOT / "data" / "interim" / "temporal_spatial_avg"

OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "feature_target_4D_coarse"
CROP = {"j": slice(62, 118), "i": slice(2, 58)}
BATCH_SIZE = 8
OVERWRITE = False

FILES = {
    "U": "U_12hmean_coarse.nc",
    "V": "V_12hmean_coarse.nc",
    "W": "W_12hmean_coarse.nc",
    "T": "T_12hmean_coarse.nc",
    "S": "S_12hmean_coarse.nc",
    "Eta": "Eta_12hmean_coarse.nc",
    "H": "kpphbl_12hmean_coarse.nc",
    "wT": "wT_sub_12h_blocks_true.nc",
}

NETCDF_VARIABLES = {
    "U": "U",
    "V": "V",
    "W": "W",
    "T": "T",
    "S": "S",
    "Eta": "Eta",
    "H": "kpphbl",
    "wT": "wT_sub_kC",
}

# Each output lists the NetCDF fields needed to calculate it.
OUTPUTS = {
    "U": ("U", "H"),
    "V": ("V", "H"),
    "W": ("W", "H"),
    "zeta": ("U", "V", "H"),
    "strain": ("U", "V", "H"),
    "T": ("T", "H"),
    "rho": ("T", "S", "Eta", "H"),
    "kpphbl": ("H",),
    "wT": ("wT", "H"),
}


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an existing output so an interrupted conversion can resume."""
    if OVERWRITE or not path.exists():
        return np.array([], dtype=np.int64), None

    with np.load(path) as saved:
        return saved["time_index"].astype(np.int64), saved["data"].astype(np.float32)


def append_and_save(
    path: Path,
    old_times: np.ndarray,
    old_data: np.ndarray | None,
    new_times: np.ndarray,
    new_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Append new snapshots, sort by time, and save one NPZ file."""
    if old_data is None:
        times = new_times
        data = new_data
    else:
        times = np.concatenate((old_times, new_times))
        data = np.concatenate((old_data, new_data), axis=0)

    order = np.argsort(times)
    times = times[order]
    data = data[order]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, time_index=times, data=data.astype(np.float32))
    return times, data


def density(
    temperature: xr.DataArray,
    salinity: xr.DataArray,
    eta: xr.DataArray,
    grid: xr.Dataset,
) -> xr.DataArray:
    """Calculate in-situ density from potential temperature and salinity."""
    z = grid["RC"] - eta
    latitude = xr.full_like(temperature.isel(k=0), -55.0)
    longitude = xr.full_like(temperature.isel(k=0), 153.0)

    pressure = xr.apply_ufunc(gsw.p_from_z, z, latitude)
    absolute_salinity = xr.apply_ufunc(
        gsw.SA_from_SP,
        salinity,
        pressure,
        longitude,
        latitude,
    )
    conservative_temperature = xr.apply_ufunc(
        gsw.CT_from_pt,
        absolute_salinity,
        temperature,
    )
    rho = xr.apply_ufunc(
        gsw.rho,
        absolute_salinity,
        conservative_temperature,
        pressure,
    )
    return rho.where(grid["hFacC"] > 0).rename("rho")


def calculate_snapshot(
    name: str,
    fields: dict[str, xr.DataArray],
    grid: xr.Dataset,
    xgcm_grid,
) -> np.ndarray:
    """Calculate one cropped two-dimensional model field."""
    if name == "kpphbl":
        result = fields["H"]
    else:
        if name == "U":
            field = xgcm_grid.interp(
                fields["U"], "X", to="center", boundary="periodic"
            )
        elif name == "V":
            field = xgcm_grid.interp(
                fields["V"], "Y", to="center", boundary="fill"
            )
        elif name == "W":
            field = xgcm_grid.interp(
                fields["W"], "Z", to="center", boundary="fill"
            )
        elif name == "zeta":
            ocean = xr.Dataset({"U": fields["U"], "V": fields["V"]})
            field = vorticity.vorticity(grid, ocean, y_boundary="fill")
        elif name == "strain":
            ocean = xr.Dataset({"U": fields["U"], "V": fields["V"]})
            field = strain.total_strain(grid, ocean, y_boundary="fill")
        elif name == "T":
            field = fields["T"]
        elif name == "rho":
            field = density(fields["T"], fields["S"], fields["Eta"], grid)
        elif name == "wT":
            field = fields["wT"]
        else:
            raise ValueError(f"Unknown output: {name}")

        result = mld_avg.mld_avg(field, fields["H"], grid)

    return result.isel(**CROP).values.astype(np.float32, copy=False)


def convert_variable(
    name: str,
    required_fields: tuple[str, ...],
    grid: xr.Dataset,
    xgcm_grid,
) -> None:
    """Create one NPZ file from the consolidated NetCDF directory."""
    output_path = OUTPUT_DIR / f"{name}.npz"
    saved_times, saved_data = load_npz(output_path)
    saved_set = set(saved_times.tolist())

    print(f"\n=== {name} ===")

    missing = [
        SOURCE_DIR / FILES[key]
        for key in required_fields
        if not (SOURCE_DIR / FILES[key]).exists()
    ]
    if missing:
        print("skipped; missing " + ", ".join(path.name for path in missing))
        return

    datasets = {
        key: xr.open_dataset(
            SOURCE_DIR / FILES[key],
            engine="h5netcdf",
            decode_times=False,
        )
        for key in required_fields
    }

    try:
        common_times = None
        for dataset in datasets.values():
            times = dataset["time"].values.astype(np.int64)
            common_times = (
                times
                if common_times is None
                else np.intersect1d(common_times, times)
            )

        todo = np.array(
            [time for time in common_times if int(time) not in saved_set],
            dtype=np.int64,
        )

        if todo.size == 0:
            print("already complete")
            return

        snapshots = []

        for start in range(0, todo.size, BATCH_SIZE):
            batch_times = todo[start : start + BATCH_SIZE]
            loaded = {
                key: dataset[NETCDF_VARIABLES[key]]
                .sel(time=batch_times)
                .load()
                for key, dataset in datasets.items()
            }

            for index in range(batch_times.size):
                fields = {
                    key: values.isel(time=index)
                    for key, values in loaded.items()
                }
                snapshots.append(
                    calculate_snapshot(name, fields, grid, xgcm_grid)
                )

            completed = min(start + BATCH_SIZE, todo.size)
            print(f"{completed}/{todo.size}")

        new_data = np.stack(snapshots)
        saved_times, saved_data = append_and_save(
            output_path,
            saved_times,
            saved_data,
            todo,
            new_data,
        )
    finally:
        for dataset in datasets.values():
            dataset.close()

    if saved_data is not None:
        print(f"saved {output_path} {saved_data.shape}")


def main() -> None:
    """Build the grid once, then create all eight inputs and the target."""
    fine_grid = read_mitgcm.read_mitgrid_staggered(str(GRID_DIR))
    coarse_grid = build_coarse_grid.build_coarse_grid_staggered(
        fine_grid,
        box_j=10,
        box_i=10,
    )
    xgcm_grid = make_grid.make_grid_periodic_x(coarse_grid)

    for name, required_fields in OUTPUTS.items():
        convert_variable(name, required_fields, coarse_grid, xgcm_grid)


if __name__ == "__main__":
    main()
