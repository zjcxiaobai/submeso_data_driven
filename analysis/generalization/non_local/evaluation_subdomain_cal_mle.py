#!/usr/bin/env python3
"""Compute tuned MLE predictions on the lower-domain evaluation region."""

import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_PARENT = PROJECT_ROOT.parent
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_PARENT))

from submeso_data_driven.analysis.fk_mle.mle_tuning.mle_parameterization import (  # noqa: E402
    compute_mle_submeso_heat_flux_one_time,
)
from submeso_data_driven.utils import build_coarse_grid, read_mitgcm  # noqa: E402
from submesofluxnet.config import load_config  # noqa: E402
from submesofluxnet.data import centers_in_block  # noqa: E402


CONFIG = PROJECT_ROOT / "config/config.yaml"
LOWER_DATA = PROJECT_ROOT / "data/processed/lower"
NC_DIR = PROJECT_ROOT / "data/interim/temporal_spatial_avg"
GRID_DIR = Path("/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run")
OUTPUT = HERE / "evaluation_subdomain_mle_prediction.npz"

C_E = 2.25895966
LF_MIN = 1374.0
CROP = (72, 92, 12, 52)


def load_samples() -> tuple[np.ndarray, np.ndarray]:
    config = load_config(CONFIG)
    offsets = [int(value) for value in config["data"]["temporal_offsets"]]
    n_samples = np.load(LOWER_DATA / "X.npy", mmap_mode="r").shape[0]
    sample_index = centers_in_block((0, n_samples), offsets)
    time_index = np.load(LOWER_DATA / "time_index.npy")[sample_index]
    return sample_index, time_index


def build_grid() -> xr.Dataset:
    fine_grid = read_mitgcm.read_mitgrid_staggered(GRID_DIR)
    return build_coarse_grid.build_coarse_grid_staggered(fine_grid, box_j=10, box_i=10)


def open_inputs() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    kpphbl = xr.open_dataset(NC_DIR / "kpphbl_12hmean_coarse.nc", engine="h5netcdf", decode_times=False)["kpphbl"]
    temperature = xr.open_dataset(NC_DIR / "T_12hmean_coarse.nc", engine="h5netcdf", decode_times=False)["T"]
    salinity = xr.open_dataset(NC_DIR / "S_12hmean_coarse.nc", engine="h5netcdf", decode_times=False)["S"]
    eta = xr.open_dataset(NC_DIR / "Eta_12hmean_coarse.nc", engine="h5netcdf", decode_times=False)["Eta"]
    return kpphbl, temperature, salinity, eta


def calculate_mle(time_index: np.ndarray) -> np.ndarray:
    grid = build_grid()
    kpphbl, temperature, salinity, eta = open_inputs()
    j0, j1, i0, i1 = CROP
    predictions = []

    for position, time in enumerate(time_index, start=1):
        print(f"{position}/{len(time_index)}: time_index={int(time)}")
        full_map = compute_mle_submeso_heat_flux_one_time(
            time_index=int(time),
            ds_grid=grid,
            kpphbl_da=kpphbl,
            t_da=temperature,
            s_da=salinity,
            eta_da=eta,
            c_e=C_E,
            lf_min=LF_MIN,
            submeso_lmax=2751.0,
            submeso_inv_tau=2.0e-6,
            f0=-1.363e-4,
            beta=1.313e-11,
        )
        predictions.append(np.asarray(full_map[j0:j1, i0:i1], dtype=np.float32))

    return np.asarray(predictions, dtype=np.float32)[:, None]


def main() -> None:
    sample_index, time_index = load_samples()
    prediction = calculate_mle(time_index)
    np.savez_compressed(
        OUTPUT,
        sample_index=sample_index,
        time_index=time_index,
        prediction=prediction,
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
