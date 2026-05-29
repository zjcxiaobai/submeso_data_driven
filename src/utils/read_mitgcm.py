"""MITgcm binary readers with explicit xarray dimension conventions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from MITgcmutils import mds


Dims = Sequence[str]


def _mds_path(grid_dir: str | Path, name: str) -> str:
    return str(Path(grid_dir) / name)


def _as_data_array(data: np.ndarray, dims: Dims, name: str) -> xr.DataArray:
    dims = tuple(dims)
    arr = np.asarray(data)
    if arr.ndim != len(dims):
        raise ValueError(
            f"{name} has shape {arr.shape}, but dims {dims} imply {len(dims)} dimensions"
        )
    return xr.DataArray(arr, dims=dims, name=name)


# function to read grid variables
def _grid_var(grid_dir: str | Path, name: str, dims: Dims) -> xr.DataArray:
    """Read one MITgcm grid variable and attach explicit dimension names."""
    return _as_data_array(mds.rdmds(_mds_path(grid_dir, name)), dims, name)


# function to read ocean variables
def _ocean_var(grid_dir: str | Path, name: str, t: int, dims: Dims) -> xr.DataArray:
    """Read one MITgcm time-dependent variable and attach explicit dimension names."""
    return _as_data_array(mds.rdmds(_mds_path(grid_dir, name), int(t)), dims, name)


def _vertical_grid_var(grid_dir: str | Path, name: str, dim: str) -> xr.DataArray:
    """
    Read a vertical grid vector.

    MITgcm often writes vertical metrics as (k, 1, 1) / (k_p1, 1, 1), but some
    workflows store them as true 1D vectors. Normalize both cases to one dim.
    """
    arr = np.asarray(mds.rdmds(_mds_path(grid_dir, name)))
    if arr.ndim == 1:
        data = arr
    elif arr.ndim == 3 and arr.shape[1:] == (1, 1):
        data = arr[:, 0, 0]
    else:
        raise ValueError(
            f"{name} is expected to be 1D or (n, 1, 1); got shape {arr.shape}"
        )
    return xr.DataArray(data, dims=(dim,), name=name)


def _assign_ordinal_coords(ds: xr.Dataset) -> xr.Dataset:
    """Attach simple ordinal coordinates for every MITgcm staggered dimension."""
    coords = {dim: np.arange(size, dtype="int64") for dim, size in ds.sizes.items()}
    return ds.assign_coords(coords)


# function to read all the grid variables and create a dataset
def read_mitgrid_staggered(grid_dir: str | Path) -> xr.Dataset:
    """
    Read MITgcm grid variables using this repo's C-grid dimension convention.

    Dimension convention
    --------------------
    Centers:      ``(j, i)``
    U faces:      ``(j, i_g)``
    V faces:      ``(j_g, i)``
    Corners:      ``(j_g, i_g)``
    Cell centers: ``(k, j, i)`` for 3D masks

    Returns
    -------
    xr.Dataset
        Grid metrics, coordinates, vertical vectors, and wet masks with ordinal
        coordinates attached to all dimensions.
    """
    # Centers (j, i)
    XC = _grid_var(grid_dir, "XC", ("j", "i"))
    YC = _grid_var(grid_dir, "YC", ("j", "i"))
    DXF = _grid_var(grid_dir, "DXF", ("j", "i"))
    DYF = _grid_var(grid_dir, "DYF", ("j", "i"))
    RAC = _grid_var(grid_dir, "RAC", ("j", "i"))
    DXV = _grid_var(grid_dir, "DXV", ("j", "i"))
    DYU = _grid_var(grid_dir, "DYU", ("j", "i"))

    # Staggered faces
    DXC = _grid_var(grid_dir, "DXC", ("j", "i_g"))
    DYC = _grid_var(grid_dir, "DYC", ("j_g", "i"))
    RAW = _grid_var(grid_dir, "RAW", ("j", "i_g"))
    RAS = _grid_var(grid_dir, "RAS", ("j_g", "i"))
    DXG = _grid_var(grid_dir, "DXG", ("j_g", "i"))
    DYG = _grid_var(grid_dir, "DYG", ("j", "i_g"))

    # Corners / zeta points
    RAZ = _grid_var(grid_dir, "RAZ", ("j_g", "i_g"))
    XG = _grid_var(grid_dir, "XG", ("j_g", "i_g"))
    YG = _grid_var(grid_dir, "YG", ("j_g", "i_g"))

    # Vertical coordinates and metrics
    DRF = _vertical_grid_var(grid_dir, "DRF", "k")
    DRC = _vertical_grid_var(grid_dir, "DRC", "k_p1")
    RF = _vertical_grid_var(grid_dir, "RF", "k_p1")
    RC = _vertical_grid_var(grid_dir, "RC", "k")

    # Wet masks
    hFacC = _grid_var(grid_dir, "hFacC", ("k", "j", "i"))
    hFacW = _grid_var(grid_dir, "hFacW", ("k", "j", "i_g"))
    hFacS = _grid_var(grid_dir, "hFacS", ("k", "j_g", "i"))

    if DRF.sizes["k"] != hFacC.sizes["k"] or RC.sizes["k"] != hFacC.sizes["k"]:
        raise ValueError(
            "Vertical center dimensions disagree between DRF/RC and hFacC: "
            f"DRF={DRF.sizes['k']}, RC={RC.sizes['k']}, hFacC={hFacC.sizes['k']}"
        )
    if DRC.sizes["k_p1"] != RF.sizes["k_p1"]:
        raise ValueError(
            f"Vertical interface dimensions disagree: DRC={DRC.sizes['k_p1']}, "
            f"RF={RF.sizes['k_p1']}"
        )

    # MITgcm W points are read with dimension k_l in this project. Its length
    # matches the number of cell centers for the current vertical velocity files.
    k_l = xr.DataArray(np.arange(hFacC.sizes["k"], dtype="int64"), dims="k_l", name="k_l")

    ds = xr.Dataset(
        dict(
            XC=XC,
            YC=YC,
            DXF=DXF,
            DYF=DYF,
            RAC=RAC,
            DXV=DXV,
            DYU=DYU,
            DXC=DXC,
            DYC=DYC,
            RAW=RAW,
            RAS=RAS,
            DXG=DXG,
            DYG=DYG,
            RAZ=RAZ,
            XG=XG,
            YG=YG,
            DRF=DRF,
            DRC=DRC,
            RF=RF,
            RC=RC,
            hFacC=hFacC,
            hFacW=hFacW,
            hFacS=hFacS,
            k_l=k_l,
        )
    )

    return _assign_ordinal_coords(ds)
