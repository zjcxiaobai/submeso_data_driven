"""Build coarse MITgcm C-grid metric datasets."""

from __future__ import annotations

import numpy as np
import xarray as xr


_REQUIRED_GRID_VARS = (
    "RAC",
    "RAW",
    "RAS",
    "RAZ",
    "DXC",
    "DYC",
    "DXG",
    "DYG",
    "XC",
    "YC",
    "XG",
    "YG",
    "DRF",
    "DRC",
    "RF",
    "RC",
    "hFacC",
    "hFacW",
    "hFacS",
    "k_l",
)


def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Fine grid is missing required variable(s): {missing}")


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dimension(s): {missing}; got dims {da.dims}")


def _validate_inputs(ds_fine: xr.Dataset, box_j: int, box_i: int) -> None:
    if not isinstance(box_j, int) or not isinstance(box_i, int) or box_j <= 0 or box_i <= 0:
        raise ValueError("box_j and box_i must be positive integers")

    _require_variables(ds_fine, _REQUIRED_GRID_VARS)
    _require_dims(ds_fine["RAC"], ("j", "i"), "RAC")
    _require_dims(ds_fine["RAW"], ("j", "i_g"), "RAW")
    _require_dims(ds_fine["RAS"], ("j_g", "i"), "RAS")
    _require_dims(ds_fine["RAZ"], ("j_g", "i_g"), "RAZ")
    _require_dims(ds_fine["DXC"], ("j", "i_g"), "DXC")
    _require_dims(ds_fine["DYC"], ("j_g", "i"), "DYC")
    _require_dims(ds_fine["DXG"], ("j_g", "i"), "DXG")
    _require_dims(ds_fine["DYG"], ("j", "i_g"), "DYG")
    _require_dims(ds_fine["XC"], ("j", "i"), "XC")
    _require_dims(ds_fine["YC"], ("j", "i"), "YC")
    _require_dims(ds_fine["XG"], ("j_g", "i_g"), "XG")
    _require_dims(ds_fine["YG"], ("j_g", "i_g"), "YG")
    _require_dims(ds_fine["DRF"], ("k",), "DRF")
    _require_dims(ds_fine["DRC"], ("k_p1",), "DRC")
    _require_dims(ds_fine["RF"], ("k_p1",), "RF")
    _require_dims(ds_fine["RC"], ("k",), "RC")
    _require_dims(ds_fine["hFacC"], ("k", "j", "i"), "hFacC")
    _require_dims(ds_fine["hFacW"], ("k", "j", "i_g"), "hFacW")
    _require_dims(ds_fine["hFacS"], ("k", "j_g", "i"), "hFacS")


def _as_ordinal(da: xr.DataArray) -> xr.DataArray:
    """Drop auxiliary coordinates and replace dimension coordinates with 0..N-1."""
    da = da.reset_coords(drop=True)
    coords = {dim: xr.DataArray(np.arange(da.sizes[dim], dtype="int64"), dims=dim) for dim in da.dims}
    return da.assign_coords(coords)


def _assign_ordinal_coords(ds: xr.Dataset) -> xr.Dataset:
    coords = {dim: np.arange(size, dtype="int64") for dim, size in ds.sizes.items()}
    return ds.assign_coords(coords)


def _safe_divide(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
    return xr.where(denominator > 0, numerator / denominator, 0.0)


def _weighted_center_mean(
    value: xr.DataArray,
    weight: xr.DataArray,
    box_j: int,
    box_i: int,
    name: str,
) -> xr.DataArray:
    numerator = (value * weight).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    denominator = weight.coarsen(j=box_j, i=box_i, boundary="trim").sum()
    return _safe_divide(numerator, denominator).rename(name)


def build_coarse_grid_staggered(ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.Dataset:
    """
    Downsample a fine MITgcm C-grid to a staggered coarse grid.

    The output preserves the repo's grid convention from ``read_mitgcm``:
    centers ``(j, i)``, U faces ``(j, i_g)``, V faces ``(j_g, i)``, and
    zeta/corner points ``(j_g, i_g)``. Areas are summed to conserve area;
    horizontal line metrics are summed along the direction covered by each
    coarse cell; masks are weighted by the corresponding wet volume or face
    area. Vertical metrics are copied unchanged.
    """
    _validate_inputs(ds_fine, box_j, box_i)

    j0, i0 = 0, 0
    DRF = ds_fine["DRF"]
    DRC = ds_fine["DRC"]
    RF = ds_fine["RF"]
    RC = ds_fine["RC"]
    k_l = ds_fine["k_l"]

    # Areas and face areas are extensive quantities on their native grids.
    RAC = ds_fine["RAC"].coarsen(j=box_j, i=box_i, boundary="trim").sum().rename("RAC")
    RAW = ds_fine["RAW"].coarsen(j=box_j, i_g=box_i, boundary="trim").sum().rename("RAW")
    RAS = ds_fine["RAS"].coarsen(j_g=box_j, i=box_i, boundary="trim").sum().rename("RAS")
    RAZ = ds_fine["RAZ"].coarsen(j_g=box_j, i_g=box_i, boundary="trim").sum().rename("RAZ")

    # Line metrics on staggered faces. The untouched staggered dimension is
    # subsampled at coarse-grid face positions after summing over the cell span.
    DXG = (
        ds_fine["DXG"]
        .coarsen(i=box_i, boundary="trim")
        .sum()
        .isel(j_g=slice(j0, None, box_j))
        .rename("DXG")
    )
    DYG = (
        ds_fine["DYG"]
        .coarsen(j=box_j, boundary="trim")
        .sum()
        .isel(i_g=slice(i0, None, box_i))
        .rename("DYG")
    )
    DYC = (
        ds_fine["DYC"]
        .coarsen(i=box_i, boundary="trim")
        .sum()
        .isel(j_g=slice(j0, None, box_j))
        .rename("DYC")
    )
    DXC = (
        ds_fine["DXC"]
        .coarsen(j=box_j, boundary="trim")
        .sum()
        .isel(i_g=slice(i0, None, box_i))
        .rename("DXC")
    )

    XC = _weighted_center_mean(ds_fine["XC"], ds_fine["RAC"], box_j, box_i, "XC")
    YC = _weighted_center_mean(ds_fine["YC"], ds_fine["RAC"], box_j, box_i, "YC")
    XG = ds_fine["XG"].coarsen(j_g=box_j, i_g=box_i, boundary="trim").mean().rename("XG")
    YG = ds_fine["YG"].coarsen(j_g=box_j, i_g=box_i, boundary="trim").mean().rename("YG")

    hFacC_weight = (
        ds_fine["hFacC"] * ds_fine["RAC"] * DRF
    ).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    hFacC = _safe_divide(hFacC_weight, DRF * RAC).rename("hFacC")

    hFacW_weight = (
        ds_fine["hFacW"] * ds_fine["DYG"] * DRF
    ).coarsen(j=box_j, boundary="trim").sum()
    hFacW_den = (ds_fine["DYG"] * DRF).coarsen(j=box_j, boundary="trim").sum()
    hFacW_den = hFacW_den.isel(i_g=slice(i0, None, box_i))
    hFacW = (
        _safe_divide(
            hFacW_weight.isel(i_g=slice(i0, None, box_i)),
            hFacW_den,
        )
        .transpose("k", "j", "i_g")
        .rename("hFacW")
    )

    hFacS_weight = (
        ds_fine["hFacS"] * ds_fine["DXG"] * DRF
    ).coarsen(i=box_i, boundary="trim").sum()
    hFacS_den = (ds_fine["DXG"] * DRF).coarsen(i=box_i, boundary="trim").sum()
    hFacS_den = hFacS_den.isel(j_g=slice(j0, None, box_j))
    hFacS = (
        _safe_divide(
            hFacS_weight.isel(j_g=slice(j0, None, box_j)),
            hFacS_den,
        )
        .transpose("k", "j_g", "i")
        .rename("hFacS")
    )

    coarse_vars = {
        "RAC": RAC,
        "RAW": RAW,
        "RAS": RAS,
        "RAZ": RAZ,
        "DXG": DXG,
        "DYG": DYG,
        "DXC": DXC,
        "DYC": DYC,
        "XC": XC,
        "YC": YC,
        "XG": XG,
        "YG": YG,
        "hFacC": hFacC,
        "hFacW": hFacW,
        "hFacS": hFacS,
    }
    coarse_vars = {name: _as_ordinal(da) for name, da in coarse_vars.items()}

    ds_coarse = xr.Dataset(
        dict(
            **coarse_vars,
            DRF=DRF,
            DRC=DRC,
            RF=RF,
            RC=RC,
            k_l=k_l,
        )
    )
    return _assign_ordinal_coords(ds_coarse)
