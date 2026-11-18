"""Coarsen MITgcm ocean variables onto the repo's staggered coarse grids."""

from __future__ import annotations

import numpy as np
import xarray as xr


_CENTER_TRACERS = {"theta", "t", "salt", "s"}
_SURFACE_CENTER_FIELDS = {"kpphbl", "mld", "hbl", "eta", "ssh", "etabar"}


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    if tuple(da.dims) != dims:
        raise ValueError(f"{name} must have dims {dims}; got {da.dims}")


def _require_grid_vars(ds_grid: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds_grid]
    if missing:
        raise KeyError(f"Fine grid is missing required variable(s): {missing}")


def _validate_boxes(box_j: int, box_i: int) -> None:
    if not isinstance(box_j, int) or not isinstance(box_i, int) or box_j <= 0 or box_i <= 0:
        raise ValueError("box_j and box_i must be positive integers")


def _as_ordinal(da: xr.DataArray) -> xr.DataArray:
    """Drop auxiliary coordinates and replace dimension coordinates with 0..N-1."""
    da = da.reset_coords(drop=True)
    coords = {dim: xr.DataArray(np.arange(da.sizes[dim], dtype="int64"), dims=dim) for dim in da.dims}
    return da.assign_coords(coords)


def _safe_weighted_average(
    values: xr.DataArray,
    weights: xr.DataArray,
    coarsen_dims: dict[str, int],
    *,
    fill_value: float = np.nan,
) -> xr.DataArray:
    masked_values = values.where(weights > 0, 0.0).fillna(0.0)
    numerator = (masked_values * weights).coarsen(**coarsen_dims, boundary="trim").sum()
    denominator = weights.coarsen(**coarsen_dims, boundary="trim").sum()
    return xr.where(denominator > 0, numerator / denominator, fill_value)


def _coarsen_u(da: xr.DataArray, ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.DataArray:
    _require_dims(da, ("k", "j", "i_g"), "U")
    weights = ds_fine["hFacW"] * ds_fine["DYG"]
    coarse = _safe_weighted_average(da, weights, {"j": box_j}, fill_value=0.0)
    coarse = coarse.isel(i_g=slice(0, None, box_i)).transpose("k", "j", "i_g").rename(da.name or "U")
    return _as_ordinal(coarse)


def _coarsen_v(da: xr.DataArray, ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.DataArray:
    _require_dims(da, ("k", "j_g", "i"), "V")
    weights = ds_fine["hFacS"] * ds_fine["DXG"]
    coarse = _safe_weighted_average(da, weights, {"i": box_i}, fill_value=0.0)
    coarse = coarse.isel(j_g=slice(0, None, box_j)).transpose("k", "j_g", "i").rename(da.name or "V")
    return _as_ordinal(coarse)


def _coarsen_w(da: xr.DataArray, ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.DataArray:
    _require_dims(da, ("k_l", "j", "i"), "W")
    upper = ds_fine["hFacC"]
    lower = ds_fine["hFacC"].shift(k=-1, fill_value=0.0)
    open_fraction = xr.apply_ufunc(np.minimum, upper, lower)
    open_fraction = open_fraction.rename({"k": "k_l"}).assign_coords(k_l=da.coords.get("k_l", np.arange(da.sizes["k_l"])))
    area = (open_fraction * ds_fine["RAC"]).astype("float64")
    coarse = _safe_weighted_average(da, area, {"j": box_j, "i": box_i})
    coarse = coarse.transpose("k_l", "j", "i").rename(da.name or "W")
    return _as_ordinal(coarse)


def _coarsen_center_3d(da: xr.DataArray, ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.DataArray:
    _require_dims(da, ("k", "j", "i"), da.name or "center field")
    weights = ds_fine["hFacC"] * ds_fine["RAC"] * ds_fine["DRF"]
    coarse = _safe_weighted_average(da, weights, {"j": box_j, "i": box_i})
    coarse = coarse.transpose("k", "j", "i").rename(da.name or "center_field")
    return _as_ordinal(coarse)


def _coarsen_surface_center(da: xr.DataArray, ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.DataArray:
    _require_dims(da, ("j", "i"), da.name or "surface center field")
    wet_surface = ds_fine["hFacC"].isel(k=0) > 0
    weights = xr.where(wet_surface, ds_fine["RAC"], 0.0)
    coarse = _safe_weighted_average(da, weights, {"j": box_j, "i": box_i})
    coarse = coarse.transpose("j", "i").rename(da.name or "surface_center_field")
    return _as_ordinal(coarse)


def coarse_field(
    varname: str,
    da: xr.DataArray,
    ds_fine: xr.Dataset,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    """
    Coarsen one fine-grid MITgcm variable to a staggered coarse grid.

    Parameters
    ----------
    varname
        Variable family. Supported values are ``U``, ``V``, ``W``, ``Theta``/
        ``T``, ``Salt``/``S``, ``kpphbl``/``mld``/``hbl``, and ``Eta``/``SSH``.
    da
        Fine-grid field with dimensions matching ``varname``.
    ds_fine
        Fine static grid from ``read_mitgcm.read_mitgrid_staggered``.
    box_j, box_i
        Coarse-cell size in fine-grid points.
    """
    _validate_boxes(box_j, box_i)
    _require_grid_vars(ds_fine, ("RAC", "DRF", "hFacC", "hFacW", "hFacS", "DXG", "DYG"))

    var = varname.lower()
    if var == "u":
        return _coarsen_u(da, ds_fine, box_j, box_i)
    if var == "v":
        return _coarsen_v(da, ds_fine, box_j, box_i)
    if var == "w":
        return _coarsen_w(da, ds_fine, box_j, box_i)
    if var in _CENTER_TRACERS:
        return _coarsen_center_3d(da, ds_fine, box_j, box_i)
    if var in _SURFACE_CENTER_FIELDS:
        return _coarsen_surface_center(da, ds_fine, box_j, box_i)

    raise ValueError(f"Unsupported varname: {varname!r}")
