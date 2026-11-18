"""Mixed-layer averaging utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dimension(s): {missing}; got dims {da.dims}")


def _require_grid_vars(ds_grid: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds_grid]
    if missing:
        raise KeyError(f"Grid dataset is missing required variable(s): {missing}")


def _positive_depth_interfaces(ds_grid: xr.Dataset, k_size: int, k_coord: xr.DataArray | None) -> tuple[xr.DataArray, xr.DataArray]:
    """Return positive-down top and bottom depths for each k layer."""
    rf = -ds_grid["RF"]
    _require_dims(rf, ("k_p1",), "RF")
    if rf.sizes["k_p1"] < k_size + 1:
        raise ValueError(f"RF must have at least k_size + 1 interfaces; got {rf.sizes['k_p1']} for k={k_size}")

    z_top = rf.isel(k_p1=slice(0, k_size)).rename({"k_p1": "k"}).reset_coords(drop=True)
    z_bot = rf.isel(k_p1=slice(1, k_size + 1)).rename({"k_p1": "k"}).reset_coords(drop=True)

    if k_coord is not None:
        z_top = z_top.assign_coords(k=k_coord)
        z_bot = z_bot.assign_coords(k=k_coord)
    else:
        coord = np.arange(k_size, dtype="int64")
        z_top = z_top.assign_coords(k=coord)
        z_bot = z_bot.assign_coords(k=coord)

    return z_top, z_bot


def mixed_layer_average(
    data: xr.DataArray,
    kpphbl: xr.DataArray,
    ds_grid: xr.Dataset,
    *,
    output_name: str | None = None,
) -> xr.DataArray:
    """
    Thickness-weighted mixed-layer average of a 3D field.

    Parameters
    ----------
    data
        Three-dimensional field on tracer centers with dimensions ``(k, j, i)``.
    kpphbl
        Mixed-layer depth in meters, positive downward, on ``(j, i)``.
    ds_grid
        Grid dataset containing ``RF`` and ``hFacC`` from ``read_mitgcm`` or
        ``build_coarse_grid``.
    output_name
        Optional name for the returned 2D field.

    Returns
    -------
    xr.DataArray
        Mixed-layer average on ``(j, i)``. Cells with zero wet mixed-layer
        thickness are returned as NaN.
    """
    _require_dims(data, ("k", "j", "i"), "data")
    _require_dims(kpphbl, ("j", "i"), "kpphbl")
    _require_grid_vars(ds_grid, ("RF", "hFacC"))
    _require_dims(ds_grid["hFacC"], ("k", "j", "i"), "hFacC")

    if ds_grid["hFacC"].sizes["k"] != data.sizes["k"]:
        raise ValueError(f"data and hFacC have different k sizes: {data.sizes['k']} != {ds_grid['hFacC'].sizes['k']}")

    k_coord = data.coords.get("k")
    z_top_1d, z_bot_1d = _positive_depth_interfaces(ds_grid, data.sizes["k"], k_coord)
    z_top = z_top_1d.broadcast_like(data)
    z_bot = z_bot_1d.broadcast_like(data)

    layer_thickness = (z_bot - z_top) * ds_grid["hFacC"]
    column_depth = layer_thickness.sum("k")
    mld = xr.where(np.isfinite(kpphbl), kpphbl, 0.0).clip(min=0.0)
    mld = xr.where(column_depth > 0, mld.clip(max=column_depth), 0.0)

    mld_3d = mld.broadcast_like(data)
    overlap = xr.apply_ufunc(np.minimum, mld_3d, z_bot) - z_top
    weights = overlap.clip(min=0.0)
    weights = xr.where(weights > layer_thickness, layer_thickness, weights)

    numerator = (data * weights).sum("k", skipna=True)
    denominator = weights.where(np.isfinite(data)).sum("k")
    result = xr.where(denominator > 0, numerator / denominator, np.nan)

    name = output_name or (f"{data.name}_mld_mean" if data.name else "mld_mean")
    result = result.rename(name)
    result.attrs.update(
        long_name=f"mixed-layer average of {data.name or 'field'}",
        units=data.attrs.get("units", ""),
        method="thickness-weighted average over layer overlap with kpphbl",
    )
    return result


def mld_avg(
    data: xr.DataArray,
    kpphbl: xr.DataArray,
    ds_grid: xr.Dataset,
) -> xr.DataArray:
    """Backward-compatible wrapper for existing scripts."""
    return mixed_layer_average(data, kpphbl, ds_grid)
