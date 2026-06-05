"""Horizontal temperature-gradient diagnostics on MITgcm C-grid data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from src.utils import make_grid


def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variable(s): {missing}")


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dimension(s): {missing}; got dims {da.dims}")


def horizontal_temperature_gradient(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    temp_name: str = "T",
    hfac_c_name: str = "hFacC",
    dx_name: str = "DXC",
    dy_name: str = "DYC",
    x_face_dim: str = "i_g",
    y_face_dim: str = "j_g",
    x_center_dim: str = "i",
    y_center_dim: str = "j",
    y_boundary: str = "fill",
    output_name: str = "gradT",
    return_components: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Horizontal temperature-gradient magnitude at tracer-cell centers.

    ``dT/dx`` is computed on U faces using ``DXC`` and interpolated back to
    centers. ``dT/dy`` is computed on V faces using ``DYC`` and interpolated
    back to centers. The final magnitude is ``sqrt((dT/dx)^2 + (dT/dy)^2)``.
    """
    ds = xr.merge([ds_grid, ds_ocean])
    _require_variables(ds, (temp_name, hfac_c_name, dx_name, dy_name))
    _require_dims(ds[temp_name], ("k", y_center_dim, x_center_dim), temp_name)
    _require_dims(ds[hfac_c_name], ("k", y_center_dim, x_center_dim), hfac_c_name)
    _require_dims(ds[dx_name], (y_center_dim, x_face_dim), dx_name)
    _require_dims(ds[dy_name], (y_face_dim, x_center_dim), dy_name)

    grid = make_grid.make_grid_periodic_x(ds)
    wet_c = ds[hfac_c_name] > 0
    temp = ds[temp_name].where(wet_c)

    dtdx_face = grid.diff(temp, "X") / ds[dx_name]
    dtdy_face = grid.diff(temp, "Y", boundary=y_boundary) / ds[dy_name]

    dtdx = grid.interp(dtdx_face, "X", to="center").where(wet_c).rename("dTdx")
    dtdy = grid.interp(dtdy_face, "Y", to="center").where(wet_c).rename("dTdy")
    grad = xr.apply_ufunc(np.hypot, dtdx, dtdy).where(wet_c).rename(output_name)

    temp_units = ds[temp_name].attrs.get("units", "K")
    grad_units = f"{temp_units} m^-1"
    dtdx.attrs.update(long_name="zonal temperature gradient at tracer-cell centers", units=grad_units)
    dtdy.attrs.update(long_name="meridional temperature gradient at tracer-cell centers", units=grad_units, y_boundary=y_boundary)
    grad.attrs.update(
        long_name="horizontal temperature-gradient magnitude at tracer-cell centers",
        units=grad_units,
        definition="sqrt((dT/dx)^2 + (dT/dy)^2)",
        y_boundary=y_boundary,
    )

    if return_components:
        return xr.Dataset({grad.name: grad, dtdx.name: dtdx, dtdy.name: dtdy})
    return grad


def T_grad(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill",
) -> xr.DataArray:
    """Backward-compatible wrapper for older scripts."""
    return horizontal_temperature_gradient(ds_grid, ds_ocean, y_boundary=boundary_Y, output_name="gradT")
