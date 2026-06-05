"""Horizontal strain-rate diagnostics on MITgcm C-grid data."""

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


def total_strain(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    u_name: str = "U",
    v_name: str = "V",
    hfac_u_name: str = "hFacW",
    hfac_v_name: str = "hFacS",
    hfac_c_name: str = "hFacC",
    rac_name: str = "RAC",
    raz_name: str = "RAZ",
    drf_name: str = "DRF",
    dy_u_name: str = "DYG",
    dx_v_name: str = "DXG",
    dx_u_name: str = "DXC",
    dy_v_name: str = "DYC",
    u_face_dim: str = "i_g",
    v_face_dim: str = "j_g",
    x_center_dim: str = "i",
    y_center_dim: str = "j",
    y_boundary: str = "fill",
    output_name: str = "strain_magnitude",
    return_components: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Total horizontal strain magnitude at MITgcm tracer-cell centers.

    The returned strain is

      ``sqrt((du/dx - dv/dy)^2 + (dv/dx + du/dy)^2)``.

    Normal derivatives are computed conservatively at tracer centers using
    volume flux differences. Cross derivatives are computed first at zeta
    points using tangential line integrals, then interpolated to centers.
    """
    ds = xr.merge([ds_grid, ds_ocean])
    _require_variables(
        ds,
        (
            u_name,
            v_name,
            hfac_u_name,
            hfac_v_name,
            hfac_c_name,
            rac_name,
            raz_name,
            drf_name,
            dy_u_name,
            dx_v_name,
            dx_u_name,
            dy_v_name,
        ),
    )
    _require_dims(ds[u_name], ("k", y_center_dim, u_face_dim), u_name)
    _require_dims(ds[v_name], ("k", v_face_dim, x_center_dim), v_name)
    _require_dims(ds[hfac_u_name], ("k", y_center_dim, u_face_dim), hfac_u_name)
    _require_dims(ds[hfac_v_name], ("k", v_face_dim, x_center_dim), hfac_v_name)
    _require_dims(ds[hfac_c_name], ("k", y_center_dim, x_center_dim), hfac_c_name)
    _require_dims(ds[rac_name], (y_center_dim, x_center_dim), rac_name)
    _require_dims(ds[raz_name], (v_face_dim, u_face_dim), raz_name)
    _require_dims(ds[drf_name], ("k",), drf_name)
    _require_dims(ds[dy_u_name], (y_center_dim, u_face_dim), dy_u_name)
    _require_dims(ds[dx_v_name], (v_face_dim, x_center_dim), dx_v_name)
    _require_dims(ds[dx_u_name], (y_center_dim, u_face_dim), dx_u_name)
    _require_dims(ds[dy_v_name], (v_face_dim, x_center_dim), dy_v_name)

    grid = make_grid.make_grid_periodic_x(ds)

    wet_u = ds[hfac_u_name] > 0
    wet_v = ds[hfac_v_name] > 0
    wet_c = ds[hfac_c_name] > 0

    u = ds[u_name].where(wet_u, 0.0).fillna(0.0)
    v = ds[v_name].where(wet_v, 0.0).fillna(0.0)
    drf = ds[drf_name]

    volume_c = (drf * ds[rac_name] * ds[hfac_c_name]).transpose("k", y_center_dim, x_center_dim)

    u_normal_transport = u * ds[dy_u_name] * ds[hfac_u_name] * drf
    v_normal_transport = v * ds[dx_v_name] * ds[hfac_v_name] * drf
    dudx = grid.diff(u_normal_transport, "X") / volume_c
    dvdy = grid.diff(v_normal_transport, "Y", boundary=y_boundary) / volume_c

    u_tangential = u * ds[dx_u_name] * ds[hfac_u_name] * drf
    v_tangential = v * ds[dy_v_name] * ds[hfac_v_name] * drf
    corner_volume = drf * ds[raz_name]
    dvdx_corner = grid.diff(v_tangential, "X") / corner_volume
    dudy_corner = grid.diff(u_tangential, "Y", boundary=y_boundary) / corner_volume

    dvdx = grid.interp(grid.interp(dvdx_corner, "X", to="center"), "Y", to="center")
    dudy = grid.interp(grid.interp(dudy_corner, "X", to="center"), "Y", to="center")

    normal_component = (dudx - dvdy).where(wet_c).rename("normal_strain")
    shear_component = (dvdx + dudy).where(wet_c).rename("shear_strain")
    strain = xr.apply_ufunc(np.hypot, normal_component, shear_component).where(wet_c).rename(output_name)

    units = "s^-1"
    normal_component.attrs.update(long_name="du/dx - dv/dy at tracer-cell centers", units=units, y_boundary=y_boundary)
    shear_component.attrs.update(long_name="dv/dx + du/dy at tracer-cell centers", units=units, y_boundary=y_boundary)
    strain.attrs.update(
        long_name="horizontal total strain magnitude at tracer-cell centers",
        units=units,
        definition="sqrt((du/dx - dv/dy)^2 + (dv/dx + du/dy)^2)",
        y_boundary=y_boundary,
    )

    if return_components:
        return xr.Dataset(
            {
                strain.name: strain,
                normal_component.name: normal_component,
                shear_component.name: shear_component,
            }
        )
    return strain


def _total_strain(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill",
) -> xr.DataArray:
    """Backward-compatible wrapper for older scripts."""
    return total_strain(ds_grid, ds_ocean, y_boundary=boundary_Y, output_name="strain_magnitude")


def total_strain_coarse(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill",
) -> xr.DataArray:
    """Backward-compatible coarse-grid name used by older notebooks/scripts."""
    return _total_strain(ds_grid, ds_ocean, boundary_Y=boundary_Y)
