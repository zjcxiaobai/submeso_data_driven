"""Vertical heat-flux decomposition for MITgcm C-grid data."""

from __future__ import annotations

import numpy as np
import xarray as xr

from submeso_data_driven.utils import make_grid


RHO0 = 1035.0
CP = 3994.0
_HEAT_FLUX_FACTOR = RHO0 * CP


def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variable(s): {missing}")


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dimension(s): {missing}; got dims {da.dims}")


def _as_ordinal(da: xr.DataArray) -> xr.DataArray:
    da = da.reset_coords(drop=True)
    coords = {dim: xr.DataArray(np.arange(da.sizes[dim], dtype="int64"), dims=dim) for dim in da.dims}
    return da.assign_coords(coords)


def _area_weighted_boxmean(
    values: xr.DataArray,
    area: xr.DataArray,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    masked = values.where(area > 0, 0.0).fillna(0.0)
    numerator = (masked * area).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    denominator = area.coarsen(j=box_j, i=box_i, boundary="trim").sum()
    return xr.where(denominator > 0, numerator / denominator, np.nan)


def _open_interface_area(ds: xr.Dataset, w_name: str, hfac_c_name: str, rac_name: str) -> xr.DataArray:
    upper = ds[hfac_c_name]
    lower = ds[hfac_c_name].shift(k=-1, fill_value=0.0)
    open_fraction = xr.apply_ufunc(np.minimum, upper, lower).rename({"k": "k_l"})
    if "k_l" in ds[w_name].coords:
        open_fraction = open_fraction.assign_coords(k_l=ds[w_name].coords["k_l"])
    return (open_fraction * ds[rac_name]).astype("float64").rename("A_wet_interface")


def _convert_flux_units(da: xr.DataArray, to_wm2: bool) -> xr.DataArray:
    if to_wm2:
        out = (da * _HEAT_FLUX_FACTOR).copy()
        out.attrs["units"] = "W m^-2"
        return out
    out = da.copy()
    out.attrs["units"] = "K m s^-1"
    return out


def _set_flux_attrs(da: xr.DataArray, long_name: str, to_wm2: bool) -> xr.DataArray:
    da.attrs.update(
        long_name=long_name,
        units="W m^-2" if to_wm2 else "K m s^-1",
    )
    return da


def vertical_heat_flux_decomposition(
    ds_fine: xr.Dataset,
    ds_coarse: xr.Dataset,
    ds_ocean: xr.Dataset,
    ds_ocean_c: xr.Dataset,
    *,
    w_name: str = "W",
    temp_name: str = "T",
    hfac_c_name: str = "hFacC",
    rac_name: str = "RAC",
    boundary_Z: str = "fill",
    box_j: int = 10,
    box_i: int = 10,
    to_wm2: bool = True,
) -> xr.Dataset:
    """
    Compute total, coarse-total, mesoscale, and submesoscale vertical heat flux.

    Definitions on W interfaces ``k_l``:

    ``wT_total = W * interp_z(T)`` on the fine grid.
    ``wT_coarse_total = coarse_mean(wT_total)``.
    ``wT_meso = coarse(W) * interp_z(coarse(T))``.
    ``wT_submeso = wT_coarse_total - wT_meso``.

    All four heat-flux diagnostics are also interpolated to tracer centers ``k``.
    """
    if box_j <= 0 or box_i <= 0:
        raise ValueError("box_j and box_i must be positive integers")

    ds = xr.merge([ds_fine, ds_ocean])
    _require_variables(ds, (w_name, temp_name, hfac_c_name, rac_name))
    _require_variables(ds_ocean_c, (w_name, temp_name))
    _require_dims(ds[w_name], ("k_l", "j", "i"), w_name)
    _require_dims(ds[temp_name], ("k", "j", "i"), temp_name)
    _require_dims(ds[hfac_c_name], ("k", "j", "i"), hfac_c_name)
    _require_dims(ds[rac_name], ("j", "i"), rac_name)
    _require_dims(ds_ocean_c[w_name], ("k_l", "j", "i"), f"coarse {w_name}")
    _require_dims(ds_ocean_c[temp_name], ("k", "j", "i"), f"coarse {temp_name}")

    grid_fine = make_grid.make_grid_periodic_x(ds)
    grid_coarse = make_grid.make_grid_periodic_x(ds_coarse)

    interface_area = _open_interface_area(ds, w_name, hfac_c_name, rac_name)
    temp_interface = grid_fine.interp(ds[temp_name], "Z", to="left", boundary=boundary_Z)
    wT_total_kL = (ds[w_name] * temp_interface).where(interface_area > 0).rename("wT_total_kL")

    wT_coarse_total_kL = _area_weighted_boxmean(wT_total_kL, interface_area, box_j, box_i)
    wT_coarse_total_kL = _as_ordinal(wT_coarse_total_kL).rename("wT_coarse_total_kL")

    w_bar_kL = ds_ocean_c[w_name].rename("w_bar_kL")
    T_bar_kC = ds_ocean_c[temp_name].rename("T_bar_kC")
    T_bar_kL = grid_coarse.interp(T_bar_kC, "Z", to="left", boundary=boundary_Z).rename("T_bar_kL")
    wT_meso_kL = (w_bar_kL * T_bar_kL).rename("wT_meso_kL")
    wT_sub_kL = (wT_coarse_total_kL - wT_meso_kL).rename("wT_sub_kL")

    wT_total_kC = grid_fine.interp(wT_total_kL, "Z", to="center", boundary=boundary_Z).rename("wT_total_kC")
    # Fine and coarse horizontal fields cannot share the same j/i dimension
    # names inside one Dataset without xarray aligning them to a common index.
    wT_total_kL = wT_total_kL.rename({"j": "j_f", "i": "i_f"})
    wT_total_kC = wT_total_kC.rename({"j": "j_f", "i": "i_f"})

    wT_coarse_total_kC = grid_coarse.interp(
        wT_coarse_total_kL, "Z", to="center", boundary=boundary_Z
    ).rename("wT_coarse_total_kC")
    wT_meso_kC = grid_coarse.interp(wT_meso_kL, "Z", to="center", boundary=boundary_Z).rename("wT_meso_kC")
    wT_sub_kC = grid_coarse.interp(wT_sub_kL, "Z", to="center", boundary=boundary_Z).rename("wT_sub_kC")

    fluxes = [
        (wT_total_kL, "fine-grid total vertical heat flux at W interfaces"),
        (wT_total_kC, "fine-grid total vertical heat flux at tracer centers"),
        (wT_coarse_total_kL, "coarse mean of total vertical heat flux at W interfaces"),
        (wT_coarse_total_kC, "coarse mean of total vertical heat flux at tracer centers"),
        (wT_meso_kL, "mesoscale vertical heat flux at W interfaces"),
        (wT_meso_kC, "mesoscale vertical heat flux at tracer centers"),
        (wT_sub_kL, "submesoscale vertical heat flux at W interfaces"),
        (wT_sub_kC, "submesoscale vertical heat flux at tracer centers"),
    ]
    converted = {da.name: _set_flux_attrs(_convert_flux_units(da, to_wm2), long_name, to_wm2) for da, long_name in fluxes}

    w_bar_kL.attrs.update(long_name="coarse mean vertical velocity at W interfaces", units=ds[w_name].attrs.get("units", "m s^-1"))
    T_bar_kC.attrs.update(long_name="coarse mean temperature at tracer centers", units=ds[temp_name].attrs.get("units", "K"))
    T_bar_kL.attrs.update(long_name="coarse mean temperature interpolated to W interfaces", units=ds[temp_name].attrs.get("units", "K"))

    return xr.Dataset(
        {
            **converted,
            w_bar_kL.name: w_bar_kL,
            T_bar_kC.name: T_bar_kC,
            T_bar_kL.name: T_bar_kL,
        },
        attrs={
            "boundary_Z": boundary_Z,
            "box_j": box_j,
            "box_i": box_i,
            "rho0": RHO0,
            "cp": CP,
            "heat_flux_units": "W m^-2" if to_wm2 else "K m s^-1",
        },
    )


def submeso_vertical_heat_flux(
    ds_fine: xr.Dataset,
    ds_coarse: xr.Dataset,
    ds_ocean: xr.Dataset,
    ds_ocean_c: xr.Dataset,
    boundary_Z: str = "fill",
    box_j: int = 10,
    box_i: int = 10,
    to_wm2: bool = True,
    output: str = "full",
) -> xr.DataArray | xr.Dataset:
    """
    Vertical heat-flux interface with explicit output modes.

    Parameters
    ----------
    output
        ``"submeso_center"`` returns only ``wT_sub_kC``.
        ``"full"`` returns a Dataset containing total, coarse-total,
        mesoscale, and submesoscale heat fluxes at interfaces and centers.
    """
    budget = vertical_heat_flux_decomposition(
        ds_fine,
        ds_coarse,
        ds_ocean,
        ds_ocean_c,
        boundary_Z=boundary_Z,
        box_j=box_j,
        box_i=box_i,
        to_wm2=to_wm2,
    )

    if output == "submeso_center":
        return budget["wT_sub_kC"]
    if output == "full":
        return budget
    raise ValueError("output must be 'submeso_center' or 'full'")
