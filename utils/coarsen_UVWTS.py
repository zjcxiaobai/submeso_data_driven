"""Coarsen MITgcm variables onto the model's coarse C-grid."""

import numpy as np
import xarray as xr


def _finish(
    da: xr.DataArray,
    dims: tuple[str, ...],
    name: str,
) -> xr.DataArray:
    """Set dimension order, variable name, and ordinal coordinates."""
    da = da.transpose(*dims).rename(name).reset_coords(drop=True)
    coords = {
        dim: np.arange(da.sizes[dim], dtype=np.int64)
        for dim in dims
    }
    return da.assign_coords(coords)


def _weighted_mean(
    values: xr.DataArray,
    weights: xr.DataArray,
    coarsen_dims: dict[str, int],
    fill_value: float = np.nan,
) -> xr.DataArray:
    """Calculate a weighted mean over coarse-grid blocks."""
    values = values.where(weights > 0, 0.0).fillna(0.0)

    numerator = (
        values * weights
    ).coarsen(
        **coarsen_dims,
        boundary="trim",
    ).sum()

    denominator = weights.coarsen(
        **coarsen_dims,
        boundary="trim",
    ).sum()

    return xr.where(
        denominator > 0,
        numerator / denominator,
        fill_value,
    )


def _coarsen_u(
    da: xr.DataArray,
    grid: xr.Dataset,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    weights = grid["hFacW"] * grid["DYG"]

    coarse = _weighted_mean(
        da,
        weights,
        {"j": box_j},
        fill_value=0.0,
    )
    coarse = coarse.isel(i_g=slice(0, None, box_i))

    return _finish(coarse, ("k", "j", "i_g"), "U")


def _coarsen_v(
    da: xr.DataArray,
    grid: xr.Dataset,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    weights = grid["hFacS"] * grid["DXG"]

    coarse = _weighted_mean(
        da,
        weights,
        {"i": box_i},
        fill_value=0.0,
    )
    coarse = coarse.isel(j_g=slice(0, None, box_j))

    return _finish(coarse, ("k", "j_g", "i"), "V")


def _coarsen_w(
    da: xr.DataArray,
    grid: xr.Dataset,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    upper = grid["hFacC"]
    lower = grid["hFacC"].shift(k=-1, fill_value=0.0)

    open_fraction = xr.apply_ufunc(
        np.minimum,
        upper,
        lower,
    ).rename(k="k_l")

    k_coord = da.coords.get(
        "k_l",
        np.arange(da.sizes["k_l"]),
    )
    open_fraction = open_fraction.assign_coords(k_l=k_coord)

    weights = open_fraction * grid["RAC"]

    coarse = _weighted_mean(
        da,
        weights,
        {"j": box_j, "i": box_i},
    )

    return _finish(coarse, ("k_l", "j", "i"), "W")


def _coarsen_tracer(
    da: xr.DataArray,
    grid: xr.Dataset,
    box_j: int,
    box_i: int,
    name: str,
) -> xr.DataArray:
    weights = (
        grid["hFacC"]
        * grid["RAC"]
        * grid["DRF"]
    )

    coarse = _weighted_mean(
        da,
        weights,
        {"j": box_j, "i": box_i},
    )

    return _finish(coarse, ("k", "j", "i"), name)


def _coarsen_surface(
    da: xr.DataArray,
    grid: xr.Dataset,
    box_j: int,
    box_i: int,
    name: str,
) -> xr.DataArray:
    wet = grid["hFacC"].isel(k=0) > 0
    weights = grid["RAC"].where(wet, 0.0)

    coarse = _weighted_mean(
        da,
        weights,
        {"j": box_j, "i": box_i},
    )

    return _finish(coarse, ("j", "i"), name)


def coarse_field(
    varname: str,
    da: xr.DataArray,
    grid: xr.Dataset,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    """Coarsen one variable used by the SubmesoFluxNet pipeline."""
    name = varname.lower()

    if name == "u":
        return _coarsen_u(da, grid, box_j, box_i)

    if name == "v":
        return _coarsen_v(da, grid, box_j, box_i)

    if name == "w":
        return _coarsen_w(da, grid, box_j, box_i)

    if name in {"t", "s"}:
        return _coarsen_tracer(
            da,
            grid,
            box_j,
            box_i,
            varname,
        )

    if name in {"kpphbl", "eta"}:
        return _coarsen_surface(
            da,
            grid,
            box_j,
            box_i,
            varname,
        )

    raise ValueError(f"Unsupported variable: {varname}")