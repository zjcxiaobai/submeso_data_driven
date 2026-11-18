"""Build the coarse MITgcm C-grid used by this project."""

import numpy as np
import xarray as xr


def _ordinal(da: xr.DataArray) -> xr.DataArray:
    """Replace inherited coordinates with integer indices."""
    da = da.reset_coords(drop=True)
    return da.assign_coords(
        {
            dim: np.arange(da.sizes[dim], dtype=np.int64)
            for dim in da.dims
        }
    )


def _divide(
    numerator: xr.DataArray,
    denominator: xr.DataArray,
) -> xr.DataArray:
    """Divide while returning zero where the denominator is zero."""
    return xr.where(
        denominator > 0,
        numerator / denominator,
        0.0,
    )


def _weighted_mean(
    values: xr.DataArray,
    weights: xr.DataArray,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    numerator = (
        values * weights
    ).coarsen(
        j=box_j,
        i=box_i,
        boundary="trim",
    ).sum()

    denominator = weights.coarsen(
        j=box_j,
        i=box_i,
        boundary="trim",
    ).sum()

    return _divide(numerator, denominator)


def build_coarse_grid_staggered(
    fine: xr.Dataset,
    box_j: int,
    box_i: int,
) -> xr.Dataset:
    """Coarsen the MITgcm grid while preserving its C-grid staggering."""

    # --------------------------------------------------------------
    # Horizontal cell and corner areas
    # --------------------------------------------------------------
    rac = fine["RAC"].coarsen(
        j=box_j,
        i=box_i,
        boundary="trim",
    ).sum()

    raz = fine["RAZ"].coarsen(
        j_g=box_j,
        i_g=box_i,
        boundary="trim",
    ).sum()

    # --------------------------------------------------------------
    # Horizontal distances
    # --------------------------------------------------------------
    dxg = (
        fine["DXG"]
        .coarsen(i=box_i, boundary="trim")
        .sum()
        .isel(j_g=slice(0, None, box_j))
    )

    dyg = (
        fine["DYG"]
        .coarsen(j=box_j, boundary="trim")
        .sum()
        .isel(i_g=slice(0, None, box_i))
    )

    dyc = (
        fine["DYC"]
        .coarsen(i=box_i, boundary="trim")
        .sum()
        .isel(j_g=slice(0, None, box_j))
    )

    dxc = (
        fine["DXC"]
        .coarsen(j=box_j, boundary="trim")
        .sum()
        .isel(i_g=slice(0, None, box_i))
    )

    # --------------------------------------------------------------
    # Cell-center coordinates
    # --------------------------------------------------------------
    xc = _weighted_mean(
        fine["XC"],
        fine["RAC"],
        box_j,
        box_i,
    )

    yc = _weighted_mean(
        fine["YC"],
        fine["RAC"],
        box_j,
        box_i,
    )

    # --------------------------------------------------------------
    # Wet fractions
    # --------------------------------------------------------------
    drf = fine["DRF"]

    wet_volume = (
        fine["hFacC"] * fine["RAC"] * drf
    ).coarsen(
        j=box_j,
        i=box_i,
        boundary="trim",
    ).sum()

    hfac_c = _divide(
        wet_volume,
        rac * drf,
    )

    wet_u_area = (
        fine["hFacW"] * fine["DYG"] * drf
    ).coarsen(
        j=box_j,
        boundary="trim",
    ).sum()
    wet_u_area = wet_u_area.isel(
        i_g=slice(0, None, box_i)
    )

    total_u_area = (
        fine["DYG"] * drf
    ).coarsen(
        j=box_j,
        boundary="trim",
    ).sum()
    total_u_area = total_u_area.isel(
        i_g=slice(0, None, box_i)
    )

    hfac_w = _divide(
        wet_u_area,
        total_u_area,
    ).transpose("k", "j", "i_g")

    wet_v_area = (
        fine["hFacS"] * fine["DXG"] * drf
    ).coarsen(
        i=box_i,
        boundary="trim",
    ).sum()
    wet_v_area = wet_v_area.isel(
        j_g=slice(0, None, box_j)
    )

    total_v_area = (
        fine["DXG"] * drf
    ).coarsen(
        i=box_i,
        boundary="trim",
    ).sum()
    total_v_area = total_v_area.isel(
        j_g=slice(0, None, box_j)
    )

    hfac_s = _divide(
        wet_v_area,
        total_v_area,
    ).transpose("k", "j_g", "i")

    # --------------------------------------------------------------
    # Assemble the coarse grid
    # --------------------------------------------------------------
    variables = {
        "RAC": rac,
        "RAZ": raz,
        "DXG": dxg,
        "DYG": dyg,
        "DXC": dxc,
        "DYC": dyc,
        "XC": xc,
        "YC": yc,
        "hFacC": hfac_c,
        "hFacW": hfac_w,
        "hFacS": hfac_s,
        "DRF": fine["DRF"],
        "RF": fine["RF"],
        "RC": fine["RC"],
        "k_l": fine["k_l"],
    }

    return xr.Dataset(
        {
            name: _ordinal(value).rename(name)
            for name, value in variables.items()
        }
    )