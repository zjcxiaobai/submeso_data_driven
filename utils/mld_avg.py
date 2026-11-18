"""Mixed-layer averaging for the MITgcm fields used in this project."""

import numpy as np
import xarray as xr


def mld_avg(
    data: xr.DataArray,
    kpphbl: xr.DataArray,
    grid: xr.Dataset,
) -> xr.DataArray:
    """Calculate the thickness-weighted mixed-layer average."""
    nk = data.sizes["k"]
    k_coord = data.coords.get(
        "k",
        np.arange(nk, dtype=np.int64),
    )

    # MITgcm RF is negative downward; use positive depths.
    interfaces = -grid["RF"].values[: nk + 1]

    z_top = xr.DataArray(
        interfaces[:-1],
        dims="k",
        coords={"k": k_coord},
    ).broadcast_like(data)

    z_bottom = xr.DataArray(
        interfaces[1:],
        dims="k",
        coords={"k": k_coord},
    ).broadcast_like(data)

    # Wet thickness of each model layer.
    layer_thickness = (
        (z_bottom - z_top)
        * grid["hFacC"]
    )

    column_depth = layer_thickness.sum("k")

    # Restrict mixed-layer depth to the local water-column depth.
    mld = kpphbl.fillna(0.0).clip(min=0.0)
    mld = xr.where(
        mld < column_depth,
        mld,
        column_depth,
    )

    # Portion of each layer lying inside the mixed layer.
    mld_3d = mld.broadcast_like(data)

    overlap = (
        xr.where(mld_3d < z_bottom, mld_3d, z_bottom)
        - z_top
    ).clip(min=0.0)

    weights = xr.where(
        overlap < layer_thickness,
        overlap,
        layer_thickness,
    )

    numerator = (data * weights).sum("k", skipna=True)
    denominator = weights.where(data.notnull()).sum("k")

    result = (numerator / denominator).where(denominator > 0)

    name = f"{data.name}_mld_mean" if data.name else "mld_mean"
    return result.rename(name)