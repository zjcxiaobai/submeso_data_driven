import numpy as np
import xarray as xr

def rossby_number(
    ds_var: xr.DataArray,
    ds_grid: xr.DataArray,
    f0: float,
    beta: float,
    y_ref: float | None = None,
) -> xr.DataArray:
    """
    Rossby number Ro = zeta / f  (dimensionless)

    Inputs
    -------
    ds_var : xr.DataArray
        include the vorticity
    ds_grid : xr.DataArray
        include the coordinate variables
    f0, beta : float
        Beta-plane parameters: f(y) = f0 + beta*(y - y_ref), units s^-1 and (m^-1 s^-1).
    y_ref : float or None
        Reference latitude (in meters of YC coordinates). If None, use domain mean of YC.

    Returns
    -------
    Ro : xr.DataArray
        Rossby number, same dims as zeta_center.
    """
    zeta = ds_var["zeta"]
    YC = ds_grid["YC"]
    hFacC = ds_grid["hFacC"]
    # wet mask
    wet = (hFacC > 0)
    zeta = zeta.where(wet)
    
    YC_da = xr.DataArray(YC, dims=("j", "i"))
    
    # use a 2D wet mask for defining y_N
    wet2d = wet
    if set(wet2d.dims) - {"j", "i"}:
        wet2d = wet2d.isel({d: 0 for d in wet2d.dims if d not in ("j", "i")})
    
    # north boundary reference y (max y among wet points)
    yN = float(YC_da.where(wet2d).max())
    
    # beta-plane, anchored so f(yN) = f0
    f2d = (f0 + beta * (YC_da - yN)).where(wet2d)  # (j,i)
    
    # broadcast to zeta dims
    _, f3d = xr.broadcast(zeta, f2d)
    
    eps = 1e-12
    Ro = (zeta / f3d).where(np.abs(f3d) > eps)

    Ro.name = "Ro"
    Ro.attrs["long_name"] = "Rossby number"
    Ro.attrs["units"] = "1"
    Ro.attrs["definition"] = "Ro = zeta / f, with beta-plane f(y)=f0+beta*(y-y_ref)"
    Ro.attrs["f0"] = f0
    Ro.attrs["beta"] = beta
    Ro.attrs["y_ref"] = y_ref

    return Ro


