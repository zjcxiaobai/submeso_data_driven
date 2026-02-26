import numpy as np
import xarray as xr
from src.utils import make_grid


def isotherm_slope(
    ds_grid: xr.Dataset,
    ds_var: xr.Dataset,
    boundary_Z: str = "fill",
    eps: float = 1e-12,
):
    """
    Bathymetry-aware zonal mean using wet-area weights.

    Inputs
    ------
    ds_grid: contains hFacC (k,j,i) and RAC (j,i) and optionally coords
    ds_var : contains T (time,k,j,i) or (k,j,i)

    Returns
    -------
      Tbar(k,j)  : time+zonal mean temperature (wet-area weighted over i)
      dTdy(k,j)  : meridional gradient of Tbar
      dTdz(k,j)  : vertical gradient of Tbar
      slope(k,j) : - (dTdy / dTdz)
    """
    # merge so T inherits coords/metrics from grid if needed
    ds = xr.merge([ds_grid, ds_var])
    T = ds["T"]

    # ---- 1) time mean (simple mean is fine; bathy doesn't change in time) ----
    if "time" in T.dims:
        Tt = T.mean(dim="time")
    else:
        Tt = T

    # ---- 2) bathymetry-aware zonal mean over i using wet-area weights ----
    # weights: wet fraction * horizontal cell area
    if "hFacC" not in ds or "RAC" not in ds:
        raise ValueError("ds_grid must contain hFacC (k,j,i) and RAC (j,i) for bathy-aware mean.")

    W = (ds["hFacC"] * ds["RAC"]).astype("float64")   # (k,j,i)
    # mask out land completely
    W = W.where(W > 0)

    # weighted zonal mean at each (k,j)
    num = (Tt * W).sum(dim="i", skipna=True)
    den = W.sum(dim="i", skipna=True)
    Tbar = xr.where(den > 0, num / den, np.nan).rename("Tbar")

    # ---- 3) gradients (note: differentiate('j') assumes 'j' coord is physical) ----
    dTdy = Tbar.differentiate("j").rename("dTdy")

    try:
        dTdz = Tbar.differentiate("k")
    except Exception:
        dTdz = None

    if dTdz is None:
        raise ValueError("Could not compute dTdz by differentiate('k'). Consider using xgcm + drF/delR.")

    dTdz = dTdz.rename("dTdz")

    slope = (-dTdy / xr.where(np.abs(dTdz) > eps, dTdz, np.sign(dTdz) * eps)).rename("isotherm_slope")
    slope.attrs["long_name"] = "Isotherm slope proxy: - (dT/dy) / (dT/dz)"
    slope.attrs["notes"] = "Computed from wet-area weighted zonal mean (accounts for bathymetry via hFacC*RAC)."

    return Tbar, dTdy, dTdz, slope

