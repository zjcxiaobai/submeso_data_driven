# --------------------- module ----------------------------------
import os
from src.utils import make_grid
from src.utils import build_coarse_grid
from src.utils import read_mitgcm
import xarray as xr
import numpy as np
from MITgcmutils import mds
import xarray as xr
from src.utils import mld_avg
import gsw
# --------------------- submesoscale heat flux ----------------------------------
def _vertical_heat_flux(
    ds_fine: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Z: str = "fill",
    to_wm2=True, 
):
    """
    nan
    """
    # merge the two dataset
    ds = xr.merge([ds_fine, ds_ocean])
    grid_fine = make_grid.make_grid_periodic_x(ds)

    # high resolution vertical heat flux
    # put T on W-interfaces
    T_w = grid_fine.interp(ds["T"], "Z", to="left", boundary=boundary_Z) # (k_l,j,i)
    WT  = ds["W"] * T_w # (k_l,j,i)
    wT_kC = grid_fine.interp(WT, "Z", to="center",
                            boundary=boundary_Z).rename("wT")

    if to_wm2:
        rho0, cp = 1035.0, 3994.0  # kg m^-3, J kg^-1 K^-1
        wT_kC = (rho0 * cp * wT_kC).assign_attrs(units="W m^-2")
        
    return wT_kC
