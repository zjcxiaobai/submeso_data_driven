# codes for calculate the hortizontal temperature gradient
# --------------------- modules ----------------------------------
import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from src.utils import make_grid
from src.utils import build_coarse_grid
from src.utils import read_mitgcm
import xarray as xr
import numpy as np
from src.utils import coarsen_UVWTS
from MITgcmutils import mds
from src.utils import mld_avg
# --------------------- temperature gradient function ----------------------------------
def T_grad(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill"
) -> xr.DataArray:
    """
    input the coordinate and ocean dataset
    """
    # Merge grid + coarse velocities (expects Ubar, Vbar in ds_ocean)
    ds = xr.merge([ds_grid, ds_ocean])
    grid = make_grid.make_grid_periodic_x(ds)
    # Edge-length-weighted face velocities (m^2/s)
    dTdx_faces = grid.diff(ds.T, 'X') / (ds.DXC)   # (k, j, i_g)
    Tx = grid.interp(dTdx_faces, 'X')  # (k, j, i)
    dTdy_faces = grid.diff(ds.T, 'Y') / (ds.DYC)       # (k, j_g, i)
    Ty = grid.interp(dTdy_faces, 'Y')                                  # (k, j, i)
    # mask land
    wet = ds.hFacC > 0
    Tx = Tx.where(wet)
    Ty = Ty.where(wet)
    gradT = np.hypot(Tx, Ty)
    Tx.name = 'dTdx'; Ty.name = 'dTdy'; gradT.name = 'gradT'
    # add light metadata
    for A in (Tx, Ty, gradT):
        A.attrs.update({'units': 'K m^-1', 'long_name': A.name})   
    return gradT