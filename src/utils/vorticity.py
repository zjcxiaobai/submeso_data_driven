# codes for calculate the vorticity - match the vorticity.py
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
# --------------------- vorticity function ----------------------------------
# conservative vorticity (ζ = ∂v/∂x − ∂u/∂y) on coarse grid
def _vorticity(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill"
) -> xr.DataArray:
    """
    ζ (s^-1) on the coarse cell centers (k,j,i), using line-integral form:

      Γ_cell = d/dx [ V̄ * DXG * hFacS ]  −  d/dy [ Ū * DYG * hFacW ]
      ζ      = Γ_cell / RAC

    Notes
    -----
    • Ū is on U-faces (j,   i_g) and uses DYG, hFacW
    • V̄ is on V-faces (j_g, i  ) and uses DXG, hFacS
    • No DRF in horizontal curl (purely horizontal circulation per area)
    • X is periodic; Y boundary handled by `boundary_Y` (usually "fill" for walls)
    """
    # Merge grid + coarse velocities (expects Ubar, Vbar in ds_ocean)
    ds = xr.merge([ds_grid, ds_ocean])
    grid = make_grid.make_grid_periodic_x(ds)
    # Edge-length-weighted face velocities (m^2/s)
    maskS = (ds["hFacS"] > 0)   # (k, j_g, i)
    maskW = (ds["hFacW"] > 0)   # (k, j,   i_g)
    V_eff = xr.where(maskS, ds["V"], 0.0)
    U_eff = xr.where(maskW, ds["U"], 0.0)
    # flux
    v_lenflux = V_eff * ds["DYC"]      # (… , k, j_g, i)
    u_lenflux = U_eff * ds["DXC"]      # (… , k, j,   i_g)
    # Circulation tendency to cell centers via finite differences
    dVdx = grid.diff(v_lenflux, "X")                       # (… , k, j_g, i_g)
    dUdy = grid.diff(u_lenflux, "Y", boundary=boundary_Y)  # (… , k, j_g, i_g)
    wet  = ds["hFacC"] > 0  
    # Stokes: ζ = (dVdx - dUdy) / RAZ   (RAC is center cell area)
    zeta_G = (dVdx - dUdy) / ds["RAZ"]
    zeta = grid.interp(grid.interp(zeta_G, "X", to="center"), "Y", to="center")
    zeta.name = "vorticity_coarse_fluxform"
    zeta.attrs.update(long_name="ζ (coarse, flux-form, conservative)", units="s^-1")
    return zeta
# --------------------- main program ----------------------------------
if __name__ == "__main__":
    variable = 'zeta'
    # average box
    box_i = box_j = 10
    # --------------------- MITgcm folder path ----------------------------------
    path_scratch = '/scratch/jczhang/'
    path_MITgcm = 'MITgcm_5km/MITgcm/verification/tutorial_reentrant_channel/run/'
    grid_dir = path_scratch + path_MITgcm
    
    # --------------------- grid variable ----------------------------------
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine,box_j=10, box_i=10)
    
    # --------------------- time variable ----------------------------------
    # time parameters
    # pickup start point: 3900096, end: 4945536, end: 5391360, end: 5577984
    nStep1, nStep2, dS = 4945536, 5391360, 216
    time_index = np.arange(nStep1, nStep2 + 1, dS, dtype=int)
    num_data = time_index.size
    print("Total data number is:", num_data)
        
    # initialization
    data = []

     # --------------------- data calculation ----------------------------------
    for _ in range(len(time_index)):
        # load velocity data
        print("saved:", _ + 1)
        U = read_mitgcm._ocean_var(grid_dir, "U", time_index[_], ("k","j","i_g"))
        U_coarse = coarsen_UVWTS.coarse_field("U", U, ds_fine, box_j, box_i)
        V = read_mitgcm._ocean_var(grid_dir, "V", time_index[_], ("k","j_g","i"))
        V_coarse = coarsen_UVWTS.coarse_field("V", V, ds_fine, box_j, box_i)
        ds_ocean = xr.Dataset(dict(U=U, V=V))
        ds_ocean_c = xr.Dataset(dict(U=U_coarse, V=V_coarse))

        data_sanpshot = vorticity_coarse(ds_coarse, ds_ocean_c, "fill")
        
        # kpphbl
        diagMXL = mds.rdmds(path_scratch + path_MITgcm + 'Diags/' +'DiagMXL_2d', time_index[_])
        kpphbl = xr.DataArray(diagMXL[1, :,:], dims = ("j","i"), name = "kpphbl")
        kpphbl_coarse = coarsen_UVWTS.coarse_field("kpphbl", kpphbl, ds_fine, box_j, box_i)
        data_coarse_mld = mld_avg.mld_avg(data_sanpshot, kpphbl_coarse, ds_coarse)
        data.append(data_coarse_mld[2:118,1:59])
    data_arr = np.stack(data, axis=0)
    
    # --------------------- save the data ----------------------------------
    out_dir = path_scratch + 'data_5km/var_6000/'
    os.makedirs(out_dir, exist_ok=True)
    save_path_npz = os.path.join(out_dir, variable + '.npz')
    
    # if existing file, merge by time_index (new values win), then sort by time
    if os.path.exists(save_path_npz):
        with np.load(save_path_npz) as npz:
            t_old = npz['time_index'].astype(np.int64)
            x_old = npz['data'].astype(np.float32)
        t_cat = np.concatenate([t_old, time_index])
        x_cat = np.concatenate([x_old, data_arr], axis=0)

        # prefer NEW values on duplicates: unique over reversed indices
        idx_rev = np.arange(t_cat.size - 1, -1, -1)
        t_rev = t_cat[idx_rev]
        _, first_rev_idx = np.unique(t_rev, return_index=True)
        keep_rev = np.sort(first_rev_idx)
        keep = idx_rev[keep_rev]

        t_keep = t_cat[keep]
        x_keep = x_cat[keep]
        order = np.argsort(t_keep)
        t_final = t_keep[order]
        x_final = x_keep[order]
    else:
        # first time save
        order = np.argsort(time_index)
        t_final = time_index[order]
        x_final = data_arr[order]

    # save both arrays together
    np.savez_compressed(save_path_npz, time_index=t_final.astype(np.int64), data=x_final.astype(np.float32))
    print('Saved (appended) to', save_path_npz, 'dims:', x_final.shape, 'timesteps:', t_final.size)

    # --------------------- check NANs ---------------------------------
    has_nan = np.isnan(x_final).any()
    num_nan = int(np.isnan(x_final).sum())
    print("Contains NaN:", has_nan)
    print("Number of NaNs:", num_nan)