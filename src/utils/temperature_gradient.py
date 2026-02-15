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

# --------------------- main program ----------------------------------
if __name__ == "__main__":
    variable = 'T_grad'
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