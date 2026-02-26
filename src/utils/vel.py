# --------------------- module ----------------------------------
import os
from src.utils import make_grid
from src.utils import build_coarse_grid
from src.utils import read_mitgcm
import xarray as xr
import numpy as np
from src.utils import coarsen_UVWTS
from MITgcmutils import mds
import xarray as xr
from src.utils import mld_avg
import gsw
import importlib
importlib.reload(coarsen_UVWTS)   # force reload
# --------------------- horizontal velocity ----------------------------------
def _horizontal_speed(    
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill"
) -> xr.DataArray:
    
    # merge the two dataset
    ds = xr.merge([ds_grid, ds_ocean])
    grid = make_grid.make_grid_periodic_x(ds)
    # zero closed faces
    U = ds["U"].where(ds["hFacW"] > 0)
    WU = ds["hFacW"] * ds["DYG"]  # face length x wetness 
    V = ds["V"].where(ds["hFacS"] > 0)
    WV = ds["hFacS"] * ds["DXG"] # face length x wetness 

    # calculate the center velocity in both direction
    # periodic in i direction
    U_L  = U # west face of cell i
    U_R  = U.roll(i_g=-1, roll_coords=False)  # east face of cell i, the last face will be the first face, coor will not change
    WU_L = WU
    WU_R = WU.roll(i_g=-1, roll_coords=False)
    Uc_num = U_L * WU_L + U_R * WU_R # weighted average flux
    Uc_den = WU_L + WU_R
    Uc = xr.where(Uc_den > 0, Uc_num / Uc_den, np.nan)  # (k, j, i_g)
    Uc     = Uc.rename({"i_g": "i"})  # center
    # solid wall in j direction
    V_S  = V  # south face
    V_N  = V.shift(j_g=-1, fill_value=np.nan)  # north face
    WV_S = WV
    WV_N = WV.shift(j_g=-1, fill_value=0.0)
    Vc_num = V_S * WV_S + V_N * WV_N
    Vc_den = WV_S + WV_N
    Vc     = xr.where(Vc_den > 0, Vc_num / Vc_den, np.nan).rename({"j_g": "j"})  # (k, j, i)

    # calculate speed
    hasU_face = (Uc_den > 0) 
    hasV_face = (Vc_den > 0)
    hasU = hasU_face.rename({"i_g": "i"})
    hasV = hasV_face.rename({"j_g": "j"})    
    has_face = hasU | hasV  
    vel = xr.apply_ufunc(np.hypot, Uc, Vc).where(ds["hFacC"] > 0).where(has_face)
    vel.name = "speed_coarse_center"
    vel.attrs.update(long_name="|u_h| from face-coarsened U/V, centered (periodic-x, wall-y)",
                 units="m s^-1")
    return vel
# --------------------- main program ----------------------------------
if __name__ == "__main__":
    variable = 'vel'
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
    # pickup start point: 3900096, end: 4945536, end: 5391360
    nStep1, nStep2, dS = 3900096, 4945536, 216
    time_index = np.arange(nStep1, nStep2 + 1, dS, dtype=int)
    num_data = time_index.size
    print("Total data number is:", num_data)

    # initialization
    data = []
    for _ in range(len(time_index)):
        # load velocity data
        U = read_mitgcm._ocean_var(grid_dir, "U", time_index[_], ("k","j","i_g"))
        U_coarse = coarsen_UVWTS.coarse_field("U", U, ds_fine, box_j, box_i)
        V = read_mitgcm._ocean_var(grid_dir, "V", time_index[_], ("k","j_g","i"))
        V_coarse = coarsen_UVWTS.coarse_field("V", V, ds_fine, box_j, box_i)
        ds_ocean = xr.Dataset(dict(U=U, V=V))
        ds_ocean_c = xr.Dataset(dict(U=U_coarse, V=V_coarse))
        
        data_sanpshot = vel_coarse(ds_coarse, ds_ocean_c, "fill")
        
        #  mixed layer avergae
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