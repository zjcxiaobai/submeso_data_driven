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


# --------------------- buoyancy and buoyancy gradient function ----------------------------------
def buo_coarse(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill"
) -> xr.DataArray:

    # merge the two dataset
    ds = xr.merge([ds_grid, ds_ocean])
    grid = make_grid.make_grid_periodic_x(ds)
    # water collumn
    # Etabar: sea-surface height (j,i) [m]; broadcast add → z_c(k,j,i)
    z_rel = ds["RC"] - ds["Eta"]  # here RC is negative
    
    # TEOS10 EOS
    # model output is salinity and potential temperature
    # gsw function need absolute salinity and conservative temperature
    lat0, lon0 = -55.0, 153.0
    lat2d = xr.full_like(ds["T"].isel(k=0), lat0)               # (j,i)
    lon2d = xr.full_like(ds["T"].isel(k=0), np.mod(lon0, 360.)) # (j,i)

    # TEOS-10 sea pressure from depth (returns dbar)
    p_dbar = xr.apply_ufunc(
        gsw.p_from_z, z_rel, lat2d,
        dask="parallelized",
        output_dtypes=[ds["T"].dtype],
    )


    # --- convert SP, θ -> SA, CT -> ρ ---
    SA  = xr.apply_ufunc(gsw.SA_from_SP, ds["S"], p_dbar, lon2d, lat2d,
                         dask='parallelized', output_dtypes=[ds["S"].dtype])
    CT  = xr.apply_ufunc(gsw.CT_from_pt, SA, ds["T"],
                         dask='parallelized', output_dtypes=[ds["T"].dtype])
    rho = xr.apply_ufunc(gsw.rho, SA, CT, p_dbar,
                         dask='parallelized', output_dtypes=[ds["T"].dtype])  # kg m^-3
    # constants
    g = 9.81 # gravity
    rho0 = 1035 # density
    # buoyancy
    b = (-g * (rho - rho0) / rho0).where(ds["hFacC"] > 0)  # m s^-2
    b.name = "buoyancy"
    b.attrs.update(units="m s^-2", long_name="buoyancy (TEOS-10)")
    
    # ---- buoyancy gradient ----
    # Face wet masks (avoid bleeding across land/blocked faces)
    wetU = ds["hFacW"] > 0
    wetV = ds["hFacS"] > 0
    dbdx_u = (grid.diff(b, "X") / ds["DXC"]).where(wetU)                   # (k,j,i_g)
    dbdy_v = (grid.diff(b, "Y", boundary=boundary_Y) / ds["DYC"]).where(wetV)  # (k,j_g,i)
    # Conservatively bring face-gradients to centers (weight by face wetness)
    # X: average adjacent U-faces; Y: average adjacent V-faces
    wU_c = grid.interp(wetU.astype(b.dtype), "X")  # (k,j,i) in {0,0.5,1}
    wV_c = grid.interp(wetV.astype(b.dtype), "Y")  # (k,j,i)

    dbdx_c_num = grid.interp(dbdx_u.fillna(0.0), "X")
    dbdy_c_num = grid.interp(dbdy_v.fillna(0.0), "Y")

    dbdx_c = xr.where(wU_c > 0, dbdx_c_num / xr.where(wU_c > 0, wU_c, 1.0), np.nan)
    dbdy_c = xr.where(wV_c > 0, dbdy_c_num / xr.where(wV_c > 0, wV_c, 1.0), np.nan)

    # Center wet mask for final fields
    wetC = ds["hFacC"] > 0
    dbdx_c = dbdx_c.where(wetC)
    dbdy_c = dbdy_c.where(wetC)

    # Magnitude |∇b| = sqrt((∂b/∂x)^2 + (∂b/∂y)^2)
    b_grad = xr.apply_ufunc(np.hypot, dbdx_c, dbdy_c).where(wetC)

    # --- nice names/attrs ---
    b_grad.name = "buoyancy_gradient"
    b_grad.attrs.update(units="s^-2", long_name="|∇b| at centers")

    return b, b_grad

# --------------------- main program ----------------------------------
#  change variable to 'b_grad' for buoyancy gradient and change data_sanpshot_1 to data_sanpshot_2
if __name__ == "__main__":
    variable = 'b'
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
        print("saved:", _ + 1)
        T = read_mitgcm._ocean_var(grid_dir, "T", time_index[_], ("k","j","i"))
        T_coarse = coarsen_UVWTS.coarse_field("T", T, ds_fine, box_j, box_i)
        S = read_mitgcm._ocean_var(grid_dir, "S", time_index[_], ("k","j","i"))
        S_coarse = coarsen_UVWTS.coarse_field("S", S, ds_fine, box_j, box_i)
        Eta = read_mitgcm._ocean_var(grid_dir, "Eta", time_index[_], ("j","i"))
        Eta_coarse = coarsen_UVWTS.coarse_field("Eta", Eta, ds_fine, box_j, box_i)
        
        ds_ocean = xr.Dataset(dict(T=T, S=S, Eta=Eta))
        ds_ocean_c = xr.Dataset(dict(T=T_coarse, S=S_coarse, Eta=Eta_coarse))
        [data_sanpshot_1, data_sanpshot_2] = buo_coarse(ds_coarse, ds_ocean_c, "fill")

        #  mixed layer avergae
        diagMXL = mds.rdmds(path_scratch + path_MITgcm + 'Diags/' +'DiagMXL_2d', time_index[_])
        kpphbl = xr.DataArray(diagMXL[1, :,:], dims = ("j","i"), name = "kpphbl")
        kpphbl_coarse = coarsen_UVWTS.coarse_field("kpphbl", kpphbl, ds_fine, box_j, box_i)
        data_coarse_mld = mld_avg.mld_avg(data_sanpshot_1, kpphbl_coarse, ds_coarse)
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