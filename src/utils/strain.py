#### total Starin
# the strain is defined by : 
# strain = $\sqrt{(\frac{\partial u}{\partial x}-\frac{\partial v}{\partial y})^2+(\frac{\partial v}{\partial x}+\frac{\partial u}{\partial y})^2}$

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

# ------------------------------ strain function -----------------------
def _total_strain(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill"
) -> xr.DataArray:
    """
    Flux-form strain on coarse grid, following the user's example pattern.
    D_center : xr.DataArray
        D = ∂u/∂x − ∂v/∂y at centers (k,j,i), units s^-1
    S_center : xr.DataArray
        S = ∂v/∂x + ∂u/∂y mapped to centers (k,j,i), units s^-1
    Returns
    -------
    strain   : xr.DataArray
        |S| = sqrt(D^2 + S^2) at centers (k,j,i), units s^-1
    """
    # merge grid + velocities
    ds = xr.merge([ds_grid, ds_ocean])
    # xgcm grid (periodic in X)
    grid = make_grid.make_grid_periodic_x(ds)
    # ---- collocate center metrics to faces ----
    # ---- Volumes / masks at centers & corners ----
    vol_c = ( 
        ds["DRF"] * ds["RAC"]
    ).transpose("k","j","i") * ds["hFacC"]
    wetc = ds["hFacC"] > 0
    # ---- Face-integrated fluxes (m^3/s) using face-coarsened velocities ----
    U = ds["U"].fillna(0.0)     # (k,j,  i_g)
    V = ds["V"].fillna(0.0)     # (k,j_g,i  )
    # ----------------------------------------------------------------------------
    # Normal components → give ∂u/∂x, ∂v/∂y at centers after diff
    Fu_norm = ds["U"] * ds["DYG"] * ds["hFacW"] * ds["DRF"]          # (k,j,  i_g)
    Fv_norm = ds["V"] * ds["DXG"] * ds["hFacS"] * ds["DRF"]          # (k,j_g,i  )
    # ---- Normal derivatives: faces -> centers, then divide by VOLc ----
    dUdx = grid.diff(Fu_norm, "X") / vol_c                               # (k,j,i)
    dVdy = grid.diff(Fv_norm, "Y", boundary=boundary_Y) / vol_c          # (k,j,i)
    
    # Tangential components → give ∂u/∂y, ∂v/∂x after diff (first on corners)
    Fu_tan  = ds["U"] * ds["DXC"] * ds["hFacW"] * ds["DRF"]         # (k,j,  i_g)
    Fv_tan  = ds["V"] * ds["DYC"] * ds["hFacS"] * ds["DRF"]         # (k,j_g,i  )
    # ---- Cross derivatives: faces -> corners, corners -> centers (conservative), then /VOLc ----
    dVdx_G = grid.diff(Fv_tan, "X") / (ds["DRF"] * ds["RAZ"])       # (k,j_g,i_g)
    dUdy_G = grid.diff(Fu_tan, "Y", boundary=boundary_Y) / (ds["DRF"] * ds["RAZ"])     # (k,j_g,i_g)
    # Map corner → center
    dUdy = grid.interp(grid.interp(dUdy_G, "X", to="center"), "Y", to="center")  # (k,j,i)
    dVdx = grid.interp(grid.interp(dVdx_G, "X", to="center"), "Y", to="center")  # (k,j,i)
    # ---- Assemble D, S and |S| at centers ----
    D_center = (dUdx - dVdy).where(wetc).rename("D_center")
    D_center.attrs.update(long_name="∂u/∂x − ∂v/∂y (center, flux-form)", units="s^-1")

    S_center = (dVdx + dUdy).where(wetc).rename("S_center")
    S_center.attrs.update(long_name="∂v/∂x + ∂u/∂y (center, flux-form)", units="s^-1")

    strain = xr.apply_ufunc(np.hypot, D_center, S_center).rename("strain_magnitude").where(wetc)
    strain.attrs.update(long_name="|S| = sqrt[(∂u/∂x−∂v/∂y)^2 + (∂v/∂x+∂u/∂y)^2]", units="s^-1")
    
    return strain

# --------------------- main program ----------------------------------
if __name__ == "__main__":
    variable = 'strain'
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
    nStep1, nStep2, dS = 4945536 + 216 * 4500, 6096384, 216
    time_index = np.arange(nStep1, nStep2 + 1, dS, dtype=int)
    num_data = time_index.size
    print("Total data number is:", num_data)
    
    # initialization
    data = []
    for _ in range(len(time_index)):
        # load velocity data
        print("saved:", _ + 1)
        U = read_mitgcm._ocean_var(grid_dir, "U", time_index[_], ("k","j","i_g"))
        U_coarse = coarsen_UVWTS.coarse_field("U", U, ds_fine, box_j, box_i)
        V = read_mitgcm._ocean_var(grid_dir, "V", time_index[_], ("k","j_g","i"))
        V_coarse = coarsen_UVWTS.coarse_field("V", V, ds_fine, box_j, box_i)
        ds_ocean = xr.Dataset(dict(U=U, V=V))
        ds_ocean_c = xr.Dataset(dict(U=U_coarse, V=V_coarse))

        data_sanpshot = total_strain_coarse(ds_coarse, ds_ocean_c, "fill")
        
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