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

# --------------------- submesoscale heat flux ----------------------------------
def submeso_vertical_heat_flux(
    ds_fine: xr.Dataset,
    ds_coarse: xr.Dataset,
    ds_ocean: xr.Dataset,
    ds_ocean_c: xr.Dataset,
    boundary_Z: str = "fill",
    box_j: int = 10,
    box_i: int = 10,
    to_wm2=True, 
):
    """
    Compute ⟨w'T'⟩ = ⟨wT⟩ − ⟨w⟩⟨T⟩ using 50x50 box averages (default).
    --------
    theta[k,j,i]: xarray.DataArray
        Potential temperature (K) on C-grid cell centers
    w[k,j,i]: xarray.DataArray
        Vertical velocity (m s^-1). If defined on vertical faces (k+1/2)
    hFacC[k,j,i]: 
        faction of the vertical thickness of the cell at level k, C means C-grid cell center
        1.0 - full wet, 
        0.0 - entirely land
        
    time_index: time point of variables
    box: spatial average box, km
    to_wm2: change the unit to wm2
    check_xarray: check if the data type is xarray
    
    Returns:
      wT_sub  : submesoscale vertical advective heat flux           (K·m s^-1, or W m^-2 if to_wm2=True)
      wT_bar  : coarse ⟨wT⟩                                         (same units)
      w_bar   : coarse ⟨w⟩                                          (m s^-1)
      T_bar   : coarse ⟨T⟩                                          (K)
    Notes:
      - Uses mask from hFacC; averages are NaN-skipping (equiv. to equal-area weighting on your uniform grid).
      - If W is on vertical faces (k+1/2), we interpolate T to those faces before forming wT.
    """
    # merge the two dataset
    ds = xr.merge([ds_fine, ds_ocean])
    grid_fine = make_grid.make_grid_periodic_x(ds)
    grid_coarse = make_grid.make_grid_periodic_x(ds_coarse)
    # ---------- Build open interface area Aface (k_l,j,i) ----------
    # open_frac(k_l) = min(hFacC_k, hFacC_{k+1})
    upper = ds["hFacC"]                                  # (k,j,i)
    lower = ds["hFacC"].shift(k=-1, fill_value=0.0)      # (k+1,j,i)
    open_frac = xr.ufuncs.minimum(upper, lower).rename({'k': 'k_l'})  # (k_l,j,i)
    Aface = (open_frac * ds["RAC"]).astype('float32')                 # (k_l,j,i)
    # Valid interface mask (avoid coast/land)
    wet_w = Aface > 0

    # high resolution vertical heat flux
    # put T on W-interfaces
    T_w = grid_fine.interp(ds["T"], "Z", to="left", boundary=boundary_Z) # (k_l,j,i)
    WT  = ds["W"] * T_w # (k_l,j,i)

    # ---------- Helper: area-weighted coarse mean on (j,i) ----------
    def _area_weighted_boxmean(A: xr.DataArray, WGT: xr.DataArray, box_j = 10, box_i=10 ) -> xr.DataArray:
        num = (A * WGT).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        den = (WGT   ).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        return xr.where(den > 0, num/den, np.nan)
    def as_ordinal(da: xr.DataArray) -> xr.DataArray:
        """Drop hitchhiking coords and replace each dim’s coordinate with 0..N-1."""
        da = da.reset_coords(drop=True)
        new_coords = {d: xr.DataArray(np.arange(da.sizes[d]), dims=d) for d in da.dims}
        return da.assign_coords(**new_coords)
    
    # ---------- Coarse means at k_l (interface placement) ----------
    wT_bar_kL = _area_weighted_boxmean(WT.where(wet_w),   Aface)
    w_bar_kL = (ds_ocean_c["W"])
    T_bar_kC = (ds_ocean_c["T"])
    T_bar_kL = grid_coarse.interp(T_bar_kC, "Z", to="left", boundary=boundary_Z) # (k_l,j,i))
    wT_bar_kL = as_ordinal(wT_bar_kL)
    meso_wT_KL = w_bar_kL * T_bar_kL
    # ---------- Submeso covariance at interfaces ----------
    wT_sub_kL = (wT_bar_kL - meso_wT_KL).rename("wT_sub_kL")
    wT_bar_kL = wT_bar_kL.rename("wT_bar_kL")
    w_bar_kL  = w_bar_kL.rename("w_bar_kL")
    T_bar_kL  = T_bar_kL.rename("T_bar_kL")

    # ---------- Optional: move to centers in Z (for plotting/MLD averages) ----------
    # Interpolate in vertical from k_l → k AFTER coarse-graining
    wT_sub_kC = grid_coarse.interp(wT_sub_kL, "Z", to="center",
                            boundary=boundary_Z).rename("wT_sub_kC")
    wT_bar_kC = grid_coarse.interp(wT_bar_kL, "Z", to="center",
                            boundary=boundary_Z).rename("wT_sub_kC")
    meso_wT_KC = grid_coarse.interp(meso_wT_KL, "Z", to="center",
                            boundary=boundary_Z).rename("wT_sub_kC")

    # Attach coarse coords from ds_grid if present
    if {'j','i'}.issubset(ds_coarse.coords):
        for da in (wT_sub_kL, wT_bar_kL, w_bar_kL, T_bar_kL, wT_sub_kC):
            # coarsen(...) already produced (j,i) reduced dims; enforce coarse coords
            if 'j' in da.dims: da = da.assign_coords(j=ds_coarse['j'])
            if 'i' in da.dims: da = da.assign_coords(i=ds_coarse['i'])
        # rebind names (xarray returns new objects when assign_coords)
        wT_sub_kL, wT_bar_kL, w_bar_kL, T_bar_kL, wT_sub_kC = [x for x in (wT_sub_kL,wT_bar_kL, w_bar_kL, T_bar_kL, wT_sub_kC)]

    # ---------- Units: convert to W m^-2 if requested ----------
    if to_wm2:
        rho0, cp = 1035.0, 3994.0  # kg m^-3, J kg^-1 K^-1
        meso_wT_KC = (rho0 * cp * meso_wT_KC).assign_attrs(units="W m^-2")
        wT_bar_kC = (rho0 * cp * wT_bar_kC).assign_attrs(units="W m^-2")
        wT_sub_kC = (rho0 * cp * wT_sub_kC).assign_attrs(units="W m^-2")
    else:
        for da in (wT_sub_kL, wT_bar_kL, wT_sub_kC):
            da.attrs['units'] = "K m s^-1"

    # Some helpful long_names
    wT_sub_kL.attrs['long_name'] = "submesoscale vertical heat flux at interfaces (area-weighted coarse)"
    wT_bar_kL.attrs['long_name'] = "coarse mean of wT at interfaces (area-weighted)"
    w_bar_kL.attrs['long_name']  = "coarse mean of w at interfaces (area-weighted)"
    T_bar_kL.attrs['long_name']  = "coarse mean of T (interp to interfaces) (area-weighted)"
    wT_sub_kC.attrs['long_name'] = "submesoscale vertical heat flux at centers (interp from interfaces)"

    return wT_sub_kC, wT_bar_kC, meso_wT_KC


# --------------------- main program ----------------------------------
if __name__ == "__main__":
    variable = 'submeso_wT'
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
    # pickup start point: 3900096, end: 4945536, end: 53913606096384
    nStep1, nStep2, dS = 3900096 + 216 * 9000, 6096384, 216
    time_index = np.arange(nStep1, nStep2 + 1, dS, dtype=int)
    num_data = time_index.size
    print("Total data number is:", num_data)

    # initialization
    data = []
    for _ in range(len(time_index)):
        # load velocity data
        print("saved:", _ + 1)

        W = read_mitgcm._ocean_var(grid_dir, "W", time_index[_], ("k_l","j","i"))
        W_coarse = coarsen_UVWTS.coarse_field("W", W, ds_fine, box_j, box_i)
        T = read_mitgcm._ocean_var(grid_dir, "T", time_index[_], ("k","j","i"))
        T_coarse = coarsen_UVWTS.coarse_field("T", T, ds_fine, box_j, box_i)
        
        ds_ocean = xr.Dataset(dict(W=W, T=T))
        ds_ocean_c = xr.Dataset(dict(W=W_coarse, T=T_coarse))
        
        data_sanpshot = submeso_vertical_heat_flux(ds_fine, ds_coarse, ds_ocean, ds_ocean_c, box_j = box_j, box_i = box_i,boundary_Z = "fill", to_wm2 = True)
        
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
