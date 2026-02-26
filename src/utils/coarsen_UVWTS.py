from src.utils import make_grid
from src.utils import build_coarse_grid
from src.utils import read_mitgcm
import xarray as xr
import numpy as np

def coarse_field(varname: str,
                 da: xr.DataArray,
                 ds_fine: xr.Dataset,     # fine static grid 
                 box_j: int,
                 box_i: int) -> xr.DataArray:
    """
    varname:str
    da: xarray, high-resolution data
    ds_fine: xr dataset, high-resolution grid variables
    box_j, box_i: average box in two dimensions
    
    Return a coarse snapshot of one variable at time index `t`.
    Supported varname: 'U','V','W','Theta','Salt','T','S','kpphbl'
    """
    # make varname lower case
    var = varname.lower()
    
    # weights / metrics (fine)
    RAC  = ds_fine['RAC']                          # (j,i)
    DRF  = ds_fine['DRF']                          # (k)
    hFacC = ds_fine['hFacC']                       # (k,j,i)
    hFacW = ds_fine['hFacW']                       # (k,j,i_g)
    hFacS = ds_fine['hFacS']                       # (k,j_g,i)
    DXG  = ds_fine['DXG']                          # (j_g,i)
    DYG  = ds_fine['DYG']                          # (j,i_g)
    # --- helper ---
    def as_ordinal(da: xr.DataArray) -> xr.DataArray:
        """Drop hitchhiking coords and replace each dim’s coordinate with 0..N-1."""
        da = da.reset_coords(drop=True)
        new_coords = {d: xr.DataArray(np.arange(da.sizes[d]), dims=d) for d in da.dims}
        return da.assign_coords(**new_coords)
    
    # ================= U =================
    if var in ('u',):
        # check the dimension
        if tuple(da.dims) != ('k','j','i_g'):
            raise ValueError("U must have dims ('k','j','i_g').")
        WU = (hFacW * DYG)  # face length (per unit height)
        U = da.where(WU > 0, 0.0).fillna(0.0) # set WU<=0 to 0, and set nan to 0
        # conservative average along j, then subsample i_g to coarse faces
        num = (U * WU).coarsen(j=box_j, boundary="trim").sum()
        den = (WU).coarsen(j=box_j, boundary="trim").sum()
        U_avgj = xr.where(den > 0, num / den, 0.0) # if den > 0, num/den, if not, set 0
        U_coarse = U_avgj.isel(i_g=slice(0, None, box_i))
        U_coarse = as_ordinal(U_coarse) # normalize the dimension
        return U_coarse

    # ================= V =================
    if var in ('v',):
        if tuple(da.dims) != ('k','j_g','i'):
            raise ValueError("V must have dims ('k','j_g','i').")
        WV = (hFacS * DXG)
        V = da.where(WV > 0, 0.0).fillna(0.0)
        num = (V * WV).coarsen(i=box_i, boundary="trim").sum()
        den = (WV    ).coarsen(i=box_i, boundary="trim").sum()
        V_avgi = xr.where(den > 0, num / den, 0.0)
        V_coarse = V_avgi.isel(j_g=slice(0, None, box_j))
        V_coarse = as_ordinal(V_coarse) # normalize the dimension
        return V_coarse

    # ================= W =================
    if var in ('w',):
        if tuple(da.dims) != ('k_l','j','i'):
            raise ValueError("W must have dims ('k_l','j','i').")
        # open area of vertical interface between layers at k_l:
        # use min(hFacC(k), hFacC(k+1)) to represent open fraction on the face
        upper = ds_fine.hFacC                                  # (k,j,i)
        lower = ds_fine.hFacC.shift(k=-1, fill_value=0.0)      # (k+1,j,i), bottom filled as 0 (closed)
        open_frac = xr.ufuncs.minimum(upper, lower).rename({'k': 'k_l'})  # (k_l,j,i), same length as k
        Aface = (open_frac * RAC).astype('float32')
        # Conservative horizonatl coarsening: average vertical transport, then divide by open area
        num = (da.where(Aface > 0, 0.0).fillna(0.0) * Aface).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        den = Aface.coarsen(j=box_j, i=box_i, boundary="trim").sum()
        W_coarse  = xr.where(den > 0, num / den, np.nan)
        W_coarse = as_ordinal(W_coarse)
        return W_coarse

    # ================= Theta / Salt =================
    if var in ('theta', 't', 'salt', 's'):
        if tuple(da.dims) != ('k','j','i'):
            raise ValueError("Theta/Salt must have dims ('k','j','i').")
        # Volume-weighted horizontal coarsening per level (preserves tracer content)
        w_vol = (hFacC * RAC * DRF)        # (k,j,i)
        num = (da * w_vol).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        den = (w_vol    ).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        Xc  = xr.where(den > 0, num / den, np.nan)
        Xc = as_ordinal(Xc)
        return Xc

    # ================= kpphbl =================
    if var in ('kpphbl','mld','hbl'):
        if tuple(da.dims) != ('j','i'):
            raise ValueError("kpphbl must have dims ('j','i').")
        wet_surf = (hFacC.isel(k=0) > 0)
        Warea = xr.where(wet_surf, RAC, 0.0)
        num = (da * Warea).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        den = (Warea    ).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        Hc  = xr.where(den > 0, num/den, np.nan)
        Hc = as_ordinal(Hc)
        return Hc

    # ================= Sea surface height =================
    if var in ('eta', 'ssh', 'etabar'):
        # Expect a single-level surface field at centers
        if tuple(da.dims) != ('j','i'):
            raise ValueError("Eta/SSH must have dims ('j','i').")
        # surface-wet mask (only average over ocean)
        wet_surf = (hFacC.isel(k=0) > 0)
        Warea = xr.where(wet_surf, RAC, 0.0)  # (j,i)
        # area-weighted coarse average
        num = (da * Warea).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        den = (Warea   ).coarsen(j=box_j, i=box_i, boundary="trim").sum()
        Eta_coarse = xr.where(den > 0, num / den, np.nan)
        Eta_coarse = as_ordinal(Eta_coarse)
        return Eta_coarse
        
    raise ValueError(f"Unsupported varname: {varname!r}")

if __name__ == "__main__":
    #  MITgcm path
    path_scratch = '/scratch/jczhang/'
    path_MITgcm = 'MITgcm_5km/MITgcm/verification/tutorial_reentrant_channel/run/'
    grid_dir = path_scratch + path_MITgcm
    
    # get the grid variables and create the ds
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_dir)
    ds_grid = build_coarse_grid.build_coarse_grid_staggered(ds_fine,box_j=10, box_i=10)
    U = read_mitgcm._ocean_var(grid_dir, "U", time_index[_], ("k","j","i_g"))
    Ubar = coarse_field("U", U, ds_fine, ds_grid, 10, 10)