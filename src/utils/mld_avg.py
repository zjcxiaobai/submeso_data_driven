
import xarray as xr
import numpy as np

def mld_avg(
    data: xr.DataArray,  
    kpphbl: xr.DataArray,
    ds_grid: xr.DataArray, 
):
    '''
    '''
    
    # mixed layer average
    Hcol = (-ds_grid["RF"]).isel(k_p1=-1)              # local bottom depth
    H_eff = xr.where(np.isfinite(kpphbl), kpphbl.clip(min=0), 0.0).clip(max=Hcol)
    # layer bounds from RF
    K = data.sizes['k']
    z_top_1d = ((-ds_grid["RF"]).isel(k_p1=slice(0, -1))
              .rename({'k_p1':'k'})
              .reset_coords(drop=True)
              .assign_coords(k=np.arange(K)))             # labels 0..K-1
    z_bot_1d = ((-ds_grid["RF"]).isel(k_p1=slice(1, None))
              .rename({'k_p1':'k'})
              .reset_coords(drop=True)
              .assign_coords(k=np.arange(K)))             # labels 0..K-1
    z_top = z_top_1d.broadcast_like(data)                     # (k,j,i)
    z_bot = z_bot_1d.broadcast_like(data)                     # (k,j,i)
    thk    = (z_bot - z_top) * ds_grid["hFacC"]      # actual layer thickness
    # Overlap thickness with 0..H
    H3 = H_eff.broadcast_like(data)
    w  = xr.apply_ufunc(np.minimum, H3, z_bot) - z_top
    w  = w.clip(min=0, max=thk)   
    # mixed layer average
    num = (data * w).sum("k")
    den = w.sum("k")
    data_mld_mean = xr.where(den > 0, num / den, np.nan)
    return data_mld_mean