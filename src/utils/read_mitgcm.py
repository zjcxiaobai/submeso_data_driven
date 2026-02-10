
from MITgcmutils import mds
import xarray as xr
import numpy as np

# function to read grid variables
def _grid_var(grid_dir, name, dims):
    return xr.DataArray(mds.rdmds(f"{grid_dir}/{name}"), dims=dims, name=name)
    
# function to read ocean variables
def _ocean_var(grid_dir, name, t, dims):
    return xr.DataArray(mds.rdmds(f"{grid_dir}/{name}", t), dims=dims, name=name)
    
# function to read all the grid variables and create a dataset    
def read_mitgrid_staggered(grid_dir: str) -> xr.Dataset:
    """
    read MITgcm grid variables
    MITgcm C-grid staggering:
      Centers (C):   (j,i)      -> 
      U-faces:       (j,i_g)    -> 
      V-faces:       (j_g,i)    -> 
      Corners (G):   (j_g,i_g)  -> 
      Vertical:      (k)        -> 
    """
    # centers (j,i)
    XC  = _grid_var(grid_dir, "XC",  ("j","i")) 
    YC  = _grid_var(grid_dir, "YC",  ("j","i"))
    DXF = _grid_var(grid_dir, "DXF", ("j","i")) # U-U face length
    DYF = _grid_var(grid_dir, "DYF", ("j","i")) # V-V face length
    RAC = _grid_var(grid_dir, "RAC", ("j","i")) # DXG * DYG
    DXV = _grid_var(grid_dir, "DXV", ("j","i"))
    DYU = _grid_var(grid_dir, "DYU", ("j","i"))
    # face & center (j,i_g), (j_g,i)
    DXC = _grid_var(grid_dir, "DXC", ("j","i_g"))
    DYC = _grid_var(grid_dir, "DYC", ("j_g","i"))
    RAW = _grid_var(grid_dir, "RAW", ("j","i_g"))
    RAS = _grid_var(grid_dir, "RAS", ("j_g","i"))
    DXG = _grid_var(grid_dir, "DXG", ("j_g","i"))   # V-face length
    DYG = _grid_var(grid_dir, "DYG", ("j","i_g"))   # U-face length
    
    # face/corner metrics with CORRECT dims (no assign_coords here)
    RAZ = _grid_var(grid_dir, "RAZ", ("j_g","i_g")) # vorticity area  
    XG  = _grid_var(grid_dir, "XG",  ("j_g","i_g"))
    YG  = _grid_var(grid_dir, "YG",  ("j_g","i_g"))
    # vertical & masks
    DRF_3d   = _grid_var(grid_dir, "DRF",   ("k","j","i"))  # k,1,1
    DRF = DRF_3d.isel(j=0, i=0, drop=True)                 # -> (k,)
    DRC_3d = _grid_var(grid_dir, "DRC",   ("k_p1","j","i")) 
    DRC = DRC_3d.isel(j=0, i=0, drop=True)                 # -> (k_p1,)
    RF_3d = _grid_var(grid_dir, "RF",  ("k_p1","j","i"))
    RF = RF_3d.isel(j=0, i=0, drop=True)
    RC_3d = _grid_var(grid_dir, 'RC',("k","j","i"))
    RC = RC_3d.isel(j=0, i=0, drop=True)                 # -> (k,)
    hFacC = _grid_var(grid_dir, "hFacC", ("k","j","i"))
    hFacW = _grid_var(grid_dir, "hFacW", ("k","j","i_g"))  # U faces
    hFacS = _grid_var(grid_dir, "hFacS", ("k","j_g","i"))  # V faces

    # k_l
    k_l = xr.DataArray(np.arange(49, dtype="int64"), dims="k_l", name="k_l")
    
    # build Dataset 
    ds = xr.Dataset(
        dict(
            XC=XC, YC=YC, DXF=DXF, DYF=DYF, RAC=RAC, DXV=DXV, DYU=DYU,
            DXC=DXC, DYC=DYC, RAW=RAW, RAS=RAS, DXG=DXG, DYG=DYG,
            RAZ=RAZ, XG=XG, YG=YG, 
            DRF=DRF, DRC=DRC, RF=RF, RC=RC,
            hFacC=hFacC, hFacW=hFacW, hFacS=hFacS,
            k_l = k_l
        )
    )

    # now attach dimension coordinates ONCE at the Dataset level
    nj, ni = ds.sizes["j"], ds.sizes["i"]
    ds = ds.assign_coords(
        j=np.arange(nj), i=np.arange(ni),
        j_g=np.arange(nj), i_g=np.arange(ni),
        k=np.arange(ds.sizes["k"]),
        k_l=np.arange(ds.sizes["k_l"]),
        k_p1 = np.arange(ds.sizes["k_p1"])
    )
    return ds
