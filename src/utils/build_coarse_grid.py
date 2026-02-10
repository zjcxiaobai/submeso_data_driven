import xarray as xr
import numpy as np
# make coarse Grid
#  build coarse grid staggered
def build_coarse_grid_staggered(ds_fine: xr.Dataset, box_j: int, box_i: int) -> xr.Dataset:
    """
    Downsample metrics/coords/masks to a coarse grid with correct staggering.

    Centers (j,i):   DXC,DYC,RAC -> SUM ;  XC,YC -> MEAN
    V-faces (j_g,i): DXG         -> SUM
    U-faces (j,i_g): DYG         -> SUM
    Corners (j_g,i_g): XG,YG     -> MEAN
    Masks weighted to preserve volumes/face areas.
    DRF is (k,) and unchanged.
    """
    

    # ---- centers (j,i) ----
    RAC = ds_fine["RAC"].coarsen(j=box_j, i=box_i, boundary="trim").sum().rename("RAC") # sum to conserve area
    # face area: sum to conserve the transport capacity
    RAW = ds_fine["RAW"].coarsen(j=box_j, i_g=box_i, boundary="trim").sum().rename("RAW")
    RAS = ds_fine["RAS"].coarsen(j_g=box_j, i=box_i, boundary="trim").sum().rename("RAS")
    
    # DXG DYG
    j0, i0 = 0, 0
    DXG_sum = ds_fine["DXG"].coarsen(i=box_i, boundary="trim").sum()
    DXG = DXG_sum.isel(j_g=slice(j0, None, box_j)).rename("DXG")
    
    DYG_sum = ds_fine["DYG"].coarsen(j=box_j, boundary="trim").sum()
    DYG = DYG_sum.isel(i_g=slice(i0, None, box_i)).rename("DYG")
    # DXC, DYC
    DYC_sum = ds_fine["DYC"].coarsen(i=box_i, boundary="trim").sum()
    DYC = DYC_sum.isel(j_g=slice(j0, None, box_j)).rename("DYC")

    DXC_sum = ds_fine["DXC"].coarsen(j=box_j, boundary="trim").sum()
    DXC = DXC_sum.isel(i_g=slice(i0, None, box_i)).rename("DXC")
    # depth metrics remain the same
    DRF = ds_fine["DRF"]
    DRC = ds_fine["DRC"]
    RF = ds_fine["RF"]
    RC = ds_fine["RC"]
    k_l = ds_fine["k_l"]
    #  for plotting/coord: area weighted mean 
    XC_weight = (ds_fine["XC"] * ds_fine["RAC"]).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    XC = (XC_weight / RAC).rename("XC")
    YC_weight = (ds_fine["YC"] * ds_fine["RAC"]).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    YC = (YC_weight / RAC).rename("YC")
    # mask coarsen
    # to satisfy the volume conservation: V_c = RAC_c * DRF * hFacC_c, DRF do not change
    hFacC_weight = (ds_fine["hFacC"] * ds_fine["RAC"] * ds_fine["DRF"]).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    hFacC = (hFacC_weight / (DRF * RAC)).rename("hFacC")
    # U face: flux conservation - face length weighted
    hFacW_weight = (ds_fine["hFacW"] * ds_fine["DYG"] * ds_fine["DRF"]).coarsen(j=box_j, boundary="trim").sum()
    hFacW_den = (ds_fine["DYG"] * ds_fine["DRF"]).coarsen(j=box_j, boundary="trim").sum()
    hFacW = (hFacW_weight / hFacW_den).isel(i_g=slice(i0, None, box_i)).where(hFacW_den.isel(i_g=slice(i0,None,box_i)) > 0, 0.0)
    # V face
    hFacS_weight = (ds_fine["hFacS"] * ds_fine["DXG"] * ds_fine["DRF"]).coarsen(i=box_i, boundary="trim").sum()
    hFacS_den = (ds_fine["DXG"] * ds_fine["DRF"]).coarsen(i=box_i, boundary="trim").sum()
    hFacS = (hFacS_weight / hFacS_den).isel(j_g=slice(j0, None, box_j)).where(hFacS_den.isel(j_g=slice(j0,None,box_j)) > 0, 0.0)
    # G point
    RAZ = ds_fine["RAZ"].coarsen(j_g=box_j, i_g=box_i, boundary="trim").sum().rename("RAZ")
    # normalize the dimension to align
    def as_ordinal(da: xr.DataArray) -> xr.DataArray:
        """Drop hitchhiking coords and replace each dim’s coordinate with 0..N-1."""
        da = da.reset_coords(drop=True)
        new_coords = {d: xr.DataArray(np.arange(da.sizes[d]), dims=d) for d in da.dims}
        return da.assign_coords(**new_coords)

    # apply AFTER coarsen/subsample, BEFORE combining
    RAC  = as_ordinal(RAC)
    RAW  = as_ordinal(RAW)
    RAS  = as_ordinal(RAS)
    RAZ  = as_ordinal(RAZ)
    DYG  = as_ordinal(DYG)
    DXG  = as_ordinal(DXG)
    hFacC = as_ordinal(hFacC)
    hFacW = as_ordinal(hFacW)
    hFacS = as_ordinal(hFacS)
    XC   = as_ordinal(XC)
    YC   = as_ordinal(YC)
    DXC = as_ordinal(DXC)
    DYC = as_ordinal(DYC)
    
    # assemble coarse package with correct coords
    dsC = xr.Dataset(
        dict(
            # centers
            RAC=RAC, RAW=RAW, RAS=RAS, DXG=DXG, DYG=DYG,
            DRF=DRF, DRC=DRC, RF=RF, RC=RC,
            DXC=DXC, DYC=DYC,
            XC=XC, YC=YC,
            hFacC=hFacC, hFacW=hFacW, hFacS=hFacS,
            RAZ=RAZ,
            k_l=k_l,
        )
    )
    # attach the coordinate
    nj, ni = dsC.sizes["j"], dsC.sizes["i"]
    dsC = dsC.assign_coords(
        j=np.arange(nj), i=np.arange(ni),
        j_g=np.arange(nj), i_g=np.arange(ni),
        k=np.arange(dsC.sizes["k"]),
        k_l=np.arange(dsC.sizes["k_l"]),
        k_p1 = np.arange(dsC.sizes["k_p1"])
    )

    return dsC