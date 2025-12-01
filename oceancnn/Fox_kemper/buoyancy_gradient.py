import xarray as xr
import numpy as np
import make_grid
import gsw

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
    z_rel = ds["RC"] - ds["Eta"]  # RC is negative, RC - Eta gives the depth of water column
    
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
    
    return b, b_grad, dbdx_u, dbdy_v