from __future__ import annotations

import sys
from pathlib import Path

import gsw
import numpy as np
from submeso_data_driven.utils import make_grid
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submeso_data_driven.utils import mld_avg  # noqa: E402


def buo_coarse(ds_grid: xr.Dataset, ds_ocean: xr.Dataset, boundary_y: str = "fill"):
    ds = xr.merge([ds_grid, ds_ocean])
    grid = make_grid.make_grid_periodic_x(ds)
    z_rel = ds["RC"] - ds["Eta"]

    lat0, lon0 = -55.0, 153.0
    lat2d = xr.full_like(ds["T"].isel(k=0), lat0)
    lon2d = xr.full_like(ds["T"].isel(k=0), np.mod(lon0, 360.0))

    p_dbar = xr.apply_ufunc(
        gsw.p_from_z,
        z_rel,
        lat2d,
        dask="parallelized",
        output_dtypes=[ds["T"].dtype],
    )
    sa = xr.apply_ufunc(
        gsw.SA_from_SP,
        ds["S"],
        p_dbar,
        lon2d,
        lat2d,
        dask="parallelized",
        output_dtypes=[ds["S"].dtype],
    )
    ct = xr.apply_ufunc(gsw.CT_from_pt, sa, ds["T"], dask="parallelized", output_dtypes=[ds["T"].dtype])
    rho = xr.apply_ufunc(gsw.rho, sa, ct, p_dbar, dask="parallelized", output_dtypes=[ds["T"].dtype])

    b = (-9.81 * (rho - 1035.0) / 1035.0).where(ds["hFacC"] > 0)
    b.name = "buoyancy"
    b.attrs.update(units="m s^-2", long_name="buoyancy (TEOS-10)")

    wet_u = ds["hFacW"] > 0
    wet_v = ds["hFacS"] > 0
    dbdx_u = (grid.diff(b, "X") / ds["DXC"]).where(wet_u)
    dbdy_v = (grid.diff(b, "Y", boundary=boundary_y) / ds["DYC"]).where(wet_v)
    return b, dbdx_u, dbdy_v


def _k_name(da: xr.DataArray) -> str:
    for cand in ("k", "Z", "z", "lev", "k_c"):
        if cand in da.dims:
            return cand
    return da.dims[0]


def mld_integrals_for_MEI(
    dbdx_u: xr.DataArray,
    dbdy_v: xr.DataArray,
    kpphbl_c: xr.DataArray,
    ds_grid: xr.Dataset,
    grid,
    sigma_is_buoyancy: bool = True,
    g: float = 9.81,
    rho0: float = 1035.0,
):
    rf = ds_grid["RF"]
    k_size = rf.sizes["k_p1"] - 1
    k_index = xr.DataArray(np.arange(k_size), dims=("k",), name="k")
    dbdx_u = dbdx_u.rename({_k_name(dbdx_u): "k"}).assign_coords(k=k_index).reset_coords(drop=True)
    dbdy_v = dbdy_v.rename({_k_name(dbdy_v): "k"}).assign_coords(k=k_index).reset_coords(drop=True)

    h_c = kpphbl_c.clip(min=0.0)
    h_u = grid.interp(h_c, "X", to="left")
    h_v = grid.interp(h_c, "Y", to="left")

    ztop = (-rf.isel(k_p1=slice(0, -1))).rename(k_p1="k").assign_coords(k=k_index)
    zbot = (-rf.isel(k_p1=slice(1, None))).rename(k_p1="k").assign_coords(k=k_index)
    ztop_u = ztop.broadcast_like(dbdx_u)
    zbot_u = zbot.broadcast_like(dbdx_u)
    ztop_v = ztop.broadcast_like(dbdy_v)
    zbot_v = zbot.broadcast_like(dbdy_v)

    thk_u = (zbot_u - ztop_u) * ds_grid["hFacW"]
    thk_v = (zbot_v - ztop_v) * ds_grid["hFacS"]
    h3u = h_u.broadcast_like(dbdx_u)
    h3v = h_v.broadcast_like(dbdy_v)
    min_hu = xr.where(h3u <= zbot_u, h3u, zbot_u)
    min_hv = xr.where(h3v <= zbot_v, h3v, zbot_v)
    w_u = xr.where(h3u > ztop_u, min_hu - ztop_u, 0.0)
    w_v = xr.where(h3v > ztop_v, min_hv - ztop_v, 0.0)
    w_u = xr.where(w_u > 0, xr.where(w_u < thk_u, w_u, thk_u), 0.0)
    w_v = xr.where(w_v > 0, xr.where(w_v < thk_v, w_v, thk_v), 0.0)

    int_x = (dbdx_u.fillna(0.0) * w_u).sum("k")
    int_y = (dbdy_v.fillna(0.0) * w_v).sum("k")
    if not sigma_is_buoyancy:
        fac = -g / rho0
        int_x = fac * int_x
        int_y = fac * int_y

    int_x = int_x.where((ds_grid["hFacW"] > 0).any("k")).rename("dBuoyX_Hu")
    int_y = int_y.where((ds_grid["hFacS"] > 0).any("k")).rename("dBuoyY_Hv")
    return int_x, int_y


def compute_Lf_MEI(
    dBuoyX_Hu: xr.DataArray,
    dBuoyY_Hv: xr.DataArray,
    b: xr.DataArray,
    H_C: xr.DataArray,
    f_eff_C: xr.DataArray,
    ds_grid: xr.Dataset,
    grid,
    LfMin: float = 1000.0,
):
    rf = ds_grid["RF"]
    k_size = rf.sizes["k_p1"] - 1
    k_idx = xr.DataArray(np.arange(k_size), dims=("k",), name="k")

    def to_k(da):
        for cand in ("k", "Z", "z", "lev", "k_c"):
            if cand in da.dims:
                if cand != "k":
                    da = da.rename({cand: "k"})
                break
        if "k" in da.dims:
            da = da.assign_coords(k=k_idx).reset_coords(drop=True)
        return da

    b = to_k(b)
    hfac_c = to_k(ds_grid["hFacC"])
    ztop = (-rf.isel(k_p1=slice(0, -1))).rename(k_p1="k").assign_coords(k=k_idx)
    zbot = (-rf.isel(k_p1=slice(1, None))).rename(k_p1="k").assign_coords(k=k_idx)
    ztop_3d = ztop.broadcast_like(b)
    zbot_3d = zbot.broadcast_like(b)

    d_bx_c = grid.interp(dBuoyX_Hu, "X")
    d_by_c = grid.interp(dBuoyY_Hv, "Y")
    msq_h = np.sqrt(0.25 * (d_bx_c**2 + d_by_c**2))

    thk_c = (zbot_3d - ztop_3d) * hfac_c
    h3c = H_C.assign_coords().broadcast_like(b)
    min_hc = xr.where(h3c <= zbot_3d, h3c, zbot_3d)
    w_c = xr.where(h3c > ztop_3d, min_hc - ztop_3d, 0.0)
    w_c = xr.where(w_c > 0, xr.where(w_c < thk_c, w_c, thk_c), 0.0)

    db = b - b.shift(k=1)
    dz = ztop_3d - ztop_3d.shift(k=1)
    sigma_r = (db / dz).where(dz.notnull())
    sigma_r_pos = xr.where(sigma_r < 0.0, -sigma_r, 0.0)
    nh_term = (sigma_r_pos * w_c).sum("k")
    nh_mix_lay = np.sqrt(nh_term * H_C.clip(min=0.0)).fillna(0.0)

    f_c = xr.where(f_eff_C > 0, f_eff_C, np.nan)
    cand1 = msq_h / (f_c**2)
    cand2 = nh_mix_lay / f_c
    lf_center = np.maximum(np.maximum(cand1.fillna(0.0), cand2.fillna(0.0)), LfMin).rename("Lf")
    lf_u = grid.interp(lf_center, "X", to="left").rename("Lf_U")
    lf_v = grid.interp(lf_center, "Y", to="left").rename("Lf_V")
    return lf_center, lf_u, lf_v


def compute_fk_submeso_heat_flux_one_time(
    *,
    time_index: int,
    ds_grid: xr.Dataset,
    kpphbl_da: xr.DataArray,
    t_da: xr.DataArray,
    s_da: xr.DataArray,
    eta_da: xr.DataArray,
    c_e: float,
    submeso_lmax: float = 2751.0,
    submeso_inv_tau: float = 2.0e-6,
    lf_min: float = 1000.0,
    f0: float = -1.363e-4,
    beta: float = 1.313e-11,
    rho0: float = 1035.0,
    cp: float = 3990.0,
) -> np.ndarray:
    grid = make_grid.make_grid_periodic_x(ds_grid)
    kpphbl = kpphbl_da.sel(time=time_index)
    t = t_da.sel(time=time_index)
    s = s_da.sel(time=time_index)
    eta = eta_da.sel(time=time_index)

    b, dbdx_u, dbdy_v = buo_coarse(ds_grid, xr.Dataset(dict(T=t, S=s, Eta=eta)), "fill")
    d_buoy_x_hu, d_buoy_y_hv = mld_integrals_for_MEI(dbdx_u, dbdy_v, kpphbl, ds_grid, grid)

    yc = ds_grid["YC"]
    f = f0 + beta * (yc - float(yc.mean()))
    f_eff = np.hypot(f, submeso_inv_tau)
    f_u = grid.interp(f_eff, "X", to="left")
    f_v = grid.interp(f_eff, "Y", to="left")

    _, lf_u, lf_v = compute_Lf_MEI(
        dBuoyX_Hu=d_buoy_x_hu,
        dBuoyY_Hv=d_buoy_y_hv,
        b=b,
        H_C=kpphbl.clip(min=0.0),
        f_eff_C=f_eff,
        ds_grid=ds_grid,
        grid=grid,
        LfMin=lf_min,
    )

    rf = ds_grid["RF"].rename(k_p1="k")
    h_u = grid.interp(kpphbl.clip(min=0.0), "X", to="left")
    h_v = grid.interp(kpphbl.clip(min=0.0), "Y", to="left")
    z2h_u = xr.where(h_u > 0, 2.0 * rf.broadcast_like(dbdx_u) / h_u.broadcast_like(dbdx_u), 0.0)
    z2h_v = xr.where(h_v > 0, 2.0 * rf.broadcast_like(dbdy_v) / h_v.broadcast_like(dbdy_v), 0.0)

    def mu_from(z2h):
        q = (1.0 + z2h) ** 2
        mu = (1.0 - q) * (1.0 + (5.0 / 21.0) * q)
        return xr.where(mu > 0.0, mu, 0.0)

    mu_u = mu_from(z2h_u).transpose("k", "j", "i_g")
    mu_v = mu_from(z2h_v).transpose("k", "j_g", "i")
    dx_cap_u = ds_grid["DXC"].clip(max=submeso_lmax)
    dy_cap_v = ds_grid["DYC"].clip(max=submeso_lmax)

    d_buoy_y_on_u = grid.interp(grid.interp(d_buoy_y_hv, "Y"), "X", to="left")
    d_buoy_x_on_v = grid.interp(grid.interp(d_buoy_x_hu, "X"), "Y", to="left")
    psi_u0 = (c_e * d_buoy_y_on_u * h_u * dx_cap_u / (lf_u * f_u)).fillna(0.0)
    psi_v0 = (-c_e * d_buoy_x_on_v * h_v * dy_cap_v / (lf_v * f_v)).fillna(0.0)
    psi_x = psi_u0.broadcast_like(mu_u) * mu_u
    psi_y = psi_v0.broadcast_like(mu_v) * mu_v

    dtdx_u = (grid.diff(t, "X") / ds_grid["DXC"]).where(ds_grid["hFacW"] > 0)
    dtdy_v = (grid.diff(t, "Y", boundary="fill") / ds_grid["DYC"]).where(ds_grid["hFacS"] > 0)
    dtdx_c = grid.interp(dtdx_u.fillna(0.0), "X")
    dtdy_c = grid.interp(dtdy_v.fillna(0.0), "Y")
    dtdx_w = 0.5 * (dtdx_c + dtdx_c.shift(k=1))
    dtdy_w = 0.5 * (dtdy_c + dtdy_c.shift(k=1))

    psi_x_c = grid.interp(psi_x, "X")
    psi_y_c = grid.interp(psi_y, "Y")
    qz_t_cross = (rho0 * cp) * (psi_x_c * dtdy_w - psi_y_c * dtdx_w)
    qz_t_cross.name = "Qz_T_cross"
    qz_t_cross.attrs["units"] = "W m^-2"
    return np.asarray(mld_avg.mld_avg(qz_t_cross, kpphbl, ds_grid), dtype=np.float32)
