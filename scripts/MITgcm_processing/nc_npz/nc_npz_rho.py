import os
import numpy as np
import xarray as xr
import gsw

from src.utils import mld_avg
from src.utils import build_coarse_grid
from src.utils import read_mitgcm


# -------------------- helpers: load/merge npz --------------------
def load_existing_npz(npz_path: str):
    if not os.path.exists(npz_path):
        return np.array([], dtype=np.int64), None
    with np.load(npz_path) as npz:
        t_old = npz["time_index"].astype(np.int64)
        x_old = npz["data"].astype(np.float32)
    return t_old, x_old


def merge_time_series_npz(t_old, x_old, t_new, x_new):
    """Merge on time; NEW wins duplicates; return sorted."""
    if t_old.size == 0:
        order = np.argsort(t_new)
        return t_new[order], x_new[order]

    t_cat = np.concatenate([t_old, t_new])
    x_cat = np.concatenate([x_old, x_new], axis=0)

    idx_rev = np.arange(t_cat.size - 1, -1, -1)
    t_rev = t_cat[idx_rev]
    _, first_rev_idx = np.unique(t_rev, return_index=True)
    keep_rev = np.sort(first_rev_idx)
    keep = idx_rev[keep_rev]

    t_keep = t_cat[keep]
    x_keep = x_cat[keep]
    order = np.argsort(t_keep)
    return t_keep[order], x_keep[order]


# -------------------- density calculation --------------------
def rho_coarse(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
) -> xr.DataArray:
    """
    TEOS-10 density calculation matching src/utils/buo.py.

    Model output salinity is practical salinity and T is potential temperature.
    """
    ds = xr.merge([ds_grid, ds_ocean])

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
    ct = xr.apply_ufunc(
        gsw.CT_from_pt,
        sa,
        ds["T"],
        dask="parallelized",
        output_dtypes=[ds["T"].dtype],
    )
    rho = xr.apply_ufunc(
        gsw.rho,
        sa,
        ct,
        p_dbar,
        dask="parallelized",
        output_dtypes=[ds["T"].dtype],
    ).where(ds["hFacC"] > 0)

    rho.name = "rho"
    rho.attrs.update(units="kg m^-3", long_name="in-situ density (TEOS-10)")
    return rho


# -------------------- core: compute rho_mld for one source --------------------
def _compute_rho_mld_for_times(
    T_nc: str,
    S_nc: str,
    Eta_nc: str,
    KPPHBL_nc: str,
    ds_coarse_grid: xr.Dataset,
    time_vals: np.ndarray,
    crop: dict | None = None,
    batch: int = 8,
):
    """
    Compute TEOS-10 density from T,S,Eta and MLD-average it.

    Returns:
      t_new: (N,) int64
      x_new: (N, j, i) float32
    """
    out_list = []
    t_done = []

    with xr.open_dataset(T_nc, engine="h5netcdf", decode_times=False) as dsT, \
         xr.open_dataset(S_nc, engine="h5netcdf", decode_times=False) as dsS, \
         xr.open_dataset(Eta_nc, engine="h5netcdf", decode_times=False) as dsE, \
         xr.open_dataset(KPPHBL_nc, engine="h5netcdf", decode_times=False) as dsH:

        t_name = "T" if "T" in dsT.data_vars else list(dsT.data_vars)[0]
        s_name = "S" if "S" in dsS.data_vars else list(dsS.data_vars)[0]
        e_name = "Eta" if "Eta" in dsE.data_vars else list(dsE.data_vars)[0]
        h_name = "kpphbl" if "kpphbl" in dsH.data_vars else list(dsH.data_vars)[0]

        for i0 in range(0, time_vals.size, batch):
            tb = time_vals[i0:i0 + batch]

            Tb = dsT[t_name].sel(time=tb).load()
            Sb = dsS[s_name].sel(time=tb).load()
            Eb = dsE[e_name].sel(time=tb).load()
            Hb = dsH[h_name].sel(time=tb).load()

            for k in range(tb.size):
                T_k = Tb.isel(time=k)
                S_k = Sb.isel(time=k)
                E_k = Eb.isel(time=k)
                H_k = Hb.isel(time=k)

                ds_ocean_c = xr.Dataset(dict(T=T_k, S=S_k, Eta=E_k))
                rho_3d = rho_coarse(ds_coarse_grid, ds_ocean_c)
                rho_mld = mld_avg.mld_avg(rho_3d, H_k, ds_coarse_grid)

                if crop is not None:
                    rho_mld = rho_mld.isel(**crop)

                out_list.append(rho_mld.values.astype(np.float32, copy=False))
                t_done.append(int(tb[k]))

            print(f"[batch] computed {min(i0 + batch, time_vals.size)}/{time_vals.size}")

    x_new = np.stack(out_list, axis=0)
    t_new = np.array(t_done, dtype=np.int64)
    return t_new, x_new


# -------------------- main: merge sources into one npz output --------------------
def update_rho_mld_from_sources(
    sources: list[dict],
    out_npz: str,
    ds_coarse_grid: xr.Dataset,
    crop: dict | None = None,
    batch: int = 8,
):
    """
    sources: list of dict like:
      {
        "T_nc": "...",
        "S_nc": "...",
        "Eta_nc": "...",
        "H_nc": "kpphbl...",
      }

    Eta time is reference: only times present in Eta are used (and must exist in T,S,H too).
    Output NPZ keys: time_index, data
    """
    t_old, x_old = load_existing_npz(out_npz)
    saved_set = set(t_old.tolist())

    for si, src in enumerate(sources):
        T_nc = src["T_nc"]
        S_nc = src["S_nc"]
        Eta_nc = src["Eta_nc"]
        H_nc = src["H_nc"]

        print(f"\n[source {si+1}/{len(sources)}]")
        print(f"  T_nc={T_nc}")
        print(f"  S_nc={S_nc}")
        print(f"  Eta_nc={Eta_nc}")
        print(f"  H_nc={H_nc}")

        with xr.open_dataset(T_nc, engine="h5netcdf", decode_times=False) as dsT, \
             xr.open_dataset(S_nc, engine="h5netcdf", decode_times=False) as dsS, \
             xr.open_dataset(Eta_nc, engine="h5netcdf", decode_times=False) as dsE, \
             xr.open_dataset(H_nc, engine="h5netcdf", decode_times=False) as dsH:

            tT = dsT["time"].values.astype(np.int64)
            tS = dsS["time"].values.astype(np.int64)
            tE = dsE["time"].values.astype(np.int64)
            tH = dsH["time"].values.astype(np.int64)

            t_common = np.intersect1d(np.intersect1d(np.intersect1d(tE, tT), tS), tH)
            t_todo = np.array([tt for tt in t_common if int(tt) not in saved_set], dtype=np.int64)
            t_todo.sort()

            print(f"[info] T:{tT.size} S:{tS.size} Eta:{tE.size} H:{tH.size} common:{t_common.size}")
            print(f"[info] already_saved:{t_old.size} todo:{t_todo.size}")

        if t_todo.size == 0:
            print("[ok] nothing new in this source.")
            continue

        t_new, x_new = _compute_rho_mld_for_times(
            T_nc=T_nc,
            S_nc=S_nc,
            Eta_nc=Eta_nc,
            KPPHBL_nc=H_nc,
            ds_coarse_grid=ds_coarse_grid,
            time_vals=t_todo,
            crop=crop,
            batch=batch,
        )

        if x_old is None:
            t_old, x_old = t_new, x_new
        else:
            t_old, x_old = merge_time_series_npz(t_old, x_old, t_new, x_new)

        saved_set = set(t_old.tolist())

        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez_compressed(out_npz, time_index=t_old.astype(np.int64), data=x_old.astype(np.float32))
        print(f"[ok] checkpoint saved after source {si+1}: {out_npz}")

    print("\n[ok] final saved:", out_npz, "shape:", x_old.shape, "timesteps:", t_old.size)
    print("[check] NaNs:", bool(np.isnan(x_old).any()), "count:", int(np.isnan(x_old).sum()))


# -------------------- example main --------------------
if __name__ == "__main__":
    path_scratch = "/scratch/jczhang/"
    path_MITgcm = "MITgcm_BI/verification/tutorial_reentrant_channel/run/"
    grid_dir = path_scratch + path_MITgcm

    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=10, box_i=10)

    folder1 = path_scratch + "data_5km/save_coarse/5500080_13827312_nc"
    folder2 = path_scratch + "data_5km/save_coarse/13901616_nc"

    sources = [
        {
            "T_nc": os.path.join(folder1, "T_12hmean_coarse.nc"),
            "S_nc": os.path.join(folder1, "S_12hmean_coarse.nc"),
            "Eta_nc": os.path.join(folder1, "Eta_12hmean_coarse.nc"),
            "H_nc": os.path.join(folder1, "kpphbl_12hmean_coarse.nc"),
        },
        {
            "T_nc": os.path.join(folder2, "T_12hmean_coarse.nc"),
            "S_nc": os.path.join(folder2, "S_12hmean_coarse.nc"),
            "Eta_nc": os.path.join(folder2, "Eta_12hmean_coarse.nc"),
            "H_nc": os.path.join(folder2, "kpphbl_12hmean_coarse.nc"),
        },
    ]

    out_npz = path_scratch + "submeso_data_driven/data/mld/rho.npz"
    crop = {"j": slice(62, 118), "i": slice(2, 58)}  # -> 56x56

    update_rho_mld_from_sources(
        sources=sources,
        out_npz=out_npz,
        ds_coarse_grid=ds_coarse,
        crop=crop,
        batch=8,
    )

# python -m data.save_mld.nc_npz_rho
