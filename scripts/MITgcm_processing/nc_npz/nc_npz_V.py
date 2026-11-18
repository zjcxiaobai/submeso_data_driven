import os
import numpy as np
import xarray as xr

from src.utils import mld_avg
from src.utils import make_grid
from src.utils import build_coarse_grid
from submeso_data_driven.utils import read_mitgcm


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

    # keep last occurrence for duplicates (new wins)
    idx_rev = np.arange(t_cat.size - 1, -1, -1)
    t_rev = t_cat[idx_rev]
    _, first_rev_idx = np.unique(t_rev, return_index=True)
    keep_rev = np.sort(first_rev_idx)
    keep = idx_rev[keep_rev]

    t_keep = t_cat[keep]
    x_keep = x_cat[keep]
    order = np.argsort(t_keep)
    return t_keep[order], x_keep[order]


# -------------------- core: compute V_center_mld for one source --------------------
def _compute_Vcenter_mld_for_times(
    V_nc: str,
    KPPHBL_nc: str,
    ds_coarse_grid: xr.Dataset,
    time_vals: np.ndarray,
    interp_to: str = "center",
    interp_boundary: str = "fill",   # <-- V: fill
    crop: dict | None = None,
    batch: int = 8,
):
    out_list = []
    t_done = []

    # build interpolation grid ONCE (periodic in X only, per your helper)
    grid_c = make_grid.make_grid_periodic_x(ds_coarse_grid)

    with xr.open_dataset(V_nc, engine="h5netcdf", decode_times=False) as dsV, \
         xr.open_dataset(KPPHBL_nc, engine="h5netcdf", decode_times=False) as dsH:

        h_name = "kpphbl" if "kpphbl" in dsH.data_vars else list(dsH.data_vars)[0]

        for i0 in range(0, time_vals.size, batch):
            tb = time_vals[i0:i0 + batch]

            Vb = dsV["V"].sel(time=tb).load()
            Hb = dsH[h_name].sel(time=tb).load()

            for k in range(tb.size):
                V_k = Vb.isel(time=k)
                H_k = Hb.isel(time=k)

                # V is staggered in Y -> interp along "Y" to center
                V_kc = grid_c.interp(V_k, "Y", to=interp_to, boundary=interp_boundary)

                V_mld = mld_avg.mld_avg(V_kc, H_k, ds_coarse_grid)

                if crop is not None:
                    V_mld = V_mld.isel(**crop)

                out_list.append(V_mld.values.astype(np.float32, copy=False))
                t_done.append(int(tb[k]))

            print(f"[batch] computed {min(i0 + batch, time_vals.size)}/{time_vals.size}")

    x_new = np.stack(out_list, axis=0)
    t_new = np.array(t_done, dtype=np.int64)
    return t_new, x_new


# -------------------- main: merge sources into one npz --------------------
def update_Vcenter_mld_from_sources(
    sources: list[dict],
    out_npz: str,
    ds_coarse_grid: xr.Dataset,
    interp_to: str = "center",
    interp_boundary: str = "fill",   # <-- V: fill
    crop: dict | None = None,
    batch: int = 8,
):
    t_old, x_old = load_existing_npz(out_npz)
    saved_set = set(t_old.tolist())

    for si, src in enumerate(sources):
        V_nc = src["V_nc"]
        H_nc = src["H_nc"]

        print(f"\n[source {si+1}/{len(sources)}]")
        print(f"  V_nc={V_nc}")
        print(f"  H_nc={H_nc}")

        with xr.open_dataset(V_nc, engine="h5netcdf", decode_times=False) as dsV, \
             xr.open_dataset(H_nc, engine="h5netcdf", decode_times=False) as dsH:

            tV = dsV["time"].values.astype(np.int64)
            tH = dsH["time"].values.astype(np.int64)

            t_common = np.intersect1d(tV, tH)
            t_todo = np.array([t for t in t_common if int(t) not in saved_set], dtype=np.int64)
            t_todo.sort()

            print(f"[info] V:{tV.size} H:{tH.size} common:{t_common.size}")
            print(f"[info] already_saved:{t_old.size} todo:{t_todo.size}")

        if t_todo.size == 0:
            print("[ok] nothing new in this source.")
            continue

        t_new, x_new = _compute_Vcenter_mld_for_times(
            V_nc=V_nc,
            KPPHBL_nc=H_nc,
            ds_coarse_grid=ds_coarse_grid,
            time_vals=t_todo,
            interp_to=interp_to,
            interp_boundary=interp_boundary,
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
    path_MITgcm  = "MITgcm_BI/verification/tutorial_reentrant_channel/run/"
    grid_dir     = path_scratch + path_MITgcm

    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=10, box_i=10)

    folder1 = path_scratch + "data_5km/save_coarse/5500080_13827312_nc"
    folder2 = path_scratch + "data_5km/save_coarse/13901616_nc"

    sources = [
        {
            "V_nc": os.path.join(folder1, "V_12hmean_coarse.nc"),
            "H_nc": os.path.join(folder1, "kpphbl_12hmean_coarse.nc"),
        },
        {
            "V_nc": os.path.join(folder2, "V_12hmean_coarse.nc"),
            "H_nc": os.path.join(folder2, "kpphbl_12hmean_coarse.nc"),
        },
    ]

    out_npz = path_scratch + "submeso_data_driven/data/mld/V.npz"

    crop = {"j": slice(62, 118), "i": slice(2, 58)}  # -> 56x56

    update_Vcenter_mld_from_sources(
        sources=sources,
        out_npz=out_npz,
        ds_coarse_grid=ds_coarse,
        interp_to="center",
        interp_boundary="fill",   # <-- V: fill
        crop=crop,
        batch=8,
    )