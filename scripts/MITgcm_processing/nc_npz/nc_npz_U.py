import os
import numpy as np
import xarray as xr

from submeso_data_driven.utils import mld_avg
from submeso_data_driven.utils import make_grid
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


# -------------------- core: compute U_center_mld for one source --------------------
def _compute_Ucenter_mld_for_times(
    U_nc: str,
    KPPHBL_nc: str,
    ds_coarse_grid: xr.Dataset,
    time_vals: np.ndarray,
    interp_to: str = "center",               # <-- choose what your grid.interp expects ("C" is common for centers)
    interp_boundary: str = "periodic",  # <-- periodic-x
    crop: dict | None = None,
    batch: int = 8,
):
    """
    Read U only -> interp to center (make_grid) -> MLD avg for times.
    Returns:
      t_new: (N,) int64
      x_new: (N, j, i) float32  (after crop)
    """
    out_list = []
    t_done = []

    # build interpolation grid ONCE (important for speed)
    grid_c = make_grid.make_grid_periodic_x(ds_coarse_grid)

    with xr.open_dataset(U_nc, engine="h5netcdf", decode_times=False) as dsU, \
         xr.open_dataset(KPPHBL_nc, engine="h5netcdf", decode_times=False) as dsH:

        # robust H var name
        h_name = "kpphbl" if "kpphbl" in dsH.data_vars else list(dsH.data_vars)[0]

        for i0 in range(0, time_vals.size, batch):
            tb = time_vals[i0:i0 + batch]

            Ub = dsU["U"].sel(time=tb).load()
            Hb = dsH[h_name].sel(time=tb).load()

            for k in range(tb.size):
                U_k = Ub.isel(time=k)   # likely dims: (k, j, i_g) or similar
                H_k = Hb.isel(time=k)   # likely dims: (j, i) or (j, i) with coords

                # ---- stagger -> center interpolation using your grid tool ----
                # "X" means interpolate along x-direction staggering
                U_kc = grid_c.interp(U_k, "X", to=interp_to, boundary=interp_boundary)

                # ---- MLD average -> (j,i) ----
                U_mld = mld_avg.mld_avg(U_kc, H_k, ds_coarse_grid)

                # optional crop
                if crop is not None:
                    U_mld = U_mld.isel(**crop)

                out_list.append(U_mld.values.astype(np.float32, copy=False))
                t_done.append(int(tb[k]))

            print(f"[batch] computed {min(i0 + batch, time_vals.size)}/{time_vals.size}")

    x_new = np.stack(out_list, axis=0)
    t_new = np.array(t_done, dtype=np.int64)
    return t_new, x_new


# -------------------- main: merge TWO (or N) sources into one npz --------------------
def update_Ucenter_mld_from_sources(
    sources: list[dict],
    out_npz: str,
    ds_coarse_grid: xr.Dataset,
    interp_to: str = "center",
    interp_boundary: str = "periodic",
    crop: dict | None = None,
    batch: int = 8,
):
    """
    sources: list of dict, each dict like:
      {
        "U_nc": "path/to/U_12hmean_coarse.nc",
        "H_nc": "path/to/kpphbl_12hmean_coarse.nc",
      }

    Sources are processed in order; later sources win duplicates (new wins).
    Output NPZ keys: time_index, data
    """
    # load existing output (so you can resume)
    t_old, x_old = load_existing_npz(out_npz)
    saved_set = set(t_old.tolist())

    for si, src in enumerate(sources):
        U_nc = src["U_nc"]
        H_nc = src["H_nc"]

        print(f"\n[source {si+1}/{len(sources)}]")
        print(f"  U_nc={U_nc}")
        print(f"  H_nc={H_nc}")

        # figure out common times available in THIS source
        with xr.open_dataset(U_nc, engine="h5netcdf", decode_times=False) as dsU, \
             xr.open_dataset(H_nc, engine="h5netcdf", decode_times=False) as dsH:

            tU = dsU["time"].values.astype(np.int64)
            tH = dsH["time"].values.astype(np.int64)

            t_common = np.intersect1d(tU, tH)
            t_todo = np.array([t for t in t_common if int(t) not in saved_set], dtype=np.int64)
            t_todo.sort()

            print(f"[info] U:{tU.size} H:{tH.size} common:{t_common.size}")
            print(f"[info] already_saved:{t_old.size} todo:{t_todo.size}")

        if t_todo.size == 0:
            print("[ok] nothing new in this source.")
            continue

        # compute for this source
        t_new, x_new = _compute_Ucenter_mld_for_times(
            U_nc=U_nc,
            KPPHBL_nc=H_nc,
            ds_coarse_grid=ds_coarse_grid,
            time_vals=t_todo,
            interp_to=interp_to,
            interp_boundary=interp_boundary,
            crop=crop,
            batch=batch,
        )

        # merge into running output (new wins)
        if x_old is None:
            t_old, x_old = t_new, x_new
        else:
            t_old, x_old = merge_time_series_npz(t_old, x_old, t_new, x_new)

        saved_set = set(t_old.tolist())

        # save after each source (safer for long runs)
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

    # build coarse grid metrics once
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_dir)
    ds_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=10, box_i=10)

    # two data sources (folders)
    folder1 = path_scratch + "data_5km/save_coarse/5500080_13827312_nc"
    folder2 = path_scratch + "data_5km/save_coarse/13901616_nc"

    sources = [
        {
            "U_nc": os.path.join(folder1, "U_12hmean_coarse.nc"),
            "H_nc": os.path.join(folder1, "kpphbl_12hmean_coarse.nc"),
        },
        {
            "U_nc": os.path.join(folder2, "U_12hmean_coarse.nc"),
            "H_nc": os.path.join(folder2, "kpphbl_12hmean_coarse.nc"),
        },
    ]

    out_npz = path_scratch + "submeso_data_driven/data/mld/U.npz"

    # optional crop
    crop = {"j": slice(62, 118), "i": slice(2, 58)}  # -> 56x56

    update_Ucenter_mld_from_sources(
        sources=sources,
        out_npz=out_npz,
        ds_coarse_grid=ds_coarse,
        interp_to="center",
        interp_boundary="periodic",
        crop=crop,
        batch=8,
    )