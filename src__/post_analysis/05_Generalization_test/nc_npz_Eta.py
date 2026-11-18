import os
import numpy as np
import xarray as xr

from submeso_data_driven.utils import build_coarse_grid
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


# -------------------- core: read Eta for one source --------------------
def _read_Eta_for_times(
    Eta_nc: str,
    time_vals: np.ndarray,
    crop: dict | None = None,
    batch: int = 8,
):
    """
    Read Eta for times.
    Returns:
      t_new: (N,) int64
      x_new: (N, j, i) float32  (after crop)
    """
    out_list = []
    t_done = []

    with xr.open_dataset(Eta_nc, engine="h5netcdf", decode_times=False) as dsE:
        # robust var name
        eta_name = "Eta" if "Eta" in dsE.data_vars else list(dsE.data_vars)[0]

        for i0 in range(0, time_vals.size, batch):
            tb = time_vals[i0:i0 + batch]
            Eb = dsE[eta_name].sel(time=tb).load()

            for k in range(tb.size):
                E_k = Eb.isel(time=k)

                if crop is not None:
                    E_k = E_k.isel(**crop)

                out_list.append(E_k.values.astype(np.float32, copy=False))
                t_done.append(int(tb[k]))

            print(f"[batch] read {min(i0 + batch, time_vals.size)}/{time_vals.size}")

    x_new = np.stack(out_list, axis=0)
    t_new = np.array(t_done, dtype=np.int64)
    return t_new, x_new


# -------------------- main: merge sources into one npz --------------------
def update_Eta_from_sources(
    sources: list[dict],
    out_npz: str,
    crop: dict | None = None,
    batch: int = 8,
):
    """
    sources: list of dict, each dict like:
      {
        "Eta_nc": "path/to/Eta_12hmean_coarse.nc",
      }

    Sources processed in order; later sources win duplicates (new wins).
    Output NPZ keys: time_index, data
    """
    t_old, x_old = load_existing_npz(out_npz)
    saved_set = set(t_old.tolist())

    for si, src in enumerate(sources):
        Eta_nc = src["Eta_nc"]

        print(f"\n[source {si+1}/{len(sources)}]")
        print(f"  Eta_nc={Eta_nc}")

        with xr.open_dataset(Eta_nc, engine="h5netcdf", decode_times=False) as dsE:
            tE = dsE["time"].values.astype(np.int64)
            t_todo = np.array([t for t in tE if int(t) not in saved_set], dtype=np.int64)
            t_todo.sort()

            print(f"[info] Eta:{tE.size} already_saved:{t_old.size} todo:{t_todo.size}")

        if t_todo.size == 0:
            print("[ok] nothing new in this source.")
            continue

        t_new, x_new = _read_Eta_for_times(
            Eta_nc=Eta_nc,
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
    path_MITgcm  = "MITgcm_1km/verification/tutorial_reentrant_channel/output"
    grid_dir     = path_scratch + path_MITgcm

    # (not strictly needed for Eta, but keeping consistency with your other scripts)
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_dir)
    _ = build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=50, box_i=50)

    folder2 = path_scratch + "submeso_data_driven/src/post_analysis/05_Generalization_test"

    sources = [
        {"Eta_nc": os.path.join(folder2, "Eta_12hmean_coarse.nc")},
    ]

    out_npz = path_scratch + "submeso_data_driven/src/post_analysis/05_Generalization_test/Eta.npz"

    crop = {"j": slice(20, 40), "i": slice(0, 20)}  

    update_Eta_from_sources(
        sources=sources,
        out_npz=out_npz,
        crop=crop,
        batch=8,
    )