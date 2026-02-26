from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np
import torch


def _r2_from_sums(ss_res: float, ss_tot: float, eps: float = 1e-12) -> float:
    # If variance is ~0, R2 is undefined; return nan
    if ss_tot <= eps:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


@torch.no_grad()
def eval_r2_suite(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    norm: Optional[Dict[str, object]] = None,
    denorm_y: bool = False,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Returns:
      - pixel_r2: global R2 over all pixels from all images
      - image_r2_list: list of R2 per image (length = n_images)
      - image_r2_mean / median (nan-aware)
      - climate_r2: R2 between time-mean truth map and time-mean predicted map
      - bookkeeping: n_images, n_pixels
    Notes:
      * Assumes model outputs (B,1,H,W) and y is (B,1,H,W) or (B,H,W).
      * denorm_y is optional; R2 is invariant to affine scaling if applied to both,
        but you may want denorm for consistency with other metrics.
    """
    model.eval()

    # de-normalization scalars (optional)
    y_mu = None
    y_sd = None
    if denorm_y and norm is not None:
        y_mu = norm.get("y_mu", None)
        y_sd = norm.get("y_sd", None)
        if y_mu is not None and isinstance(y_mu, (np.ndarray,)):
            y_mu = float(np.asarray(y_mu).item())
        if y_sd is not None and isinstance(y_sd, (np.ndarray,)):
            y_sd = float(np.asarray(y_sd).item())
        if y_mu is not None:
            y_mu = float(y_mu)
        if y_sd is not None:
            y_sd = float(y_sd)

    # --- Global pixel R2 streaming accumulators ---
    n_pix_total = 0
    sum_y = 0.0
    sum_y2 = 0.0
    ss_res_total = 0.0

    # --- Per-image R2 list ---
    image_r2_list: List[float] = []

    # --- Climate mean maps (time mean over samples) ---
    sum_true_map = None  # (1,H,W)
    sum_pred_map = None  # (1,H,W)
    n_images = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)

        # yb can be (B,H,W) or (B,1,H,W)
        yb = yb.to(device, non_blocking=True)
        if yb.ndim == 3:
            yb = yb.unsqueeze(1)  # -> (B,1,H,W)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        # optional denorm
        if denorm_y and (y_mu is not None) and (y_sd is not None):
            y_true = yb * y_sd + y_mu
            y_pred = pred * y_sd + y_mu
        else:
            y_true = yb
            y_pred = pred

        B = y_true.shape[0]
        n_images += B

        # ---- per-image R2 (vectorized) ----
        yt = y_true.reshape(B, -1).double()
        yp = y_pred.reshape(B, -1).double()

        yt_mean = yt.mean(dim=1, keepdim=True)
        ss_tot_i = ((yt - yt_mean) ** 2).sum(dim=1)
        ss_res_i = ((yt - yp) ** 2).sum(dim=1)

        r2_i = 1.0 - ss_res_i / (ss_tot_i + eps)  # if ss_tot_i==0 -> ~1 - huge/eps -> negative; handle below
        # Replace zero-variance cases with nan (more honest)
        r2_i = torch.where(ss_tot_i <= eps, torch.full_like(r2_i, float("nan")), r2_i)
        image_r2_list.extend(r2_i.detach().cpu().tolist())

        # ---- global pixel accumulators ----
        # accumulate sums on CPU-friendly scalars
        yt_all = yt.reshape(-1)
        yp_all = yp.reshape(-1)

        n_pix = yt_all.numel()
        n_pix_total += n_pix
        sum_y += float(yt_all.sum().item())
        sum_y2 += float((yt_all ** 2).sum().item())
        ss_res_total += float(((yt_all - yp_all) ** 2).sum().item())

        # ---- climate mean maps ----
        # keep as (1,H,W) sums
        yt_map_sum = y_true.sum(dim=0, keepdim=False).double()  # (1,H,W)
        yp_map_sum = y_pred.sum(dim=0, keepdim=False).double()
        if sum_true_map is None:
            sum_true_map = yt_map_sum
            sum_pred_map = yp_map_sum
        else:
            sum_true_map += yt_map_sum
            sum_pred_map += yp_map_sum

    # ----- global pixel R2 -----
    mean_y = sum_y / max(1, n_pix_total)
    ss_tot_total = sum_y2 - (n_pix_total * (mean_y ** 2))
    pixel_r2 = _r2_from_sums(ss_res_total, ss_tot_total, eps=eps)

    # ----- climate R2 (time-mean maps) -----
    if n_images > 0 and sum_true_map is not None:
        true_mean_map = (sum_true_map / n_images).reshape(-1)
        pred_mean_map = (sum_pred_map / n_images).reshape(-1)

        tm = true_mean_map.mean()
        ss_tot_clim = float(((true_mean_map - tm) ** 2).sum().item())
        ss_res_clim = float(((true_mean_map - pred_mean_map) ** 2).sum().item())
        climate_r2 = _r2_from_sums(ss_res_clim, ss_tot_clim, eps=eps)
    else:
        climate_r2 = float("nan")

    # ----- per-image stats (nan-aware) -----
    r2_arr = np.asarray(image_r2_list, dtype=np.float64)
    image_r2_mean = float(np.nanmean(r2_arr)) if r2_arr.size else float("nan")
    image_r2_median = float(np.nanmedian(r2_arr)) if r2_arr.size else float("nan")

    return {
        "pixel_r2": float(pixel_r2),
        "image_r2_list": image_r2_list,  # requested: full list
        "image_r2_mean": image_r2_mean,
        "image_r2_median": image_r2_median,
        "climate_r2": float(climate_r2),
        "n_images": int(n_images),
        "n_pixels": int(n_pix_total),
    }