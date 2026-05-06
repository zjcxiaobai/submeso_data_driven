from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np
import torch
import torch.distributed as dist


def _r2_from_sums(ss_res: float, ss_tot: float, eps: float = 1e-12) -> float:
    if ss_tot <= eps:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _all_reduce_sum_scalar(value: float, device: torch.device) -> float:
    t = torch.tensor(value, dtype=torch.float64, device=device)
    if _is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_reduce_sum_int(value: int, device: torch.device) -> int:
    t = torch.tensor(value, dtype=torch.long, device=device)
    if _is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _all_reduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    if _is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


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
    Distributed-safe evaluation.

    Returns global metrics aggregated across all ranks:
      - pixel_r2
      - image_r2_list
      - image_r2_mean / image_r2_median
      - climate_r2
      - n_images
      - n_pixels

    Expected:
      * model outputs (B,1,H,W)
      * y is (B,1,H,W) or (B,H,W)
      * loader may be sharded by DistributedSampler
    """
    model.eval()

    # optional de-normalization scalars
    y_mu = None
    y_sd = None
    if denorm_y and norm is not None:
        y_mu = norm.get("y_mu", None)
        y_sd = norm.get("y_sd", None)

        if y_mu is not None and isinstance(y_mu, np.ndarray):
            y_mu = float(np.asarray(y_mu).item())
        if y_sd is not None and isinstance(y_sd, np.ndarray):
            y_sd = float(np.asarray(y_sd).item())

        if y_mu is not None:
            y_mu = float(y_mu)
        if y_sd is not None:
            y_sd = float(y_sd)

    # local accumulators
    n_pix_total_local = 0
    sum_y_local = 0.0
    sum_y2_local = 0.0
    ss_res_total_local = 0.0

    image_r2_list_local: List[float] = []

    sum_true_map_local = None
    sum_pred_map_local = None
    n_images_local = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if yb.ndim == 3:
            yb = yb.unsqueeze(1)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        if denorm_y and (y_mu is not None) and (y_sd is not None):
            y_true = yb * y_sd + y_mu
            y_pred = pred * y_sd + y_mu
        else:
            y_true = yb
            y_pred = pred

        B = y_true.shape[0]
        n_images_local += B

        # per-image R2
        yt = y_true.reshape(B, -1).double()
        yp = y_pred.reshape(B, -1).double()

        yt_mean = yt.mean(dim=1, keepdim=True)
        ss_tot_i = ((yt - yt_mean) ** 2).sum(dim=1)
        ss_res_i = ((yt - yp) ** 2).sum(dim=1)

        r2_i = 1.0 - ss_res_i / (ss_tot_i + eps)
        r2_i = torch.where(
            ss_tot_i <= eps,
            torch.full_like(r2_i, float("nan")),
            r2_i,
        )
        image_r2_list_local.extend(r2_i.detach().cpu().tolist())

        # global pixel accumulators
        yt_all = yt.reshape(-1)
        yp_all = yp.reshape(-1)

        n_pix = yt_all.numel()
        n_pix_total_local += n_pix
        sum_y_local += float(yt_all.sum().item())
        sum_y2_local += float((yt_all ** 2).sum().item())
        ss_res_total_local += float(((yt_all - yp_all) ** 2).sum().item())

        # climate mean map accumulators
        yt_map_sum = y_true.sum(dim=0, keepdim=False).double()   # (1,H,W)
        yp_map_sum = y_pred.sum(dim=0, keepdim=False).double()

        if sum_true_map_local is None:
            sum_true_map_local = yt_map_sum
            sum_pred_map_local = yp_map_sum
        else:
            sum_true_map_local += yt_map_sum
            sum_pred_map_local += yp_map_sum

    # ---- reduce scalar accumulators across ranks ----
    n_pix_total = _all_reduce_sum_int(n_pix_total_local, device)
    n_images = _all_reduce_sum_int(n_images_local, device)

    sum_y = _all_reduce_sum_scalar(sum_y_local, device)
    sum_y2 = _all_reduce_sum_scalar(sum_y2_local, device)
    ss_res_total = _all_reduce_sum_scalar(ss_res_total_local, device)

    # ---- reduce climate map accumulators across ranks ----
    if sum_true_map_local is None:
        # handle empty loader safely
        sum_true_map_local = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)
        sum_pred_map_local = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)

    sum_true_map = _all_reduce_sum_tensor(sum_true_map_local.clone())
    sum_pred_map = _all_reduce_sum_tensor(sum_pred_map_local.clone())

    # ---- gather image_r2_list from all ranks ----
    if _is_distributed():
        gathered_lists = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_lists, image_r2_list_local)
        image_r2_list = []
        for lst in gathered_lists:
            image_r2_list.extend(lst)
    else:
        image_r2_list = image_r2_list_local

    # ---- global pixel R2 ----
    mean_y = sum_y / max(1, n_pix_total)
    ss_tot_total = sum_y2 - (n_pix_total * (mean_y ** 2))
    pixel_r2 = _r2_from_sums(ss_res_total, ss_tot_total, eps=eps)

    # ---- climate R2 ----
    if n_images > 0:
        true_mean_map = (sum_true_map / n_images).reshape(-1)
        pred_mean_map = (sum_pred_map / n_images).reshape(-1)

        tm = true_mean_map.mean()
        ss_tot_clim = float(((true_mean_map - tm) ** 2).sum().item())
        ss_res_clim = float(((true_mean_map - pred_mean_map) ** 2).sum().item())
        climate_r2 = _r2_from_sums(ss_res_clim, ss_tot_clim, eps=eps)
    else:
        climate_r2 = float("nan")

    # ---- per-image stats ----
    r2_arr = np.asarray(image_r2_list, dtype=np.float64)
    image_r2_mean = float(np.nanmean(r2_arr)) if r2_arr.size else float("nan")
    image_r2_median = float(np.nanmedian(r2_arr)) if r2_arr.size else float("nan")

    return {
        "pixel_r2": float(pixel_r2),
        "image_r2_list": image_r2_list,
        "image_r2_mean": image_r2_mean,
        "image_r2_median": image_r2_median,
        "climate_r2": float(climate_r2),
        "n_images": int(n_images),
        "n_pixels": int(n_pix_total),
    }