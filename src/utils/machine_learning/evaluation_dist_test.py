from __future__ import annotations

from typing import Dict, Optional, List, Tuple
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


def _get_y_denorm_scalars(norm: Optional[Dict[str, object]]) -> Tuple[Optional[float], Optional[float]]:
    if norm is None:
        return None, None

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

    return y_mu, y_sd


def _build_snapshot_mask(
    y_true_norm: torch.Tensor,
    norm: Optional[Dict[str, object]] = None,
    threshold_frac: float = 0.15,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Build a per-snapshot binary mask from the target.

    Mask rule:
        mask = |y_true_phys| >= threshold_frac * max(|y_true_phys|)   (per snapshot)

    Returns mask with same shape as y_true_norm: (B,1,H,W)
    """
    if y_true_norm.ndim == 3:
        y_true_norm = y_true_norm.unsqueeze(1)

    y_mu, y_sd = _get_y_denorm_scalars(norm)

    # build mask in physical units if possible
    if (y_mu is not None) and (y_sd is not None):
        y_phys = y_true_norm * y_sd + y_mu
    else:
        y_phys = y_true_norm

    abs_y = y_phys.abs()
    B = abs_y.shape[0]

    flat = abs_y.reshape(B, -1)
    max_abs = flat.max(dim=1).values  # (B,)
    tau = threshold_frac * max_abs

    valid = (max_abs > eps).view(B, 1, 1, 1)
    mask = abs_y >= tau.view(B, 1, 1, 1)
    mask = mask & valid

    return mask.to(y_true_norm.dtype)


@torch.no_grad()
def eval_pixel_mse(
    model,
    loader,
    device,
    norm: Optional[Dict[str, object]] = None,
    threshold_frac: float = 0.15,
):
    model.eval()

    sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    n = torch.tensor(0.0, device=device, dtype=torch.float64)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if yb.ndim == 3:
            yb = yb.unsqueeze(1)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        mask = _build_snapshot_mask(yb, norm=norm, threshold_frac=threshold_frac).double()
        err2 = ((pred - yb).double() ** 2) * mask

        sse += err2.sum()
        n += mask.sum()

    if _is_distributed():
        dist.all_reduce(sse, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    return float((sse / torch.clamp(n, min=1.0)).item())


@torch.no_grad()
def eval_pixel_mae(
    model,
    loader,
    device,
    norm: Optional[Dict[str, object]] = None,
    threshold_frac: float = 0.15,
):
    model.eval()

    sae = torch.tensor(0.0, device=device, dtype=torch.float64)
    n = torch.tensor(0.0, device=device, dtype=torch.float64)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if yb.ndim == 3:
            yb = yb.unsqueeze(1)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        mask = _build_snapshot_mask(yb, norm=norm, threshold_frac=threshold_frac).double()
        abs_err = (pred - yb).abs().double() * mask

        sae += abs_err.sum()
        n += mask.sum()

    if _is_distributed():
        dist.all_reduce(sae, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    return float((sae / torch.clamp(n, min=1.0)).item())


@torch.no_grad()
def eval_r2_suite(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    norm: Optional[Dict[str, object]] = None,
    denorm_y: bool = False,
    threshold_frac: float = 0.15,
    eps: float = 1e-12,
) -> Dict[str, object]:
    """
    Masked evaluation.

    The mask is built from the TRUE target of each snapshot:
        |y_true_phys| >= threshold_frac * max(|y_true_phys|)

    Metrics are computed only on masked pixels.
    """
    model.eval()

    y_mu = None
    y_sd = None
    if denorm_y and norm is not None:
        y_mu, y_sd = _get_y_denorm_scalars(norm)

    n_pix_total_local = 0
    sum_y_local = 0.0
    sum_y2_local = 0.0
    ss_res_total_local = 0.0

    image_r2_list_local: List[float] = []

    sum_true_map_local = None
    sum_pred_map_local = None
    sum_mask_map_local = None
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

        # mask always built from physical target if norm is available
        mask = _build_snapshot_mask(yb, norm=norm, threshold_frac=threshold_frac).double()

        B = y_true.shape[0]
        n_images_local += B

        yt = y_true.reshape(B, -1).double()
        yp = y_pred.reshape(B, -1).double()
        mk = mask.reshape(B, -1).double()

        # per-image masked R2
        valid_counts = mk.sum(dim=1)
        yt_mean = (yt * mk).sum(dim=1) / torch.clamp(valid_counts, min=1.0)

        ss_tot_i = (((yt - yt_mean.unsqueeze(1)) ** 2) * mk).sum(dim=1)
        ss_res_i = (((yt - yp) ** 2) * mk).sum(dim=1)

        r2_i = 1.0 - ss_res_i / (ss_tot_i + eps)
        r2_i = torch.where(
            valid_counts <= 0,
            torch.full_like(r2_i, float("nan")),
            r2_i,
        )
        r2_i = torch.where(
            ss_tot_i <= eps,
            torch.full_like(r2_i, float("nan")),
            r2_i,
        )
        image_r2_list_local.extend(r2_i.detach().cpu().tolist())

        # global masked pixel accumulators
        yt_all = yt.reshape(-1)
        yp_all = yp.reshape(-1)
        mk_all = mk.reshape(-1)

        n_pix = int(mk_all.sum().item())
        n_pix_total_local += n_pix
        sum_y_local += float((yt_all * mk_all).sum().item())
        sum_y2_local += float(((yt_all ** 2) * mk_all).sum().item())
        ss_res_total_local += float((((yt_all - yp_all) ** 2) * mk_all).sum().item())

        # masked climate mean map
        yt_map_sum = (y_true.double() * mask).sum(dim=0, keepdim=False)
        yp_map_sum = (y_pred.double() * mask).sum(dim=0, keepdim=False)
        mk_map_sum = mask.sum(dim=0, keepdim=False).double()

        if sum_true_map_local is None:
            sum_true_map_local = yt_map_sum
            sum_pred_map_local = yp_map_sum
            sum_mask_map_local = mk_map_sum
        else:
            sum_true_map_local += yt_map_sum
            sum_pred_map_local += yp_map_sum
            sum_mask_map_local += mk_map_sum

    n_pix_total = _all_reduce_sum_int(n_pix_total_local, device)
    n_images = _all_reduce_sum_int(n_images_local, device)

    sum_y = _all_reduce_sum_scalar(sum_y_local, device)
    sum_y2 = _all_reduce_sum_scalar(sum_y2_local, device)
    ss_res_total = _all_reduce_sum_scalar(ss_res_total_local, device)

    if sum_true_map_local is None:
        sum_true_map_local = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)
        sum_pred_map_local = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)
        sum_mask_map_local = torch.zeros(1, 1, 1, dtype=torch.float64, device=device)

    sum_true_map = _all_reduce_sum_tensor(sum_true_map_local.clone())
    sum_pred_map = _all_reduce_sum_tensor(sum_pred_map_local.clone())
    sum_mask_map = _all_reduce_sum_tensor(sum_mask_map_local.clone())

    if _is_distributed():
        gathered_lists = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_lists, image_r2_list_local)
        image_r2_list = []
        for lst in gathered_lists:
            image_r2_list.extend(lst)
    else:
        image_r2_list = image_r2_list_local

    mean_y = sum_y / max(1, n_pix_total)
    ss_tot_total = sum_y2 - (n_pix_total * (mean_y ** 2))
    pixel_r2 = _r2_from_sums(ss_res_total, ss_tot_total, eps=eps)

    # masked climate R2
    valid_map = sum_mask_map > 0
    if valid_map.any():
        true_mean_map = torch.zeros_like(sum_true_map)
        pred_mean_map = torch.zeros_like(sum_pred_map)

        true_mean_map[valid_map] = sum_true_map[valid_map] / sum_mask_map[valid_map]
        pred_mean_map[valid_map] = sum_pred_map[valid_map] / sum_mask_map[valid_map]

        tm = true_mean_map[valid_map].mean()
        ss_tot_clim = float(((true_mean_map[valid_map] - tm) ** 2).sum().item())
        ss_res_clim = float(((true_mean_map[valid_map] - pred_mean_map[valid_map]) ** 2).sum().item())
        climate_r2 = _r2_from_sums(ss_res_clim, ss_tot_clim, eps=eps)
    else:
        climate_r2 = float("nan")

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
        "threshold_frac": float(threshold_frac),
    }