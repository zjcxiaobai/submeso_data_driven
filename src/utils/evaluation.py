import torch
import torch.nn as nn
from typing import Dict, Optional, Any


def _eval_on_loader_both(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 1e-12,
    norm: Optional[Dict[str, Any]] = None,
    # If an image has almost no variance in y, per-image R2 is unstable.
    # If skip_low_var_r2=True, we drop those samples from image_r2 stats.
    skip_low_var_r2: bool = True,
    sst_min: float = 1e-10,
) -> Dict[str, float]:
    """
    Evaluate both:
      (A) pixel-pooled metrics: flatten all pixels across dataset
      (B) per-image averaged metrics: compute per-sample MSE/R2, then average

    If `norm` is provided and contains y_mu/y_sd, also computes metrics in
    denormalized ("physical") space.

    Returns (always):
      pixel_mse, pixel_r2, image_mse_mean, image_r2_mean, image_r2_median, n_images, n_pixels

    Additionally (if norm provided):
      pixel_mse_phys, pixel_r2_phys, image_mse_mean_phys, image_r2_mean_phys, image_r2_median_phys
    """
    model.eval()

    # -------- helpers --------
    def _metrics_from_flat(yhat: torch.Tensor, y: torch.Tensor):
        """
        yhat, y: (B, P) float tensors on device
        Returns:
          sse_global, n_global, y_sum, y_sq_sum,
          mse_sum_img, r2_sum_img, r2_list, n_img_valid_r2
        """
        diff = yhat - y

        # pixel pooled
        sse_global = (diff * diff).sum()  # scalar tensor
        n_global = y.numel()
        y_sum = y.sum()
        y_sq_sum = (y * y).sum()

        # per-image
        mse_per_img = (diff * diff).mean(dim=1)  # (B,)
        y_mean = y.mean(dim=1, keepdim=True)
        sst = ((y - y_mean) ** 2).sum(dim=1)     # (B,)
        sse = (diff ** 2).sum(dim=1)             # (B,)

        if skip_low_var_r2:
            mask = sst > sst_min
            r2_per_img = torch.empty_like(sst)
            r2_per_img[:] = float("nan")
            r2_per_img[mask] = 1.0 - (sse[mask] / (sst[mask] + eps))
            n_img_valid_r2 = int(mask.sum().item())
        else:
            r2_per_img = 1.0 - (sse / (sst + eps))
            n_img_valid_r2 = int(sst.numel())

        return (
            sse_global, n_global, y_sum, y_sq_sum,
            mse_per_img.sum(), r2_per_img, int(mse_per_img.numel()), n_img_valid_r2
        )

    def _finalize(sse_global, n_global, y_sum, y_sq_sum, mse_sum_img, r2_list, n_img, n_img_valid_r2):
        pixel_mse = float((sse_global / max(1, n_global)).item())
        sst_global = y_sq_sum - (y_sum * y_sum) / max(1, n_global)
        pixel_r2 = float((1.0 - (sse_global / (sst_global + eps))).item())

        image_mse_mean = float((mse_sum_img / max(1, n_img)).item())

        # r2_list may contain NaNs if skip_low_var_r2=True
        r2_cpu = r2_list.detach().cpu()
        if skip_low_var_r2:
            r2_cpu = r2_cpu[~torch.isnan(r2_cpu)]
        r2_vals = r2_cpu.tolist()

        if len(r2_vals) == 0:
            image_r2_mean = float("nan")
            image_r2_median = float("nan")
        else:
            image_r2_mean = float(sum(r2_vals) / len(r2_vals))
            r2_vals_sorted = sorted(r2_vals)
            mid = len(r2_vals_sorted) // 2
            if len(r2_vals_sorted) % 2 == 1:
                image_r2_median = float(r2_vals_sorted[mid])
            else:
                image_r2_median = float(0.5 * (r2_vals_sorted[mid - 1] + r2_vals_sorted[mid]))

        return pixel_mse, pixel_r2, image_mse_mean, image_r2_mean, image_r2_median, n_img, n_global, n_img_valid_r2

    # -------- accumulators (normalized space) --------
    sse_g = torch.tensor(0.0, device=device)
    n_g = 0
    y_sum_g = torch.tensor(0.0, device=device)
    y_sq_sum_g = torch.tensor(0.0, device=device)

    mse_sum_img = torch.tensor(0.0, device=device)
    r2_all = []  # list of tensors (B,)
    n_img = 0
    n_img_valid_r2 = 0

    # -------- accumulators (physical space), if norm is provided --------
    do_phys = norm is not None and ("y_mu" in norm) and ("y_sd" in norm) and (norm["y_sd"] is not None)
    if do_phys:
        y_mu = float(norm["y_mu"])
        y_sd = float(norm["y_sd"])
        sse_g_p = torch.tensor(0.0, device=device)
        n_g_p = 0
        y_sum_g_p = torch.tensor(0.0, device=device)
        y_sq_sum_g_p = torch.tensor(0.0, device=device)
        mse_sum_img_p = torch.tensor(0.0, device=device)
        r2_all_p = []
        n_img_p = 0
        n_img_valid_r2_p = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            out = model(data)

            # Align shapes
            if out.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            if out.ndim == 3 and target.ndim == 4:
                out = out.unsqueeze(1)

            target = target.to(dtype=out.dtype)

            B = out.shape[0]
            yhat = out.reshape(B, -1)
            y = target.reshape(B, -1)

            # normalized-space metrics
            sse_b, n_b, ysum_b, ysq_b, mse_sum_b, r2_b, n_img_b, n_img_valid_b = _metrics_from_flat(yhat, y)
            sse_g += sse_b
            n_g += n_b
            y_sum_g += ysum_b
            y_sq_sum_g += ysq_b
            mse_sum_img += mse_sum_b
            r2_all.append(r2_b)
            n_img += n_img_b
            n_img_valid_r2 += n_img_valid_b

            # physical-space metrics (denormalize y and yhat)
            if do_phys:
                yhat_p = yhat * y_sd + y_mu
                y_p = y * y_sd + y_mu
                sse_bp, n_bp, ysum_bp, ysq_bp, mse_sum_bp, r2_bp, n_img_bp, n_img_valid_bp = _metrics_from_flat(yhat_p, y_p)
                sse_g_p += sse_bp
                n_g_p += n_bp
                y_sum_g_p += ysum_bp
                y_sq_sum_g_p += ysq_bp
                mse_sum_img_p += mse_sum_bp
                r2_all_p.append(r2_bp)
                n_img_p += n_img_bp
                n_img_valid_r2_p += n_img_valid_bp

    # stack r2 tensors
    r2_stack = torch.cat(r2_all, dim=0) if len(r2_all) else torch.tensor([], device="cpu")
    pixel_mse, pixel_r2, image_mse_mean, image_r2_mean, image_r2_median, n_images, n_pixels, n_img_valid = _finalize(
        sse_g, n_g, y_sum_g, y_sq_sum_g, mse_sum_img, r2_stack, n_img, n_img_valid_r2
    )

    out_dict: Dict[str, float] = {
        "pixel_mse": float(pixel_mse),
        "pixel_r2": float(pixel_r2),
        "image_mse_mean": float(image_mse_mean),
        "image_r2_mean": float(image_r2_mean),
        "image_r2_median": float(image_r2_median),
        "n_images": int(n_images),
        "n_pixels": int(n_pixels),
        "n_images_valid_r2": int(n_img_valid),
    }

    if do_phys:
        r2_stack_p = torch.cat(r2_all_p, dim=0) if len(r2_all_p) else torch.tensor([], device="cpu")
        pm_p, pr2_p, imse_p, ir2m_p, ir2med_p, nimg_p, npix_p, nimg_valid_p = _finalize(
            sse_g_p, n_g_p, y_sum_g_p, y_sq_sum_g_p, mse_sum_img_p, r2_stack_p, n_img_p, n_img_valid_r2_p
        )
        out_dict.update({
            "pixel_mse_phys": float(pm_p),
            "pixel_r2_phys": float(pr2_p),
            "image_mse_mean_phys": float(imse_p),
            "image_r2_mean_phys": float(ir2m_p),
            "image_r2_median_phys": float(ir2med_p),
            "n_images_valid_r2_phys": int(nimg_valid_p),
        })

    return out_dict
