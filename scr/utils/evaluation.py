import torch
import torch.nn as nn
from typing import Tuple, Dict

def _eval_on_loader_both(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Evaluate both:
      (A) pixel-pooled metrics: flatten all pixels across dataset (like your current code)
      (B) per-image averaged metrics: compute per-sample R2 (and MSE), then average across samples

    Returns a dict with:
      pixel_mse, pixel_r2, image_mse_mean, image_r2_mean, image_r2_median
    """
    model.eval()

    # --- accumulators for pixel-pooled (global) ---
    sse_global = 0.0  # sum of squared errors over all pixels
    n_global = 0      # total number of pixels
    # for global R2 we need SST about the *global* mean of y over all pixels
    y_sum_global = 0.0
    y_sq_sum_global = 0.0

    # --- accumulators for per-image (samplewise) ---
    mse_sum_img = 0.0
    r2_sum_img = 0.0
    r2_list = []
    n_img = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            out = model(data)

            # Align shapes: (B,H,W) -> (B,1,H,W) if needed
            if out.ndim == 4 and target.ndim == 3:
                target = target.unsqueeze(1)
            if out.ndim == 3 and target.ndim == 4:
                out = out.unsqueeze(1)

            # Match dtype (helps avoid float64 surprises)
            target = target.to(dtype=out.dtype)

            B = out.shape[0]
            # Flatten per-sample as (B, P) where P = C*H*W (or H*W if C=1)
            yhat = out.reshape(B, -1)
            y    = target.reshape(B, -1)

            # ---------------- pixel-pooled MSE + R2 ----------------
            diff = (yhat - y)
            sse_global += float((diff * diff).sum().item())
            n_global += int(y.numel())

            y_sum_global += float(y.sum().item())
            y_sq_sum_global += float((y * y).sum().item())

            # ---------------- per-image MSE + R2 (average over images) ----------------
            # MSE per image: mean over pixels
            mse_per_img = (diff * diff).mean(dim=1)  # (B,)

            # R2 per image: 1 - SSE/SST (SST about per-image mean)
            y_mean = y.mean(dim=1, keepdim=True)     # (B,1)
            sst = ((y - y_mean) ** 2).sum(dim=1)     # (B,)
            sse = (diff ** 2).sum(dim=1)             # (B,)
            r2_per_img = 1.0 - (sse / (sst + eps))   # (B,)

            # Accumulate
            mse_sum_img += float(mse_per_img.sum().item())
            r2_sum_img  += float(r2_per_img.sum().item())
            r2_list.extend(r2_per_img.detach().cpu().tolist())
            n_img += B

    # Final pixel-pooled MSE
    pixel_mse = sse_global / max(1, n_global)

    # Final pixel-pooled R2:
    # SST_global = sum(y^2) - (sum(y)^2)/N
    sst_global = y_sq_sum_global - (y_sum_global * y_sum_global) / max(1, n_global)
    pixel_r2 = 1.0 - (sse_global / (sst_global + eps))

    # Per-image means
    image_mse_mean = mse_sum_img / max(1, n_img)
    image_r2_mean  = r2_sum_img / max(1, n_img)

    # Also nice: median per-image R2 (robust to a few terrible snapshots)
    r2_sorted = sorted(r2_list)
    if r2_sorted:
        mid = len(r2_sorted) // 2
        image_r2_median = r2_sorted[mid] if (len(r2_sorted) % 2 == 1) else 0.5 * (r2_sorted[mid-1] + r2_sorted[mid])
    else:
        image_r2_median = float("nan")

    return {
        "pixel_mse": float(pixel_mse),
        "pixel_r2": float(pixel_r2),
        "image_mse_mean": float(image_mse_mean),
        "image_r2_mean": float(image_r2_mean),
        "image_r2_median": float(image_r2_median),
        "n_images": int(n_img),
        "n_pixels": int(n_global),
    }
