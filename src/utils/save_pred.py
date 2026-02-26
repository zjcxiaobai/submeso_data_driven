import os
from pathlib import Path
import numpy as np
import torch

@torch.no_grad()
def save_predictions_npy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_dir: str | Path,
    prefix: str = "test",
    norm: dict | None = None,
    denorm_y: bool = True,
):
    """
    Saves:
      {out_dir}/{prefix}_y_true.npy  shape (N,1,H,W)
      {out_dir}/{prefix}_y_pred.npy  shape (N,1,H,W)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    y_true_all = []
    y_pred_all = []

    y_mu = y_sd = None
    if denorm_y and norm is not None:
        y_mu = norm.get("y_mu", None)
        y_sd = norm.get("y_sd", None)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # ensure (B,1,H,W)
        if yb.ndim == 3:
            yb = yb.unsqueeze(1)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        # optional de-normalize to physical units
        if denorm_y and (y_mu is not None) and (y_sd is not None):
            pred = pred * y_sd + y_mu
            yb   = yb   * y_sd + y_mu

        y_true_all.append(yb.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0).astype(np.float32)
    y_pred_all = np.concatenate(y_pred_all, axis=0).astype(np.float32)

    np.save(out_dir / f"{prefix}_y_true.npy", y_true_all)
    np.save(out_dir / f"{prefix}_y_pred.npy", y_pred_all)

    print(f"Saved {prefix} y_true: {y_true_all.shape} -> {out_dir / (prefix + '_y_true.npy')}")
    print(f"Saved {prefix} y_pred: {y_pred_all.shape} -> {out_dir / (prefix + '_y_pred.npy')}")