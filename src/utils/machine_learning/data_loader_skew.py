# src/data/prepare_dataloaders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets_unet_and_seq import SpatialMapDataset, RollingWindowManyToOneDataset

# === your normalization module (train-only z-score) ===
# Expected API:
#   _fit_z_norm_X(X_train_nchw) -> (mu, sd) with shapes broadcastable like (1,C,1,1)
#   _fit_z_norm_y(y_train)      -> (mu, sd) scalars
from .zscore_per_channel import _fit_z_norm_X, _fit_z_norm_y  # <- adjust names if needed

ArrayLike = Union[np.ndarray, torch.Tensor]
SplitMode = Literal["time_ordered", "random"]
YTransform = Literal["none", "asinh"]


@dataclass(frozen=True)
class SplitInfo:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    skip_idx: np.ndarray


def _split_indices(
    n: int,
    *,
    frac_train: float,
    frac_skip: float,
    frac_val: float,
    frac_test: float,
    mode: SplitMode,
    seed: int,
) -> SplitInfo:
    total = frac_train + frac_skip + frac_val + frac_test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1. Got {total:.6f}")

    idx = np.arange(n, dtype=np.int64)
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)

    n_train = int(round(frac_train * n))
    n_skip = int(round(frac_skip * n))
    n_val = int(round(frac_val * n))
    n_test = n - (n_train + n_skip + n_val)

    i0 = 0
    i1 = i0 + n_train
    i2 = i1 + n_skip
    i3 = i2 + n_val
    i4 = i3 + n_test

    return SplitInfo(
        train_idx=idx[i0:i1],
        skip_idx=idx[i1:i2],
        val_idx=idx[i2:i3],
        test_idx=idx[i3:i4],
    )


def _default_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Target asinh helpers
# =========================
def _fit_asinh_scale_y(y_train: np.ndarray, q: float = 90.0) -> float:
    """
    Fit scale s for asinh(y/s) using TRAIN ONLY.
    Robust choice: percentile of |y| (e.g. 90th).
    """
    yf = np.asarray(y_train, dtype=np.float32).reshape(-1)
    yf = yf[np.isfinite(yf)]
    s = np.nanpercentile(np.abs(yf), q)
    return float(s + 1e-12)


def _apply_asinh_y(y: np.ndarray, s: float) -> np.ndarray:
    return np.arcsinh(np.asarray(y, dtype=np.float32) / s).astype(np.float32)


def prepare_unet_dataloaders(
    X: ArrayLike,              # (N,C,H,W)
    y: ArrayLike,              # (N,1,H,W) or (N,H,W)
    *,
    batch_size: int = 16,
    num_workers: int = 8,
    seed: int = 42,
    split_mode: SplitMode = "time_ordered",
    frac_train: float = 0.6,
    frac_skip: float = 0.2,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    normalize_x: bool = True,
    normalize_y: bool = False,
    shuffle_train: bool = True,
    device: Optional[torch.device] = None,
    # ---- NEW (target transform) ----
    y_transform: YTransform = "none",
    y_asinh_q: float = 90.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    UNet case: X(N,C,H,W), y(N,1,H,W) ready → split → fit train-only norm → DataLoaders.
    Optionally apply asinh transform to y (train-only fitted scale).
    """
    Xn = np.asarray(X, dtype=np.float32)
    yn = np.asarray(y, dtype=np.float32)

    if Xn.ndim != 4:
        raise ValueError(f"Expected X (N,C,H,W). Got {Xn.shape}")
    if yn.ndim not in (3, 4):
        raise ValueError(f"Expected y (N,H,W) or (N,1,H,W). Got {yn.shape}")

    splits = _split_indices(
        Xn.shape[0],
        frac_train=frac_train,
        frac_skip=frac_skip,
        frac_val=frac_val,
        frac_test=frac_test,
        mode=split_mode,
        seed=seed,
    )

    X_mu = X_sd = None
    y_mu = y_sd = None
    y_asinh_scale = None  # NEW

    if normalize_x:
        X_mu, X_sd = _fit_z_norm_X(Xn[splits.train_idx])  # train-only stats

    # ---- NEW: apply target transform using TRAIN ONLY ----
    if y_transform == "asinh":
        y_asinh_scale = _fit_asinh_scale_y(yn[splits.train_idx], q=y_asinh_q)
        yn = _apply_asinh_y(yn, y_asinh_scale)  # transform ALL splits with train-fitted scale

    # (optional) z-score y AFTER transform, still train-only
    if normalize_y:
        y_mu, y_sd = _fit_z_norm_y(yn[splits.train_idx])

    train_ds = SpatialMapDataset(
        Xn[splits.train_idx], yn[splits.train_idx],
        X_mu=X_mu, X_sd=X_sd, y_mu=y_mu, y_sd=y_sd
    )
    val_ds = SpatialMapDataset(
        Xn[splits.val_idx], yn[splits.val_idx],
        X_mu=X_mu, X_sd=X_sd, y_mu=y_mu, y_sd=y_sd
    )
    test_ds = SpatialMapDataset(
        Xn[splits.test_idx], yn[splits.test_idx],
        X_mu=X_mu, X_sd=X_sd, y_mu=y_mu, y_sd=y_sd
    )

    dev = _default_device(device)
    pin = (dev.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,        num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,        num_workers=num_workers, pin_memory=pin)

    meta = {
        "task": "unet_spatial",
        "split_mode": split_mode,
        "fractions": dict(frac_train=frac_train, frac_skip=frac_skip, frac_val=frac_val, frac_test=frac_test),
        "indices": {k: getattr(splits, k).copy() for k in ("train_idx", "skip_idx", "val_idx", "test_idx")},
        "norm": {
            "X_mu": X_mu, "X_sd": X_sd,
            "y_mu": y_mu, "y_sd": y_sd,
            "y_transform": y_transform,
            "y_asinh_q": y_asinh_q,
            "y_asinh_scale": y_asinh_scale,
        },
    }
    return train_loader, val_loader, test_loader, meta


def prepare_unet_lstm_many_to_one_dataloaders(
    X: ArrayLike,              # base snapshots (N,C,H,W)
    y: ArrayLike,              # base targets   (N,1,H,W) or (N,H,W)
    *,
    T: int = 7,
    stride: int = 1,
    target: Literal["last", "next"] = "last",
    batch_size: int = 8,
    num_workers: int = 8,
    seed: int = 42,
    split_mode: SplitMode = "time_ordered",
    frac_train: float = 0.6,
    frac_skip: float = 0.2,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    normalize_x: bool = True,
    normalize_y: bool = True,
    shuffle_train: bool = True,
    device: Optional[torch.device] = None,
    # ---- NEW (target transform) ----
    y_transform: YTransform = "none",
    y_asinh_q: float = 90.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """
    UNet+LSTM (many-to-one), using rolling windows ON-THE-FLY:
      x_i = X[s:s+T] -> (T,C,H,W)
      y_i = y[s+T-1] (target='last') or y[s+T] (target='next')

    No giant (Nw,T,C,H,W) arrays are created.
    Optionally apply asinh transform to y (train-only fitted scale).
    """
    Xn = np.asarray(X, dtype=np.float32)
    yn = np.asarray(y, dtype=np.float32)

    if Xn.ndim != 4:
        raise ValueError(f"Expected X (N,C,H,W). Got {Xn.shape}")
    if yn.ndim not in (3, 4):
        raise ValueError(f"Expected y (N,H,W) or (N,1,H,W). Got {yn.shape}")
    if T <= 0:
        raise ValueError("T must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    N = Xn.shape[0]
    max_start = N - T if target == "last" else N - T - 1
    if max_start < 0:
        raise ValueError(f"Not enough time steps N={N} for T={T} and target={target}")

    starts_all = np.arange(0, max_start + 1, stride, dtype=np.int64)
    n_windows = starts_all.size

    splits = _split_indices(
        n_windows,
        frac_train=frac_train,
        frac_skip=frac_skip,
        frac_val=frac_val,
        frac_test=frac_test,
        mode=split_mode,
        seed=seed,
    )

    starts_train = starts_all[splits.train_idx]
    starts_val   = starts_all[splits.val_idx]
    starts_test  = starts_all[splits.test_idx]

    # ====== Fit normalization using TRAIN ONLY, without materializing windows ======
    X_mu = X_sd = None
    y_mu = y_sd = None
    y_asinh_scale = None  # NEW

    if normalize_x:
        frame_mask = np.zeros((N,), dtype=bool)
        for s in starts_train:
            frame_mask[s : s + T] = True
        X_mu, X_sd = _fit_z_norm_X(Xn[frame_mask])  # train-only frames

    # Training target indices (train-only)
    if target == "last":
        t_targets_train = starts_train + (T - 1)
    else:
        t_targets_train = starts_train + T

    # ---- NEW: apply target transform using TRAIN ONLY ----
    if y_transform == "asinh":
        y_asinh_scale = _fit_asinh_scale_y(yn[t_targets_train], q=y_asinh_q)
        yn = _apply_asinh_y(yn, y_asinh_scale)  # transform ALL times with train-fitted scale

    # (optional) z-score y AFTER transform, still train-only
    if normalize_y:
        y_mu, y_sd = _fit_z_norm_y(yn[t_targets_train])

    train_ds = RollingWindowManyToOneDataset(
        Xn, yn, starts=starts_train, T=T, target=target,
        X_mu=X_mu, X_sd=X_sd, y_mu=y_mu, y_sd=y_sd
    )
    val_ds = RollingWindowManyToOneDataset(
        Xn, yn, starts=starts_val, T=T, target=target,
        X_mu=X_mu, X_sd=X_sd, y_mu=y_mu, y_sd=y_sd
    )
    test_ds = RollingWindowManyToOneDataset(
        Xn, yn, starts=starts_test, T=T, target=target,
        X_mu=X_mu, X_sd=X_sd, y_mu=y_mu, y_sd=y_sd
    )

    dev = _default_device(device)
    pin = (dev.type == "cuda")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,        num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,        num_workers=num_workers, pin_memory=pin)

    meta = {
        "task": "unet_lstm_many_to_one",
        "window": dict(T=T, stride=stride, target=target),
        "split_mode": split_mode,
        "fractions": dict(frac_train=frac_train, frac_skip=frac_skip, frac_val=frac_val, frac_test=frac_test),
        "window_indices": {
            "starts_train": starts_train.copy(),
            "starts_val": starts_val.copy(),
            "starts_test": starts_test.copy(),
        },
        "norm": {
            "X_mu": X_mu, "X_sd": X_sd,
            "y_mu": y_mu, "y_sd": y_sd,
            "y_transform": y_transform,
            "y_asinh_q": y_asinh_q,
            "y_asinh_scale": y_asinh_scale,
        },
    }
    return train_loader, val_loader, test_loader, meta