# src/data/datasets_unet_and_seq.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

# the function can accept both input types
ArrayLike = Union[np.ndarray, torch.Tensor]


class SpatialMapDataset(Dataset):
    """
    For UNet (spatial):
      X: (N, C, H, W)
      y: (N, 1, H, W) or (N, H, W)

    Optionally applies z-score normalization using provided stats.
    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        X_mu: Optional[np.ndarray] = None,  # ideally (1,C,1,1)
        X_sd: Optional[np.ndarray] = None,  # ideally (1,C,1,1)
        y_mu: Optional[float] = None,
        y_sd: Optional[float] = None,
    ):
        # check dims
        if X.ndim != 4:
            raise ValueError(f"SpatialMapDataset expects X (N,C,H,W). Got {X.shape}")
        if y.ndim not in (3, 4):
            raise ValueError(f"SpatialMapDataset expects y (N,H,W) or (N,1,H,W). Got {y.shape}")

        self.X = torch.from_numpy(np.ascontiguousarray(X))  # (N,C,H,W)
        self.y = torch.from_numpy(np.ascontiguousarray(y))  # (N,1,H,W) or (N,H,W)

        # store stats as torch tensors
        self.X_mu = None if X_mu is None else torch.from_numpy(X_mu)
        self.X_sd = None if X_sd is None else torch.from_numpy(X_sd)
        self.y_mu = None if y_mu is None else torch.tensor(float(y_mu), dtype=torch.float32)
        self.y_sd = None if y_sd is None else torch.tensor(float(y_sd), dtype=torch.float32)

        # squeeze ONCE here (and nowhere else)
        if self.X_mu is not None and self.X_mu.ndim == 4:
            self.X_mu = self.X_mu.squeeze(0)  # (1,C,1,1) -> (C,1,1)
        if self.X_sd is not None and self.X_sd.ndim == 4:
            self.X_sd = self.X_sd.squeeze(0)  # (1,C,1,1) -> (C,1,1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]  # (C,H,W)
        y = self.y[idx]  # (1,H,W) or (H,W)

        if self.X_mu is not None and self.X_sd is not None:
            x = (x - self.X_mu) / self.X_sd

        if self.y_mu is not None and self.y_sd is not None:
            y = (y - self.y_mu) / self.y_sd

        return x, y


class RollingWindowManyToOneDataset(Dataset):
    """
    For UNet+LSTM (many-to-one) without materializing huge (Nw,T,C,H,W) arrays.

    Base inputs:
      X: (N, C, H, W)
      y: (N, 1, H, W) or (N, H, W)

    For a window length T:
      x_i = X[s : s+T] -> (T,C,H,W)
      y_i = y[s+T-1] (last) or y[s+T] (next)
    """

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        starts: Sequence[int],
        T: int,
        target: str = "last",  # "last" or "next"
        X_mu: Optional[np.ndarray] = None,  # ideally (1,C,1,1)
        X_sd: Optional[np.ndarray] = None,
        y_mu: Optional[float] = None,
        y_sd: Optional[float] = None,
    ):
        if X.ndim != 4:
            raise ValueError(f"RollingWindowManyToOneDataset expects X (N,C,H,W). Got {X.shape}")
        if y.ndim not in (3, 4):
            raise ValueError(f"RollingWindowManyToOneDataset expects y (N,H,W) or (N,1,H,W). Got {y.shape}")
        if target not in ("last", "next"):
            raise ValueError("target must be 'last' or 'next'")

        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.y = torch.from_numpy(np.ascontiguousarray(y))
        self.starts = np.asarray(starts, dtype=np.int64)
        self.T = int(T)
        self.target = target

        self.X_mu = None if X_mu is None else torch.from_numpy(X_mu)
        self.X_sd = None if X_sd is None else torch.from_numpy(X_sd)
        self.y_mu = None if y_mu is None else torch.tensor(float(y_mu), dtype=torch.float32)
        self.y_sd = None if y_sd is None else torch.tensor(float(y_sd), dtype=torch.float32)

        # squeeze ONCE here (and nowhere else)
        if self.X_mu is not None and self.X_mu.ndim == 4:
            self.X_mu = self.X_mu.squeeze(0)  # (1,C,1,1) -> (C,1,1)
        if self.X_sd is not None and self.X_sd.ndim == 4:
            self.X_sd = self.X_sd.squeeze(0)  # (1,C,1,1) -> (C,1,1)

        N = self.X.shape[0]
        max_needed = self.T if self.target == "last" else self.T + 1
        if self.starts.size == 0:
            raise ValueError("starts is empty (no windows). Check N and T.")
        if self.starts.min() < 0 or self.starts.max() + max_needed > N:
            raise ValueError(f"Invalid starts range for N={N}, T={self.T}, target={self.target}.")

    def __len__(self) -> int:
        return self.starts.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = self.X[s : s + self.T]  # (T,C,H,W)

        t = s + (self.T - 1) if self.target == "last" else s + self.T
        y = self.y[t]  # (1,H,W) or (H,W)

        if self.X_mu is not None and self.X_sd is not None:
            x = (x - self.X_mu) / self.X_sd

        if self.y_mu is not None and self.y_sd is not None:
            y = (y - self.y_mu) / self.y_sd

        return x, y