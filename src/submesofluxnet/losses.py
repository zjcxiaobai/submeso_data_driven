"""Prediction and latent-space losses."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_loss(name: str) -> nn.Module:
    """Construct the configured pointwise prediction loss."""
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in {"mae", "l1"}:
        return nn.L1Loss()
    if name in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {name}")


def latent_smoothness(features: torch.Tensor) -> torch.Tensor:
    """Penalize first differences of the latent maps in both spatial directions."""
    dx = features[..., :, 1:] - features[..., :, :-1]
    dy = features[..., 1:, :] - features[..., :-1, :]
    return torch.mean(dx**2) + torch.mean(dy**2)


def latent_decorrelation(features: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Penalize off-diagonal correlations among latent channels."""
    channels = features.shape[1]
    if channels <= 1:
        return features.new_tensor(0.0)
    flat = features.permute(1, 0, 2, 3).reshape(channels, -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.std(dim=1, keepdim=True).clamp_min(eps)
    correlation = flat @ flat.t() / flat.shape[1]
    identity = torch.eye(channels, dtype=correlation.dtype, device=correlation.device)
    return torch.mean((correlation - identity) ** 2)


def latent_regularization(
    features: torch.Tensor,
    *,
    smooth_weight: float,
    corr_weight: float,
) -> torch.Tensor:
    """Combine the optional latent-map regularization terms."""
    regularization = features.new_tensor(0.0)
    if smooth_weight > 0.0:
        regularization = regularization + smooth_weight * latent_smoothness(features)
    if corr_weight > 0.0:
        regularization = regularization + corr_weight * latent_decorrelation(features)
    return regularization
