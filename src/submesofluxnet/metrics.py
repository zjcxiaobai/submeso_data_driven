"""Metrics for spatial heat-flux predictions."""

from __future__ import annotations

import numpy as np


def r2_score(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Return global and snapshot-level coefficients of determination."""
    residual = pred - target
    sample_axes = tuple(range(1, target.ndim))
    sample_sse = np.sum(residual**2, axis=sample_axes)
    sample_mean = np.mean(target, axis=sample_axes, keepdims=True)
    sample_sst = np.sum((target - sample_mean) ** 2, axis=sample_axes)
    sample_r2 = np.full(sample_sse.shape, np.nan, dtype=np.float64)
    np.divide(sample_sse, sample_sst, out=sample_r2, where=sample_sst > 0)
    sample_r2[sample_sst > 0] = 1.0 - sample_r2[sample_sst > 0]
    overall = 1.0 - float(np.sum(residual**2)) / float(np.sum((target - np.mean(target)) ** 2))
    return overall, sample_r2
