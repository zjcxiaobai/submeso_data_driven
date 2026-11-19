"""Evaluation metrics for SubmesoFluxNet."""

import numpy as np


def r2_score(prediction: np.ndarray, target: np.ndarray):
    """Calculate overall and snapshot-level R2 scores."""
    error = (prediction - target) ** 2

    overall_r2 = 1.0 - error.sum() / ((target - target.mean()) ** 2).sum()

    axes = tuple(range(1, target.ndim))
    target_mean = target.mean(axis=axes, keepdims=True)
    sample_r2 = 1.0 - error.sum(axis=axes) / (
        (target - target_mean) ** 2
    ).sum(axis=axes)

    return float(overall_r2), sample_r2
