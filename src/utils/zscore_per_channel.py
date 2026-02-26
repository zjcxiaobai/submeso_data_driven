# general modules
import numpy as np

# ------------------------------- Z-score normalization -------------------------------------
def _fit_z_norm_X(X_train):
    """
    X_train: (N, C, H, W) -> per-channel mu, sd from TRAIN.
    This function computes the z-score normalization of X
    Which transform the mean = 0, and standard deviation = 1
    mean and std computed per channel [1, C, 1, 1]
    """
    # mean/std over N and all spatial dims, per channel
    axes = tuple(i for i in range(X_train.ndim) if i not in (1,))  # normalize all dims except channel dim
    mu = X_train.mean(axis=axes, keepdims=True)                  # shape (1, C, 1, 1)
    sd = X_train.std(axis=axes, ddof=1, keepdims=True)    # shape (1, C, 1, 1)
    return mu.astype(np.float32), sd.astype(np.float32)

def _fit_z_norm_y(y_train):
    """
    y_train: (N, 1, H, W)
    """
    mu = y_train.mean()
    sd = y_train.std(ddof=1)
    return np.float32(mu), np.float32(sd)

#  apply the az-score normalization to X_train, X_val, X_test
def _apply_z_norm_X(X, mu, sd):
    return (X - mu) / sd

def _apply_z_norm_y(y, mu, sd):
    return (y - mu) / sd