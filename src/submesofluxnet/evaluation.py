"""Validation metrics and held-out test evaluation."""

from __future__ import annotations

import csv

import numpy as np
import torch

from .config import PROJECT_ROOT
from .data import NormStats, make_loaders
from .losses import build_loss
from .metrics import r2_score
from .model import build_model
from .runtime import get_device, seed_everything


def evaluate_metrics(model, loader, loss_fn, device):
    """Evaluate normalized loss and aggregate regression metrics."""
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            batch = x.shape[0]
            residual = (prediction - y).double()
            target = y.double()
            totals[0] += loss.detach().double() * batch
            totals[1] += torch.mean((prediction - y) ** 2).double() * batch
            totals[2] += torch.mean(torch.abs(prediction - y)).double() * batch
            totals[3] += batch
            totals[4] += torch.sum(residual**2)
            totals[5] += torch.sum(target)
            totals[6] += torch.sum(target**2)
            totals[7] += target.numel()
    denominator = totals[3].clamp_min(1.0)
    sst = totals[6] - totals[5] ** 2 / totals[7].clamp_min(1.0)
    r2 = 1.0 - totals[4] / sst if sst > 0 else torch.tensor(float("nan"), device=device)
    return {
        "loss": float((totals[0] / denominator).item()),
        "mse": float((totals[1] / denominator).item()),
        "mae": float((totals[2] / denominator).item()),
        "r2": float(r2.item()),
    }


def denormalize_y(values: np.ndarray, stats: NormStats) -> np.ndarray:
    """Restore the target field to physical units."""
    return values * stats.y_std[None, ...] + stats.y_mean[None, ...]


def evaluate(config: dict) -> None:
    """Evaluate the best checkpoint and save test predictions and metrics."""
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    _, val_loader, test_loader, stats = make_loaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(
        PROJECT_ROOT / config["train"]["checkpoint"],
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state"])
    loss_fn = build_loss(str(config["train"].get("loss", "huber")))
    val_metrics = evaluate_metrics(model, val_loader, loss_fn, device)

    model.eval()
    predictions = []
    targets = []
    latent_maps = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            prediction = model(x).cpu().numpy()
            latent = model.encode_features(x).cpu().numpy()
            predictions.append(denormalize_y(prediction, stats))
            targets.append(denormalize_y(y.numpy(), stats))
            latent_maps.append(latent)

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    latent_features = np.concatenate(latent_maps, axis=0)
    overall_r2, sample_r2 = r2_score(prediction, target)
    mse = np.mean((prediction - target) ** 2, axis=(1, 2, 3))
    mae = np.mean(np.abs(prediction - target), axis=(1, 2, 3))
    sample_indices = np.asarray(test_loader.dataset.indices, dtype=np.int64)

    prediction_path = PROJECT_ROOT / config["output"]["predictions"]
    r2_path = PROJECT_ROOT / config["output"]["r2_series"]
    summary_path = PROJECT_ROOT / config["output"]["summary"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        sample_index=sample_indices,
        latent_features=latent_features,
        prediction=prediction,
        target=target,
        r2=sample_r2,
        mse=mse,
        mae=mae,
        overall_r2=np.asarray(overall_r2),
        val_r2=np.asarray(val_metrics["r2"]),
    )
    with r2_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_position", "sample_index", "r2", "mse", "mae"])
        writer.writerows(
            zip(range(len(sample_indices)), sample_indices, sample_r2, mse, mae)
        )
    summary = {
        "val_r2": val_metrics["r2"],
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "test_r2": float(overall_r2),
        "mean_snapshot_r2": float(np.mean(sample_r2)),
        "test_mse": float(np.mean(mse)),
        "test_mae": float(np.mean(mae)),
        "latent_feature_shape": list(latent_features.shape),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        handle.write("\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n")
    print(
        {key: round(value, 8) for key, value in summary.items() if isinstance(value, float)},
        flush=True,
    )
    print(f"Saved {prediction_path}", flush=True)
    print(f"Saved {r2_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
