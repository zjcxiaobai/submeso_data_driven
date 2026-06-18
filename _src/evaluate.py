#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data_utils import build_dataloaders
from engine.trainer import build_loss, evaluate
from models import build_model
from utils import get_device, load_config, project_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained regression model.")
    parser.add_argument("--config", type=Path, default=project_dir() / "configs/unet_regression.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=None,
        help="Path for denormalized test predictions/targets as a compressed .npz file.",
    )
    parser.add_argument(
        "--r2-series-output",
        type=Path,
        default=None,
        help="Path for per-test-sample R2 time series as a CSV file.",
    )
    return parser.parse_args()


def denormalize_y(values: torch.Tensor, stats, enabled: bool, device: torch.device) -> torch.Tensor:
    if not enabled:
        return values
    y_mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=device)
    y_std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=device)
    return values * y_std + y_mean


def sample_r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = pred - target
    reduce_axes = tuple(range(1, target.ndim))
    sse = np.sum(residual**2, axis=reduce_axes)
    target_mean = np.mean(target, axis=reduce_axes, keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=reduce_axes)
    r2 = np.full_like(sse, np.nan, dtype=np.float64)
    np.divide(sse, sst, out=r2, where=sst > 0.0)
    r2[sst > 0.0] = 1.0 - r2[sst > 0.0]
    return r2


def overall_r2(pred: np.ndarray, target: np.ndarray) -> float:
    sse = float(np.sum((pred - target) ** 2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    return float(1.0 - sse / sst) if sst > 0.0 else float("nan")


def save_test_outputs(model, test_loader, stats, config: dict, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    output_dir = Path(config["train"]["history_csv"]).parent
    predictions_output = args.predictions_output or output_dir / "test_predictions_denorm.npz"
    r2_series_output = args.r2_series_output or output_dir / "test_r2_series.csv"
    normalize_y = bool(config["data"].get("normalize_y", True))

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            pred = denormalize_y(pred, stats, normalize_y, device)
            y = denormalize_y(y, stats, normalize_y, device)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    r2_series = sample_r2(prediction, target)
    reduce_axes = tuple(range(1, target.ndim))
    mse_series = np.mean((prediction - target) ** 2, axis=reduce_axes)
    mae_series = np.mean(np.abs(prediction - target), axis=reduce_axes)
    sample_indices = np.asarray(test_loader.dataset.indices, dtype=np.int64)

    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    denorm_metrics = {
        "mse": float(np.mean((prediction - target) ** 2)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "r2": overall_r2(prediction, target),
    }

    np.savez_compressed(
        predictions_output,
        sample_index=sample_indices,
        prediction=prediction,
        target=target,
        r2=r2_series,
        mse=mse_series,
        mae=mae_series,
        overall_mse=np.asarray(denorm_metrics["mse"]),
        overall_mae=np.asarray(denorm_metrics["mae"]),
        overall_r2=np.asarray(denorm_metrics["r2"]),
    )

    r2_series_output.parent.mkdir(parents=True, exist_ok=True)
    with r2_series_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "r2", "mse", "mae"])
        writer.writeheader()
        for test_position, (sample_index, r2, mse, mae) in enumerate(
            zip(sample_indices, r2_series, mse_series, mae_series)
        ):
            writer.writerow(
                {
                    "test_position": test_position,
                    "sample_index": int(sample_index),
                    "r2": float(r2),
                    "mse": float(mse),
                    "mae": float(mae),
                }
            )

    print(f"Saved denormalized test predictions: {predictions_output}", flush=True)
    print(f"Saved per-sample R2 series: {r2_series_output}", flush=True)
    return denorm_metrics


def main() -> None:
    args = parse_args()
    root = project_dir()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    checkpoint_path = args.checkpoint or Path(config["train"]["best_checkpoint"])

    _, _, test_loader, stats = build_dataloaders(config, root, int(config.get("seed", 42)))
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(model, test_loader, build_loss(config.get("loss", {"name": "mse"})), device)
    print({key: round(value, 8) for key, value in metrics.items()}, flush=True)
    denorm_metrics = save_test_outputs(model, test_loader, stats, config, device, args)
    print({f"denorm_{key}": round(value, 8) for key, value in denorm_metrics.items()}, flush=True)


if __name__ == "__main__":
    main()
