#!/usr/bin/env python3
"""Evaluate the current upper_40x20 model_analysis checkpoint on lower_40x20 samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils.npy_dataset import (  # noqa: E402
    NormStats,
    NpyRegressionDataset,
    load_or_compute_stats,
    parse_temporal_offsets,
    split_indices,
    valid_center_indices,
)
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an upper_40x20 model on the complete lower_40x20 domain time series."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "analysis/model_analysis/outputs/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Default: analysis/model_analysis/outputs/checkpoints/best.pt.",
    )
    parser.add_argument("--eval-x", type=Path, default=PROJECT_DIR / "data/lower_40x20/X.npy")
    parser.add_argument("--eval-y", type=Path, default=PROJECT_DIR / "data/lower_40x20/y.npy")
    parser.add_argument("--stats-x", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/X.npy")
    parser.add_argument("--stats-y", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/y.npy")
    parser.add_argument("--stats-path", type=Path, default=PROJECT_DIR / "analysis/model_analysis/outputs/norm_stats.npz")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_model_analysis",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_stats(path: Path) -> NormStats:
    if not path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {path}")
    with np.load(path) as data:
        return NormStats(
            x_mean=np.asarray(data["x_mean"], dtype=np.float32),
            x_std=np.asarray(data["x_std"], dtype=np.float32),
            y_mean=np.asarray(data["y_mean"], dtype=np.float32),
            y_std=np.asarray(data["y_std"], dtype=np.float32),
        )


def denormalize_y(values: torch.Tensor, stats: NormStats, enabled: bool, device: torch.device) -> torch.Tensor:
    if not enabled:
        return values
    y_mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=device)
    y_std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=device)
    return values * y_std + y_mean


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = prediction - target
    reduce_axes = tuple(range(1, target.ndim))
    sse = np.sum(residual**2, axis=reduce_axes)
    target_mean = np.mean(target, axis=reduce_axes, keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=reduce_axes)
    r2 = np.full(sse.shape, np.nan, dtype=np.float64)
    np.divide(sse, sst, out=r2, where=sst > 0.0)
    r2[sst > 0.0] = 1.0 - r2[sst > 0.0]
    return r2


def overall_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    sse = float(np.sum((prediction - target) ** 2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - sse / sst if sst > 0.0 else float("nan")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    data_cfg = config["data"]
    device = get_device(args.device or config.get("device", "auto"))
    batch_size = int(args.batch_size or data_cfg.get("batch_size", 64))
    normalize_y = bool(data_cfg.get("normalize_y", True))
    temporal_offsets = parse_temporal_offsets(data_cfg)

    stats_probe = np.load(args.stats_x, mmap_mode="r")
    stats_centers = valid_center_indices(stats_probe.shape[0], temporal_offsets)
    train_pos, _, _ = split_indices(stats_centers.shape[0], data_cfg["split"], seed)
    stats_scope = str(data_cfg.get("stats_scope", "train")).lower()
    stats_idx = stats_centers[train_pos] if stats_scope == "train" else stats_centers
    stats = load_or_compute_stats(
        args.stats_x,
        args.stats_y,
        stats_idx,
        args.stats_path,
        temporal_offsets=temporal_offsets,
        stats_scope=stats_scope,
    )
    dataset = NpyRegressionDataset(
        args.eval_x,
        args.eval_y,
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=normalize_y,
        temporal_offsets=temporal_offsets,
    )
    sample_indices = valid_center_indices(len(dataset), temporal_offsets)
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = args.checkpoint or (PROJECT_DIR / "analysis/model_analysis/outputs/checkpoints/best.pt")
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            pred = denormalize_y(pred, stats, normalize_y, device)
            y = denormalize_y(y, stats, normalize_y, device)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    r2 = sample_r2(prediction, target)
    reduce_axes = tuple(range(1, target.ndim))
    mse = np.mean((prediction - target) ** 2, axis=reduce_axes)
    mae = np.mean(np.abs(prediction - target), axis=reduce_axes)
    metrics = {
        "overall_r2": overall_r2(prediction, target),
        "mean_per_sample_r2": float(np.nanmean(r2)),
        "median_per_sample_r2": float(np.nanmedian(r2)),
        "overall_mse": float(np.mean((prediction - target) ** 2)),
        "overall_mae": float(np.mean(np.abs(prediction - target))),
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictions_output = args.output_prefix.with_name(args.output_prefix.name + "_predictions_denorm.npz")
    r2_output = args.output_prefix.with_name(args.output_prefix.name + "_r2_series.csv")
    summary_output = args.output_prefix.with_name(args.output_prefix.name + "_summary.txt")

    np.savez_compressed(
        predictions_output,
        sample_index=sample_indices.astype(np.int64),
        prediction=prediction,
        target=target,
        r2=r2,
        mse=mse,
        mae=mae,
        overall_r2=np.asarray(metrics["overall_r2"]),
        overall_mse=np.asarray(metrics["overall_mse"]),
        overall_mae=np.asarray(metrics["overall_mae"]),
    )

    with r2_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "r2", "mse", "mae"])
        writer.writeheader()
        for test_position, (sample_index, sample_r2_value, sample_mse, sample_mae) in enumerate(
            zip(sample_indices, r2, mse, mae)
        ):
            writer.writerow(
                {
                    "test_position": test_position,
                    "sample_index": int(sample_index),
                    "r2": float(sample_r2_value),
                    "mse": float(sample_mse),
                    "mae": float(sample_mae),
                }
            )

    with summary_output.open("w") as handle:
        handle.write(f"Config: {args.config}\n")
        handle.write(f"Checkpoint: {checkpoint_path}\n")
        handle.write(f"Eval X: {args.eval_x}\n")
        handle.write(f"Eval y: {args.eval_y}\n")
        handle.write(f"Stats X: {args.stats_x}\n")
        handle.write(f"Stats y: {args.stats_y}\n")
        handle.write(f"Stats path: {args.stats_path}\n")
        handle.write(f"Temporal offsets: {temporal_offsets}\n")
        handle.write(f"Samples evaluated: {sample_indices.size}\n")
        handle.write(f"First sample_index: {int(sample_indices[0])}\n")
        handle.write(f"Last sample_index: {int(sample_indices[-1])}\n")
        for key, value in metrics.items():
            handle.write(f"{key}: {value:.8f}\n")

    print(f"Saved denormalized predictions: {predictions_output}", flush=True)
    print(f"Saved R2 series: {r2_output}", flush=True)
    print(f"Saved summary: {summary_output}", flush=True)
    print({key: round(value, 8) for key, value in metrics.items()}, flush=True)


if __name__ == "__main__":
    main()
