#!/usr/bin/env python3
"""Apply the trained upper-domain model to the lower-domain dataset."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from submesofluxnet.config import PROJECT_ROOT, load_config
from submesofluxnet.data import TemporalSequenceDataset, centers_in_block, load_stats
from submesofluxnet.model import build_model
from submesofluxnet.runtime import get_device


HERE = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "config/config.yaml"
CHECKPOINT = PROJECT_ROOT / "results/model/temporal_bottleneck.pt"
STATS = PROJECT_ROOT / "results/model/norm_stats.npz"
LOWER_DATA = PROJECT_ROOT / "data/processed/lower"
PREDICTIONS_OUTPUT = HERE / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz"
R2_OUTPUT = HERE / "upper_model_on_lower_40x20_model_analysis_r2_series.csv"


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction.reshape(len(prediction), -1)
    target = target.reshape(len(target), -1)
    sse = np.sum((prediction - target) ** 2, axis=1)
    sst = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


def denormalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std[None, ...] + mean[None, ...]


def main() -> None:
    config = load_config(CONFIG)
    data_cfg = config["data"]
    offsets = [int(value) for value in data_cfg["temporal_offsets"]]
    stats = load_stats(STATS)

    dataset = TemporalSequenceDataset(
        LOWER_DATA / "X.npy",
        LOWER_DATA / "y.npy",
        temporal_offsets=offsets,
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=bool(data_cfg.get("normalize_y", True)),
    )
    sample_indices = centers_in_block((0, len(dataset)), offsets)
    time_index = np.load(LOWER_DATA / "time_index.npy")[sample_indices]
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=0,
    )

    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())

    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    if data_cfg.get("normalize_y", True):
        prediction = denormalize(prediction, stats.y_mean, stats.y_std)
        target = denormalize(target, stats.y_mean, stats.y_std)

    r2 = sample_r2(prediction, target)

    np.savez_compressed(
        PREDICTIONS_OUTPUT,
        sample_index=sample_indices,
        time_index=time_index,
        prediction=prediction,
        target=target,
    )

    with R2_OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["test_position", "sample_index", "time_index", "r2"])
        writer.writerows(zip(range(len(r2)), sample_indices, time_index, r2))

    print(f"Saved {PREDICTIONS_OUTPUT}")
    print(f"Saved {R2_OUTPUT}")


if __name__ == "__main__":
    main()
