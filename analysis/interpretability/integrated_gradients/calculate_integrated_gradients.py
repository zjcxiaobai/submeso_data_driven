#!/usr/bin/env python3
"""Calculate grid-cell Integrated Gradients for the final SubmesoFluxNet model."""

from pathlib import Path

import numpy as np
import torch

from submesofluxnet.config import PROJECT_ROOT, load_config
from submesofluxnet.data import TemporalSequenceDataset, centers_in_block, load_stats, split_raw_blocks
from submesofluxnet.model import build_model
from submesofluxnet.runtime import get_device, seed_everything


HERE = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "config/config.yaml"
CHECKPOINT = PROJECT_ROOT / "results/model/temporal_bottleneck.pt"
PREDICTIONS = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
STATS = PROJECT_ROOT / "results/model/norm_stats.npz"
OUTPUT = HERE / "top10_gridcell_ig.npz"

FEATURE_NAMES = [
    "U velocity",
    "V velocity",
    "W velocity",
    "Vorticity",
    "Strain",
    "Potential temperature",
    "Density",
    "KPP boundary layer depth",
]
STEPS = 64
IG_BATCH_SIZE = 16
FLUX_QUANTILE = 0.10
NUM_SNAPSHOTS = 90
DX_KM = 50.0
DY_KM = 50.0


def denormalize(values: torch.Tensor, stats) -> torch.Tensor:
    mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=values.device)
    return values * std + mean


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_y: int,
    target_x: int,
    stats,
) -> np.ndarray:
    """Attribute one predicted grid cell relative to a zero normalized baseline."""
    total_gradient = torch.zeros_like(x)

    for start in range(1, STEPS + 1, IG_BATCH_SIZE):
        stop = min(start + IG_BATCH_SIZE - 1, STEPS)
        alpha = torch.linspace(start / STEPS, stop / STEPS, stop - start + 1, device=x.device)
        scaled = (alpha[:, None, None, None, None] * x).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        prediction = denormalize(model(scaled), stats)
        prediction[:, 0, target_y, target_x].sum().backward()
        total_gradient += scaled.grad.detach().sum(dim=0, keepdim=True)

    return (x * total_gradient / STEPS)[0].cpu().numpy()


def select_snapshots(test_indices: np.ndarray) -> tuple[np.ndarray, float]:
    with np.load(PREDICTIONS) as data:
        sample_index = data["sample_index"].astype(np.int64)
        prediction = data["prediction"]

    mean_flux = np.mean(np.abs(prediction), axis=(1, 2, 3))
    keep = np.isin(sample_index, test_indices)
    sample_index = sample_index[keep]
    mean_flux = mean_flux[keep]

    threshold = float(np.quantile(mean_flux, 1.0 - FLUX_QUANTILE))
    candidates = np.sort(sample_index[mean_flux >= threshold])
    positions = np.rint(np.linspace(0, len(candidates) - 1, NUM_SNAPSHOTS)).astype(int)
    return candidates[positions], threshold


def target_cells(field: np.ndarray) -> list[tuple[int, int]]:
    width = field.shape[1]
    flat_indices = [
        np.nanargmax(field),
        np.nanargmin(field),
        np.nanargmin(np.abs(field)),
    ]
    return [(int(index // width), int(index % width)) for index in flat_indices]


def recenter(image: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    height, width = image.shape
    shift_y = height // 2 - target_y
    shifted = np.roll(image, width // 2 - target_x, axis=1)
    centered = np.full_like(shifted, np.nan)

    source_start = max(0, -shift_y)
    source_stop = min(height, height - shift_y)
    destination_start = source_start + shift_y
    destination_stop = source_stop + shift_y
    centered[destination_start:destination_stop] = shifted[source_start:source_stop]
    return centered


def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def main() -> None:
    config = load_config(CONFIG)
    seed_everything(int(config["seed"]))
    device = get_device(config["device"])
    offsets = [int(value) for value in config["data"]["temporal_offsets"]]

    x_path = PROJECT_ROOT / config["data"]["x_path"]
    y_path = PROJECT_ROOT / config["data"]["y_path"]
    stats = load_stats(STATS)
    n_samples = np.load(x_path, mmap_mode="r").shape[0]
    test_block = split_raw_blocks(n_samples, config["data"]["split"])[2]
    test_indices = centers_in_block(test_block, offsets)
    dataset = TemporalSequenceDataset(
        x_path,
        y_path,
        temporal_offsets=offsets,
        stats=stats,
        normalize_x=True,
        normalize_y=True,
    )
    selected_indices, flux_threshold = select_snapshots(test_indices)

    model = build_model(config).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    first_x, _ = dataset[int(selected_indices[0])]
    _, channels, height, width = first_x.shape
    centered_sum = np.zeros((channels, height, width), dtype=np.float64)
    centered_count = np.zeros_like(centered_sum)
    lag_sum = np.zeros((len(offsets), channels), dtype=np.float64)
    signed_sum = np.zeros((channels, height, width), dtype=np.float64)
    signed_count = np.zeros((height, width), dtype=np.int64)

    for snapshot_number, sample_index in enumerate(selected_indices, start=1):
        x, _ = dataset[int(sample_index)]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            field = denormalize(model(x), stats)[0, 0].cpu().numpy()

        for target_y, target_x in target_cells(field):
            attribution = integrated_gradients(model, x, target_y, target_x, stats)
            absolute = np.abs(attribution)
            lag_sum += absolute.sum(axis=(2, 3))
            signed_sum[:, target_y, target_x] += attribution.sum(axis=(0, 2, 3))
            signed_count[target_y, target_x] += 1

            feature_maps = absolute.sum(axis=0)
            for feature in range(channels):
                centered = recenter(feature_maps[feature], target_y, target_x)
                valid = np.isfinite(centered)
                centered_sum[feature, valid] += centered[valid]
                centered_count[feature, valid] += 1

        if snapshot_number % 5 == 0:
            print(f"Processed {snapshot_number}/{NUM_SNAPSHOTS} snapshots", flush=True)

    target_count = NUM_SNAPSHOTS * 3
    np.savez_compressed(
        OUTPUT,
        selected_sample_indices=selected_indices,
        flux_threshold=np.asarray(flux_threshold),
        target_centered_feature_attribution=divide(centered_sum, centered_count).astype(np.float32),
        lag_feature_attribution=(lag_sum / target_count).astype(np.float32),
        signed_target_feature_attribution=divide(signed_sum, signed_count[None]).astype(np.float32),
        signed_target_count=signed_count,
        feature_names=np.asarray(FEATURE_NAMES),
        temporal_offsets=np.asarray(offsets),
        dx_km=np.asarray(DX_KM),
        dy_km=np.asarray(DY_KM),
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
