#!/usr/bin/env python3
"""Calculate feature importance by replacing each input channel with its training mean."""

from pathlib import Path

import numpy as np
import torch

from submesofluxnet.config import load_config
from submesofluxnet.data import make_loaders
from submesofluxnet.model import build_model
from submesofluxnet.runtime import get_device, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT_ROOT / "config/config.yaml"
CHECKPOINT = PROJECT_ROOT / "results/model/temporal_bottleneck.pt"
OUTPUT = Path(__file__).resolve().with_name("model_analysis_feature_importance.npz")

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


def sample_r2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    sse = torch.sum((prediction - target) ** 2, dim=1)
    sst = torch.sum((target - target.mean(dim=1, keepdim=True)) ** 2, dim=1)
    return 1.0 - sse / sst


def main() -> None:
    config = load_config(CONFIG)
    config["output"]["stats"] = "results/model/norm_stats.npz"
    seed_everything(int(config["seed"]))
    device = get_device(config["device"])

    _, _, test_loader, _ = make_loaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    baseline_batches = []
    ablated_batches = [[] for _ in FEATURE_NAMES]

    with torch.no_grad():
        for x, target in test_loader:
            x = x.to(device)
            target = target.to(device)
            baseline_batches.append(sample_r2(model(x), target).cpu().numpy())

            for feature in range(len(FEATURE_NAMES)):
                x_ablated = x.clone()
                x_ablated[:, :, feature] = 0.0
                ablated_batches[feature].append(sample_r2(model(x_ablated), target).cpu().numpy())

    sample_index = np.asarray(test_loader.dataset.indices, dtype=np.int64)
    baseline_r2 = np.concatenate(baseline_batches)
    feature_r2 = np.stack([np.concatenate(values) for values in ablated_batches])
    feature_drop = np.mean(baseline_r2[None] - feature_r2, axis=1)

    np.savez_compressed(
        OUTPUT,
        sample_index=sample_index,
        baseline_r2=baseline_r2,
        feature_r2=feature_r2,
        feature_drop=feature_drop,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
