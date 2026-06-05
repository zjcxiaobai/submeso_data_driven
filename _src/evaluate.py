#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_utils import build_dataloaders
from engine.trainer import build_loss, evaluate
from models import build_model
from utils import get_device, load_config, project_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained regression model.")
    parser.add_argument("--config", type=Path, default=project_dir() / "configs/unet_regression.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_dir()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    checkpoint_path = args.checkpoint or Path(config["train"]["best_checkpoint"])

    _, _, test_loader, _ = build_dataloaders(config, root, int(config.get("seed", 42)))
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    metrics = evaluate(model, test_loader, build_loss(config.get("loss", {"name": "mse"})), device)
    print({key: round(value, 8) for key, value in metrics.items()}, flush=True)


if __name__ == "__main__":
    main()
