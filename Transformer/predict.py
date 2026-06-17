#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_utils.npy_dataset import NormStats
from models import build_model
from utils import get_device, load_config, project_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction for selected samples.")
    parser.add_argument("--config", type=Path, default=project_dir() / "configs/unet_regression.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=project_dir() / "outputs/predictions.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_dir()
    config = load_config(args.config)
    data_cfg = config["data"]
    device = get_device(config.get("device", "auto"))
    checkpoint_path = args.checkpoint or Path(config["train"]["best_checkpoint"])

    stats_npz = np.load(root / data_cfg["stats_path"])
    stats = NormStats(
        x_mean=stats_npz["x_mean"],
        x_std=stats_npz["x_std"],
        y_mean=stats_npz["y_mean"],
        y_std=stats_npz["y_std"],
    )
    x_all = np.load(root / data_cfg["x_path"], mmap_mode="r")
    x = np.asarray(x_all[args.indices], dtype=np.float32)
    if data_cfg.get("normalize_x", True):
        x = (x - stats.x_mean) / stats.x_std

    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
    if data_cfg.get("normalize_y", True):
        pred = pred * stats.y_std + stats.y_mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, indices=np.array(args.indices), prediction=pred)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
