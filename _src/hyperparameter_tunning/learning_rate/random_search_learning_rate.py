#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))



DEFAULT_SEARCH_CONFIG = Path(__file__).resolve().with_name("learning_rate_tunning_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-search learning rate for no-shuffle upper 40x20 UNet with base_channels=128."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SEARCH_CONFIG,
        help="Learning-rate tuning YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampled learning rates without training.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def sample_learning_rates(search_cfg: dict[str, Any]) -> list[float]:
    rng = random.Random(int(search_cfg.get("seed", 42)))
    n_trials = int(search_cfg.get("n_trials", 20))
    lr_min = float(search_cfg.get("lr_min", 1.0e-5))
    lr_max = float(search_cfg.get("lr_max", 3.0e-3))
    sampling = str(search_cfg.get("sampling", "log_uniform")).lower()

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if lr_min <= 0.0 or lr_max <= 0.0 or lr_min >= lr_max:
        raise ValueError("Require 0 < lr_min < lr_max")

    if sampling == "log_uniform":
        log_min = math.log10(lr_min)
        log_max = math.log10(lr_max)
        return [10.0 ** rng.uniform(log_min, log_max) for _ in range(n_trials)]
    if sampling == "uniform":
        return [rng.uniform(lr_min, lr_max) for _ in range(n_trials)]
    raise ValueError(f"Unknown sampling: {sampling}")


def make_train_config(search_cfg: dict[str, Any], learning_rate: float, trial_index: int) -> dict[str, Any]:
    from utils import load_config

    base_config_path = resolve_project_path(search_cfg.get("base_config", "configs/unet_no_shuffle_upper_40x20.yaml"))
    config = load_config(base_config_path)
    config = copy.deepcopy(config)

    config.setdefault("model", {})["base_channels"] = int(search_cfg.get("base_channels", 128))
    config.setdefault("optimizer", {})["lr"] = float(learning_rate)

    train_cfg = config.setdefault("train", {})
    train_cfg["epochs"] = int(search_cfg.get("epochs", train_cfg.get("epochs", 300)))
    train_cfg["early_stopping_patience"] = int(
        search_cfg.get("early_stopping_patience", train_cfg.get("early_stopping_patience", 15))
    )

    # This script does not call fit(), so these paths are intentionally not used for checkpoints.
    train_cfg["checkpoint_dir"] = ""
    train_cfg["best_checkpoint"] = ""
    train_cfg["last_checkpoint"] = ""
    train_cfg["history_csv"] = ""
    train_cfg["config_snapshot"] = ""
    config["trial_index"] = trial_index
    return config


def train_one_trial(config: dict[str, Any], trial_seed: int) -> dict[str, float]:
    import torch
    from torch.amp import GradScaler, autocast

    from data_utils import build_dataloaders
    from engine.trainer import build_loss, build_optimizer, build_scheduler, evaluate
    from models import build_model
    from utils import get_device, seed_everything

    seed_everything(trial_seed)
    device = get_device(config.get("device", "auto"))
    train_loader, val_loader, test_loader, _ = build_dataloaders(config, PROJECT_DIR, trial_seed)

    model = build_model(config["model"]).to(device)
    loss_fn = build_loss(config.get("loss", {"name": "mse"}))
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    epochs = int(config["train"].get("epochs", 300))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}), epochs)
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    grad_clip_norm = config["train"].get("grad_clip_norm")
    patience = int(config["train"].get("early_stopping_patience", 15))
    scaler = GradScaler(enabled=amp)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

        val_metrics = evaluate(model, val_loader, loss_fn, device, amp=amp)
        val_loss = val_metrics["loss"]
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"trial={config['trial_index']:03d} epoch={epoch:04d} "
            f"lr={config['optimizer']['lr']:.6g} val_loss={val_loss:.6g} val_r2={val_metrics['r2']:.6g}",
            flush=True,
        )

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, loss_fn, device, amp=amp)
    val_metrics = evaluate(model, val_loader, loss_fn, device, amp=amp)
    test_metrics = evaluate(model, test_loader, loss_fn, device, amp=amp)
    return {
        "best_epoch": float(best_epoch),
        "best_val_loss": best_val_loss,
        "train_r2": train_metrics["r2"],
        "val_r2": val_metrics["r2"],
        "test_r2": test_metrics["r2"],
    }


def append_summary_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "learning_rate",
        "base_channels",
        "best_epoch",
        "best_val_loss",
        "train_r2",
        "val_r2",
        "test_r2",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    search_config_path = args.config.expanduser()
    if not search_config_path.is_absolute():
        search_config_path = PROJECT_DIR / search_config_path

    search_cfg = load_yaml(search_config_path)
    learning_rates = sample_learning_rates(search_cfg)
    summary_csv = resolve_project_path(search_cfg.get("summary_csv", "hyperparameter_tunning/learning_rate/summary.csv"))
    base_channels = int(search_cfg.get("base_channels", 128))
    base_seed = int(search_cfg.get("seed", 42))

    for trial, learning_rate in enumerate(learning_rates, start=1):
        if args.dry_run:
            print(f"trial={trial:03d} lr={learning_rate:.8g} base_channels={base_channels}")
            continue

        config = make_train_config(search_cfg, learning_rate, trial)
        metrics = train_one_trial(config, base_seed + trial)
        row = {
            "trial": trial,
            "learning_rate": learning_rate,
            "base_channels": base_channels,
            **metrics,
        }
        append_summary_row(summary_csv, row)
        print(
            f"finished trial={trial:03d} lr={learning_rate:.6g} "
            f"train_r2={metrics['train_r2']:.6g} val_r2={metrics['val_r2']:.6g} "
            f"test_r2={metrics['test_r2']:.6g}",
            flush=True,
        )

    if args.dry_run:
        return
    print(f"Wrote summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
