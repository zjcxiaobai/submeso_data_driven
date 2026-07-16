#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DEFAULT_TUNING_CONFIG = Path(__file__).resolve().with_name("loss_function_tunning_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune loss functions for no-shuffle upper 40x20 UNet with base_channels=128."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TUNING_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print planned loss-function trials without training.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def best_learning_rate(search_cfg: dict[str, Any]) -> float:
    explicit_lr = search_cfg.get("learning_rate", "auto")
    if explicit_lr != "auto":
        return float(explicit_lr)

    summary_csv = resolve_project_path(search_cfg["learning_rate_summary_csv"])
    metric = str(search_cfg.get("learning_rate_metric", "val_r2"))
    best_row = None
    best_value = -float("inf")
    with summary_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            metric_value = finite_float(row.get(metric, ""))
            learning_rate = finite_float(row.get("learning_rate", ""))
            if metric_value is None or learning_rate is None:
                continue
            if metric_value > best_value:
                best_value = metric_value
                best_row = row

    if best_row is None:
        raise ValueError(f"No finite {metric!r} and learning_rate values found in {summary_csv}")
    return float(best_row["learning_rate"])


def loss_label(loss_cfg: dict[str, Any]) -> str:
    name = str(loss_cfg["name"])
    extras = {key: value for key, value in loss_cfg.items() if key != "name"}
    if not extras:
        return name
    suffix = ",".join(f"{key}={value}" for key, value in extras.items())
    return f"{name}({suffix})"


def make_train_config(
    search_cfg: dict[str, Any],
    *,
    loss_cfg: dict[str, Any],
    learning_rate: float,
    trial_index: int,
) -> dict[str, Any]:
    from utils import load_config

    base_config_path = resolve_project_path(search_cfg.get("base_config", "configs/unet_no_shuffle_upper_40x20.yaml"))
    config = copy.deepcopy(load_config(base_config_path))
    config.setdefault("model", {})["base_channels"] = int(search_cfg.get("base_channels", 128))
    config.setdefault("optimizer", {})["lr"] = float(learning_rate)
    config["loss_tuning"] = copy.deepcopy(loss_cfg)
    config["trial_index"] = trial_index

    train_cfg = config.setdefault("train", {})
    train_cfg["epochs"] = int(search_cfg.get("epochs", train_cfg.get("epochs", 300)))
    train_cfg["early_stopping_patience"] = int(
        search_cfg.get("early_stopping_patience", train_cfg.get("early_stopping_patience", 15))
    )
    train_cfg["checkpoint_dir"] = ""
    train_cfg["best_checkpoint"] = ""
    train_cfg["last_checkpoint"] = ""
    train_cfg["history_csv"] = ""
    train_cfg["config_snapshot"] = ""
    return config


def build_tuning_loss(loss_cfg: dict[str, Any]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class WeightedMSELoss(nn.Module):
        def __init__(self, power: float = 1.0, eps: float = 1.0e-6) -> None:
            super().__init__()
            self.power = power
            self.eps = eps

        def forward(self, pred, target):
            scale = target.detach().abs()
            scale = scale / scale.mean().clamp_min(self.eps)
            weight = scale.clamp_min(self.eps).pow(self.power)
            return torch.mean(weight * (pred - target) ** 2)

    class GradientLoss(nn.Module):
        def forward(self, pred, target):
            pred_dx = torch.roll(pred, shifts=-1, dims=-1) - pred
            target_dx = torch.roll(target, shifts=-1, dims=-1) - target
            pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
            target_dy = target[..., 1:, :] - target[..., :-1, :]
            return F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)

    class SpectralLoss(nn.Module):
        def forward(self, pred, target):
            pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
            target_fft = torch.fft.rfft2(target.float(), norm="ortho")
            pred_amp = torch.log1p(torch.abs(pred_fft))
            target_amp = torch.log1p(torch.abs(target_fft))
            return F.mse_loss(pred_amp, target_amp)

    class MSEPlusGradientLoss(nn.Module):
        def __init__(self, gradient_weight: float) -> None:
            super().__init__()
            self.gradient_weight = gradient_weight
            self.gradient_loss = GradientLoss()

        def forward(self, pred, target):
            return F.mse_loss(pred, target) + self.gradient_weight * self.gradient_loss(pred, target)

    name = str(loss_cfg["name"]).lower()
    if name == "mse":
        return nn.MSELoss()
    if name in {"mae", "l1"}:
        return nn.L1Loss()
    if name in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss()
    if name == "weighted_mse":
        return WeightedMSELoss(
            power=float(loss_cfg.get("weighted_power", 1.0)),
            eps=float(loss_cfg.get("weighted_eps", 1.0e-6)),
        )
    if name == "gradient_loss":
        return GradientLoss()
    if name == "spectral_loss":
        return SpectralLoss()
    if name in {"mse_plus_gradient", "mse+gradient", "mse_gradient"}:
        return MSEPlusGradientLoss(gradient_weight=float(loss_cfg.get("gradient_weight", 0.1)))
    raise ValueError(f"Unknown loss function: {name}")


def metric_improved(value: float, best_value: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"Unknown early_stopping_mode: {mode}")


def train_one_loss(
    config: dict[str, Any],
    *,
    search_cfg: dict[str, Any],
    trial_seed: int,
    loss_cfg: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from torch.amp import GradScaler, autocast

    from data_utils import build_dataloaders
    from engine.trainer import build_optimizer, build_scheduler, evaluate
    from models import build_model
    from utils import get_device, seed_everything

    seed_everything(trial_seed)
    device = get_device(config.get("device", "auto"))
    train_loader, val_loader, test_loader, _ = build_dataloaders(config, PROJECT_DIR, trial_seed)

    model = build_model(config["model"]).to(device)
    loss_fn = build_tuning_loss(loss_cfg)
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    epochs = int(config["train"].get("epochs", 300))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}), epochs)
    amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    grad_clip_norm = config["train"].get("grad_clip_norm")
    patience = int(config["train"].get("early_stopping_patience", 15))
    scaler = GradScaler(enabled=amp)

    early_metric = str(search_cfg.get("early_stopping_metric", "val_r2"))
    early_mode = str(search_cfg.get("early_stopping_mode", "max"))
    best_value = -float("inf") if early_mode == "max" else float("inf")
    best_epoch = 0
    best_val_loss = float("nan")
    best_state = None
    stale_epochs = 0
    status = "ok"

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp):
                pred = model(x)
                loss = loss_fn(pred, y)

            if not torch.isfinite(loss):
                status = "diverged"
                break

            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

        if status == "diverged":
            break

        val_metrics = evaluate(model, val_loader, loss_fn, device, amp=amp)
        val_loss = val_metrics["loss"]
        val_r2 = val_metrics["r2"]
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if not math.isfinite(val_loss) or not math.isfinite(val_r2):
            status = "diverged_after_best" if best_state is not None else "diverged"
            break

        current_value = val_r2 if early_metric == "val_r2" else val_loss
        if metric_improved(current_value, best_value, early_mode):
            best_value = current_value
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"trial={config['trial_index']:03d} loss={loss_label(loss_cfg)} epoch={epoch:04d} "
            f"val_loss={val_loss:.6g} val_r2={val_r2:.6g}",
            flush=True,
        )

        if stale_epochs >= patience:
            break

    if best_state is None:
        return {
            "status": status,
            "best_epoch": "",
            "best_val_loss": "",
            "train_r2": float("nan"),
            "val_r2": float("nan"),
            "test_r2": float("nan"),
        }

    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, loss_fn, device, amp=amp)
    val_metrics = evaluate(model, val_loader, loss_fn, device, amp=amp)
    test_metrics = evaluate(model, test_loader, loss_fn, device, amp=amp)
    return {
        "status": status,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_r2": train_metrics["r2"],
        "val_r2": val_metrics["r2"],
        "test_r2": test_metrics["r2"],
    }


def append_summary_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "loss_name",
        "loss_label",
        "loss_params",
        "learning_rate",
        "base_channels",
        "status",
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
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path

    search_cfg = load_yaml(config_path)
    learning_rate = best_learning_rate(search_cfg)
    summary_csv = resolve_project_path(search_cfg.get("summary_csv", "hyperparameter_tunning/loss_function/summary.csv"))
    base_channels = int(search_cfg.get("base_channels", 128))
    base_seed = int(search_cfg.get("seed", 42))
    losses = search_cfg.get("loss_functions", [])
    if not losses:
        raise ValueError("loss_functions must not be empty")

    for trial, loss_cfg in enumerate(losses, start=1):
        label = loss_label(loss_cfg)
        if args.dry_run:
            print(f"trial={trial:03d} loss={label} lr={learning_rate:.8g} base_channels={base_channels}")
            continue

        config = make_train_config(
            search_cfg,
            loss_cfg=loss_cfg,
            learning_rate=learning_rate,
            trial_index=trial,
        )
        metrics = train_one_loss(config, search_cfg=search_cfg, trial_seed=base_seed + trial, loss_cfg=loss_cfg)
        row = {
            "trial": trial,
            "loss_name": loss_cfg["name"],
            "loss_label": label,
            "loss_params": json.dumps({key: value for key, value in loss_cfg.items() if key != "name"}),
            "learning_rate": learning_rate,
            "base_channels": base_channels,
            **metrics,
        }
        append_summary_row(summary_csv, row)
        print(
            f"finished trial={trial:03d} loss={label} status={metrics['status']} "
            f"train_r2={metrics['train_r2']} val_r2={metrics['val_r2']} test_r2={metrics['test_r2']}",
            flush=True,
        )

    if not args.dry_run:
        print(f"Wrote summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
