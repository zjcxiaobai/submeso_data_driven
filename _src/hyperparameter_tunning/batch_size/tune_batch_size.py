#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


DEFAULT_TUNING_CONFIG = Path(__file__).resolve().with_name("batch_size_tunning_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune batch sizes for a test-only variable-depth U-Net model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_TUNING_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Print planned batch-size trials without training.")
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


def make_train_config(
    search_cfg: dict[str, Any],
    *,
    batch_size: int,
    learning_rate: float,
    trial_index: int,
) -> dict[str, Any]:
    from utils import load_config

    base_config_path = resolve_project_path(search_cfg.get("base_config", "configs/unet_no_shuffle_upper_40x20.yaml"))
    config = copy.deepcopy(load_config(base_config_path))
    config.setdefault("model", {})["base_channels"] = int(search_cfg.get("base_channels", 128))
    config["model"]["depth"] = int(search_cfg.get("depth", 3))
    config["model"]["kernel_size"] = int(search_cfg.get("kernel_size", 3))
    config["model"]["norm"] = str(search_cfg.get("norm", "gn"))
    config.setdefault("data", {})["batch_size"] = int(batch_size)
    config.setdefault("optimizer", {})["lr"] = float(learning_rate)
    config["loss"] = copy.deepcopy(search_cfg.get("loss", config.get("loss", {"name": "mse"})))
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


def build_variable_depth_unet(model_cfg: dict[str, Any]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class HybridPad2d(nn.Module):
        def __init__(self, pad_x: int, pad_y: int) -> None:
            super().__init__()
            self.pad_x = (pad_x, pad_x)
            self.pad_y = (pad_y, pad_y)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, (self.pad_x[0], self.pad_x[1], 0, 0), mode="circular")
            y_mode = "reflect" if x.shape[-2] > max(self.pad_y) else "replicate"
            return F.pad(x, (0, 0, self.pad_y[0], self.pad_y[1]), mode=y_mode)

    def build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "silu":
            return nn.SiLU(inplace=False)
        if name == "relu":
            return nn.ReLU(inplace=False)
        if name in {"leaky_relu", "leakyrelu"}:
            return nn.LeakyReLU(negative_slope=0.2, inplace=False)
        if name in {"none", "linear"}:
            return nn.Identity()
        raise ValueError(f"Unknown activation: {name}")

    def build_norm(kind: str, channels: int) -> nn.Module:
        if kind == "none":
            return nn.Identity()
        if kind == "bn":
            return nn.BatchNorm2d(channels, eps=1.0e-3, momentum=0.1, affine=True)
        if kind == "in":
            return nn.InstanceNorm2d(channels, eps=1.0e-5, affine=True, track_running_stats=False)
        if kind == "gn":
            groups = min(16, channels)
            while channels % groups != 0:
                groups -= 1
            return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1.0e-5)
        raise ValueError(f"Unknown norm kind: {kind}")

    class ConvBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int,
            activation: str,
            norm: str,
            dropout: float,
        ) -> None:
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                HybridPad2d(pad_x=pad, pad_y=pad),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=(norm == "none")),
                build_norm(norm, out_channels),
                build_activation(activation),
                nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
            super().__init__()
            self.net = nn.Sequential(
                ConvBlock(in_channels, out_channels, **kwargs),
                ConvBlock(out_channels, out_channels, **kwargs),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class Down(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, **kwargs))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class UpPixelShuffle(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int, **kwargs) -> None:
            super().__init__()
            scale = 2
            self.up = nn.Sequential(
                HybridPad2d(pad_x=1, pad_y=1),
                nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=3, padding=0),
                nn.PixelShuffle(scale),
            )
            self.conv = DoubleConv(out_channels + skip_channels, out_channels, **kwargs)

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(torch.cat([skip, x], dim=1))

    class DepthTuningUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_channels = int(model_cfg.get("in_channels", 8))
            out_channels = int(model_cfg.get("out_channels", 1))
            base_channels = int(model_cfg.get("base_channels", 64))
            depth = int(model_cfg.get("depth", 4))
            if depth < 2:
                raise ValueError("depth must be at least 2")

            block_kwargs = {
                "kernel_size": int(model_cfg.get("kernel_size", 3)),
                "activation": str(model_cfg.get("activation", "silu")),
                "norm": str(model_cfg.get("norm", "bn")),
                "dropout": float(model_cfg.get("dropout", 0.0)),
            }
            channels = [base_channels * (2**level) for level in range(depth)]
            self.inc = DoubleConv(in_channels, channels[0], **block_kwargs)
            self.downs = nn.ModuleList(
                Down(channels[level - 1], channels[level], **block_kwargs) for level in range(1, depth)
            )
            self.ups = nn.ModuleList(
                UpPixelShuffle(channels[level], channels[level - 1], channels[level - 1], **block_kwargs)
                for level in range(depth - 1, 0, -1)
            )
            self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            skips = []
            x = self.inc(x)
            skips.append(x)
            for down in self.downs:
                x = down(x)
                skips.append(x)
            for up, skip in zip(self.ups, reversed(skips[:-1])):
                x = up(x, skip)
            return self.outc(x)

    return DepthTuningUNet()


def metric_improved(value: float, best_value: float, mode: str) -> bool:
    if not math.isfinite(value):
        return False
    if mode == "max":
        return value > best_value
    if mode == "min":
        return value < best_value
    raise ValueError(f"Unknown early_stopping_mode: {mode}")


def train_one_batch_size(
    config: dict[str, Any],
    *,
    search_cfg: dict[str, Any],
    trial_seed: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from torch.amp import GradScaler, autocast

    from data_utils import build_dataloaders
    from engine.trainer import build_loss, build_optimizer, build_scheduler, evaluate
    from utils import get_device, seed_everything

    seed_everything(trial_seed)
    device = get_device(config.get("device", "auto"))
    train_loader, val_loader, test_loader, _ = build_dataloaders(config, PROJECT_DIR, trial_seed)

    model = build_variable_depth_unet(config["model"]).to(device)
    loss_fn = build_loss(config.get("loss", {"name": "mse"}))
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

        train_metrics = evaluate(model, train_loader, loss_fn, device, amp=amp)
        train_r2 = train_metrics["r2"]
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
            f"trial={config['trial_index']:03d} batch_size={batch_size} epoch={epoch:04d} "
            f"train_r2={train_r2:.6g} val_loss={val_loss:.6g} val_r2={val_r2:.6g}",
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
        "batch_size",
        "norm",
        "kernel_size",
        "depth",
        "base_channels",
        "learning_rate",
        "loss_name",
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
    summary_csv = resolve_project_path(search_cfg.get("summary_csv", "hyperparameter_tunning/batch_size/summary.csv"))
    base_channels = int(search_cfg.get("base_channels", 128))
    depth = int(search_cfg.get("depth", 3))
    kernel_size = int(search_cfg.get("kernel_size", 3))
    norm = str(search_cfg.get("norm", "gn"))
    base_seed = int(search_cfg.get("seed", 42))
    batch_sizes = [int(value) for value in search_cfg.get("batch_sizes", [4, 8, 16, 32, 64, 128])]
    loss_cfg = search_cfg.get("loss", {"name": "mse"})
    loss_name = str(loss_cfg.get("name", "mse"))

    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    for trial, batch_size in enumerate(batch_sizes, start=1):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if args.dry_run:
            print(
                f"trial={trial:03d} batch_size={batch_size} norm={norm} kernel_size={kernel_size} "
                f"depth={depth} base_channels={base_channels} lr={learning_rate:.8g} loss={loss_name}"
            )
            continue

        config = make_train_config(search_cfg, batch_size=batch_size, learning_rate=learning_rate, trial_index=trial)
        metrics = train_one_batch_size(
            config,
            search_cfg=search_cfg,
            trial_seed=base_seed + trial,
            batch_size=batch_size,
        )
        row = {
            "trial": trial,
            "batch_size": batch_size,
            "norm": norm,
            "kernel_size": kernel_size,
            "depth": depth,
            "base_channels": base_channels,
            "learning_rate": learning_rate,
            "loss_name": loss_name,
            **metrics,
        }
        append_summary_row(summary_csv, row)
        print(
            f"finished trial={trial:03d} batch_size={batch_size} status={metrics['status']} "
            f"train_r2={metrics['train_r2']} val_r2={metrics['val_r2']} test_r2={metrics['test_r2']}",
            flush=True,
        )

    if not args.dry_run:
        print(f"Wrote summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
