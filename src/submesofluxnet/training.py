"""Optimization and early-stopping loop for SubmesoFluxNet."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from .config import project_dir, resolve
from .data import make_loaders
from .evaluation import evaluate_metrics
from .losses import build_loss, latent_regularization
from .model import build_model
from .runtime import get_device, seed_everything


def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    *,
    smooth_weight: float,
    corr_weight: float,
    clip_norm: float = 0.0,
):
    """Optimize one epoch and return sample-weighted training metrics."""
    model.train()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        features = model.encode_features(x)
        prediction = model.decode_features(features)
        prediction_loss = loss_fn(prediction, y)
        regularization_loss = latent_regularization(
            features,
            smooth_weight=smooth_weight,
            corr_weight=corr_weight,
        )
        loss = prediction_loss + regularization_loss
        loss.backward()
        if clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        batch = x.shape[0]
        totals[0] += loss.detach().double() * batch
        totals[1] += prediction_loss.detach().double() * batch
        totals[2] += torch.mean((prediction - y) ** 2).detach().double() * batch
        totals[3] += torch.mean(torch.abs(prediction - y)).detach().double() * batch
        totals[4] += batch
    denominator = totals[4].clamp_min(1.0)
    return {
        "loss": float((totals[0] / denominator).item()),
        "pred_loss": float((totals[1] / denominator).item()),
        "mse": float((totals[2] / denominator).item()),
        "mae": float((totals[3] / denominator).item()),
    }


def append_csv(path: Path, row: dict) -> None:
    """Append one epoch to the training-history table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_scheduler(train_cfg: dict, optimizer):
    """Build the configured learning-rate scheduler."""
    name = str(train_cfg.get("lr_scheduler", "none")).lower()
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("scheduler_patience", 10)),
            min_lr=float(train_cfg.get("scheduler_min_lr", 1.0e-5)),
        )
    raise ValueError(f"Unknown scheduler: {name}")


def train(config: dict) -> None:
    """Train the configured model and retain the lowest-validation-loss checkpoint."""
    root = project_dir()
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    train_loader, val_loader, _, _ = make_loaders(config, recompute_stats=True)
    model = build_model(config).to(device)
    train_cfg = config["train"]
    loss_fn = build_loss(str(train_cfg.get("loss", "huber")))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(train_cfg, optimizer)
    checkpoint_path = resolve(root, train_cfg["checkpoint"])
    history_path = resolve(root, train_cfg["history_csv"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale_epochs = 0
    patience = int(train_cfg.get("patience", 20))
    smooth_weight = float(train_cfg.get("latent_smooth_weight", 0.0))
    corr_weight = float(train_cfg.get("latent_corr_weight", 0.0))

    for epoch in range(1, int(train_cfg.get("epochs", 250)) + 1):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            smooth_weight=smooth_weight,
            corr_weight=corr_weight,
            clip_norm=float(train_cfg.get("gradient_clip_norm", 0.0)),
        )
        train_eval = evaluate_metrics(model, train_loader, loss_fn, device)
        val_stats = evaluate_metrics(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_stats["loss"],
            "train_pred_loss": train_stats["pred_loss"],
            "train_mse": train_eval["mse"],
            "train_mae": train_eval["mae"],
            "train_r2": train_eval["r2"],
            "val_loss": val_stats["loss"],
            "val_mse": val_stats["mse"],
            "val_mae": val_stats["mae"],
            "val_r2": val_stats["r2"],
        }
        append_csv(history_path, row)
        print(
            f"epoch={epoch} train_r2={train_eval['r2']:.4f} "
            f"val_r2={val_stats['r2']:.4f} "
            f"train_loss={train_stats['loss']:.6g} "
            f"val_loss={val_stats['loss']:.6g}",
            flush=True,
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch}; best_val={best_val:.6g}",
                    flush=True,
                )
                break
        if scheduler is not None:
            scheduler.step(val_stats["loss"])
