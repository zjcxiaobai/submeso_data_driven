"""Optimization and early-stopping loop for SubmesoFluxNet."""

import csv
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .data import make_loaders
from .evaluation import evaluate_metrics
from .losses import build_loss
from .model import build_model
from .runtime import get_device, seed_everything


def train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
) -> float:
    """Optimize the model for one epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        batch = x.shape[0]
        total_loss += loss.item() * batch
        total_samples += batch

    return total_loss / total_samples


def append_csv(path: Path, row: dict) -> None:
    """Append one epoch to the training-history table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(config: dict) -> None:
    """Train the configured model and retain the lowest-validation-loss checkpoint."""
    seed_everything(config["seed"])
    device = get_device(config["device"])
    train_loader, val_loader, _, _ = make_loaders(config, recompute_stats=True)
    model = build_model(config).to(device)
    train_cfg = config["train"]
    loss_fn = build_loss(train_cfg["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=train_cfg["scheduler_factor"],
        patience=train_cfg["scheduler_patience"],
        min_lr=train_cfg["scheduler_min_lr"],
    )
    checkpoint_path = PROJECT_ROOT / train_cfg["checkpoint"]
    history_path = PROJECT_ROOT / train_cfg["history_csv"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()

    best_val = float("inf")
    stale_epochs = 0
    patience = train_cfg["patience"]

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_eval = evaluate_metrics(model, train_loader, loss_fn, device)
        val_stats = evaluate_metrics(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
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
            f"train_loss={train_loss:.6g} "
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

        scheduler.step(val_stats["loss"])
