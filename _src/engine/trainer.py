from __future__ import annotations

import copy
import csv
from pathlib import Path

import torch
import yaml
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast


class CompositeRegressionLoss(nn.Module):
    def __init__(
        self,
        *,
        base_loss: nn.Module,
        domain_mean_weight: float = 0.0,
        high_y_weight: float = 0.0,
        high_y_eps: float = 1.0e-6,
        gradient_weight: float = 0.0,
        gradient_periodic_x: bool = True,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.domain_mean_weight = domain_mean_weight
        self.high_y_weight = high_y_weight
        self.high_y_eps = high_y_eps
        self.gradient_weight = gradient_weight
        self.gradient_periodic_x = gradient_periodic_x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(pred, target)

        if self.domain_mean_weight > 0.0:
            pred_mean = pred.mean(dim=(-2, -1))
            target_mean = target.mean(dim=(-2, -1))
            loss = loss + self.domain_mean_weight * torch.nn.functional.mse_loss(pred_mean, target_mean)

        if self.high_y_weight > 0.0:
            abs_target = target.detach().abs()
            scale = abs_target / abs_target.mean().clamp_min(self.high_y_eps)
            loss = loss + self.high_y_weight * torch.mean(scale * (pred - target) ** 2)

        if self.gradient_weight > 0.0:
            loss = loss + self.gradient_weight * self._gradient_loss(pred, target)

        return loss

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.gradient_periodic_x:
            pred_dx = torch.roll(pred, shifts=-1, dims=-1) - pred
            target_dx = torch.roll(target, shifts=-1, dims=-1) - target
        else:
            pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
            target_dx = target[..., :, 1:] - target[..., :, :-1]

        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        return torch.nn.functional.mse_loss(pred_dx, target_dx) + torch.nn.functional.mse_loss(pred_dy, target_dy)


def build_loss(config: dict) -> nn.Module:
    name = config.get("name", "mse").lower()
    if name == "mse":
        base_loss: nn.Module = nn.MSELoss()
    elif name in {"mae", "l1"}:
        base_loss = nn.L1Loss()
    elif name in {"huber", "smooth_l1"}:
        base_loss = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {name}")

    return CompositeRegressionLoss(
        base_loss=base_loss,
        domain_mean_weight=float(config.get("domain_mean_weight", 0.0)),
        high_y_weight=float(config.get("high_y_weight", 0.0)),
        high_y_eps=float(config.get("high_y_eps", 1.0e-6)),
        gradient_weight=float(config.get("gradient_weight", 0.0)),
        gradient_periodic_x=bool(config.get("gradient_periodic_x", True)),
    )


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    name = config.get("name", "adamw").lower()
    lr = float(config.get("lr", 3.0e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict, epochs: int):
    name = config.get("name", "none").lower()
    if name in {"none", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(config.get("min_lr", 1.0e-6)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    raise ValueError(f"Unknown scheduler: {name}")


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist() or dist.get_rank() == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_sums(values: torch.Tensor) -> torch.Tensor:
    if is_dist():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def batch_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    return mse.detach(), mae.detach()


def evaluate(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device, amp: bool = False) -> dict[str, float]:
    model.eval()
    # loss_sum, mse_sum, mae_sum, sample_count, sse, y_sum, y2_sum, element_count
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)
            batch_size = x.size(0)
            mse, mae = batch_metrics(pred, y)
            residual = (pred - y).detach().double()
            target = y.detach().double()
            totals[0] += loss.detach().double() * batch_size
            totals[1] += mse.double() * batch_size
            totals[2] += mae.double() * batch_size
            totals[3] += batch_size
            totals[4] += torch.sum(residual ** 2)
            totals[5] += torch.sum(target)
            totals[6] += torch.sum(target ** 2)
            totals[7] += target.numel()
    totals = reduce_sums(totals)
    total_samples = totals[3].clamp_min(1.0)
    total_elements = totals[7].clamp_min(1.0)
    sst = totals[6] - totals[5] ** 2 / total_elements
    r2 = 1.0 - totals[4] / sst if sst > 0 else torch.tensor(float("nan"), device=device)
    return {
        "loss": float((totals[0] / total_samples).item()),
        "mse": float((totals[1] / total_samples).item()),
        "mae": float((totals[2] / total_samples).item()),
        "r2": float(r2.item()),
    }


def append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    existing_rows: list[dict] = []
    if path.exists():
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            old_fields = reader.fieldnames or []
            for name in old_fields:
                if name not in fieldnames:
                    fieldnames.append(name)
            for name in fieldnames:
                if name not in old_fields:
                    old_fields.append(name)
            existing_rows = list(reader)
        if set(fieldnames) != set(reader.fieldnames or []):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({name: old_row.get(name, "") for name in fieldnames})
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not path.exists() or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_config_snapshot(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = copy.deepcopy(config)
    with path.open("w") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": unwrap_model(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "best_val": best_val,
            "config": config,
        },
        path,
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    config: dict,
    device: torch.device,
    *,
    resume_checkpoint: Path | None = None,
) -> None:
    train_cfg = config["train"]
    epochs = int(train_cfg.get("epochs", 100))
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    patience = int(train_cfg.get("early_stopping_patience", 15))

    loss_fn = build_loss(config.get("loss", {"name": "mse"}))
    optimizer = build_optimizer(model, config.get("optimizer", {}))
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}), epochs)
    scaler = GradScaler(enabled=amp)

    best_val = float("inf")
    stale_epochs = 0
    start_epoch = 1
    history_csv = Path(train_cfg["history_csv"])
    best_path = Path(train_cfg["best_checkpoint"])
    last_path = Path(train_cfg["last_checkpoint"])
    config_snapshot = Path(train_cfg.get("config_snapshot", history_csv.parent / "config.yaml"))

    if is_main_process():
        save_config_snapshot(config_snapshot, config)

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
        if is_main_process():
            print(
                f"Resuming from {resume_checkpoint} at epoch={start_epoch} "
                f"best={best_val:.6g}",
                flush=True,
            )

    if start_epoch > epochs:
        if is_main_process():
            print(f"Checkpoint already reached configured epochs={epochs}.", flush=True)
        return

    for epoch in range(start_epoch, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        train_totals = torch.zeros(2, dtype=torch.float64, device=device)
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

            batch_size = x.size(0)
            train_totals[0] += loss.detach().double() * batch_size
            train_totals[1] += batch_size

        train_totals = reduce_sums(train_totals)
        train_loss = float((train_totals[0] / train_totals[1].clamp_min(1.0)).item())
        train_metrics = evaluate(model, train_loader, loss_fn, device, amp=amp)
        val_metrics = evaluate(model, val_loader, loss_fn, device, amp=amp)
        lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
        }
        if is_main_process():
            append_history(history_csv, row)

        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            if is_main_process():
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val=best_val,
                    config=config,
                )
        else:
            stale_epochs += 1

        if is_main_process():
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val=best_val,
                config=config,
            )
            print(
                f"epoch={epoch:04d} train={train_loss:.6g} train_r2={train_metrics['r2']:.6g} "
                f"val={val_metrics['loss']:.6g} val_r2={val_metrics['r2']:.6g} "
                f"val_mae={val_metrics['mae']:.6g} lr={lr:.3e} best={best_val:.6g}",
                flush=True,
            )

        if stale_epochs >= patience:
            if is_main_process():
                print(f"Early stopping after {patience} stale epochs.", flush=True)
            break
