# general
import os
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# data
import json
import yaml
import argparse
from typing import Dict, Any

# user defined
from src.utils.machine_learning.data_loader import (
    prepare_unet_dataloaders,
    prepare_unet_lstm_many_to_one_dataloaders,
)
from src.models.UNet_ConvLSTM_3En3De_pixel import UNetPhysDepth1
from src.utils.machine_learning._opt import _build_optimizer, _build_lr_scheduler
from src.utils.machine_learning.cnn_device import _seed_everything, _set_num_threads
from src.utils.machine_learning.evaluation_dist import eval_r2_suite
from src.utils.machine_learning.save_pred import save_predictions_npy


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/dict. Got: {type(cfg)}")
    return cfg


def _get(cfg: Dict[str, Any], key: str, default=None):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    return dist.get_rank() if _is_distributed() else 0


def _get_world_size():
    return dist.get_world_size() if _is_distributed() else 1


def _is_main_process():
    return _get_rank() == 0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def atomic_torch_save(obj, path: Path):
    path = Path(path)
    tmp_path = path.with_name(path.name + f".tmp.rank{_get_rank()}")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def safe_torch_load(path: Path, device):
    path = Path(path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n"
            f"This checkpoint may be corrupted, possibly because the job was killed while saving.\n"
            f"Original error: {e}"
        )


def _setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{local_rank}"))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, world_size, local_rank, device


def _cleanup_distributed():
    if _is_distributed():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        dist.destroy_process_group()


def _rebuild_loader_with_sampler(loader, batch_size, num_workers, sampler, drop_last=False):
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=loader.pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


def _make_distributed_loader(loader, batch_size, num_workers, shuffle=False, drop_last=False):
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=_get_world_size(),
        rank=_get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )
    new_loader = _rebuild_loader_with_sampler(
        loader=loader,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=drop_last,
    )
    return new_loader, sampler


@torch.no_grad()
def eval_pixel_mse(model, loader, device):
    model.eval()

    sse = torch.tensor(0.0, device=device, dtype=torch.float64)
    n = torch.tensor(0, device=device, dtype=torch.long)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if yb.ndim == 3:
            yb = yb.unsqueeze(1)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        err2 = (pred - yb).double() ** 2
        sse += err2.sum()
        n += err2.numel()

    if _is_distributed():
        dist.all_reduce(sse, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    return float((sse / torch.clamp(n, min=1)).item())
# MAE
@torch.no_grad()
def eval_pixel_mae(model, loader, device):
    model.eval()

    sae = torch.tensor(0.0, device=device, dtype=torch.float64)
    n = torch.tensor(0, device=device, dtype=torch.long)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if yb.ndim == 3:
            yb = yb.unsqueeze(1)

        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        abs_err = torch.abs(pred - yb).double()
        sae += abs_err.sum()
        n += abs_err.numel()

    if _is_distributed():
        dist.all_reduce(sae, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    return float((sae / torch.clamp(n, min=1)).item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    distributed, rank, world_size, local_rank, device = _setup_distributed()

    out_dir = Path(_get(cfg, "output.dir", "outputs"))
    ckpt_dir = Path(_get(cfg, "ckpt.dir", "checkpoints"))

    if _is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier(device_ids=[local_rank])

    seed = int(_get(cfg, "seed", 123))
    _set_num_threads()
    _seed_everything(seed + rank, deterministic=bool(_get(cfg, "deterministic", False)))

    if _is_main_process():
        print("device:", device)
        print(f"distributed={distributed} rank={rank} local_rank={local_rank} world_size={world_size}")

    data_dir = Path(_get(cfg, "data.data_dir")).expanduser()
    x_path = data_dir / _get(cfg, "data.x_name", "X.npy")
    y_path = data_dir / _get(cfg, "data.y_name", "y.npy")

    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.float32, copy=False)

    if _is_main_process():
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    mode = str(_get(cfg, "task.mode", "unet")).lower()
    split_mode = str(_get(cfg, "split.mode", "time_ordered")).lower()

    batch_size = int(_get(cfg, "loader.batch_size", 8))
    num_workers = int(_get(cfg, "loader.num_workers", 0))
    shuffle_train = bool(_get(cfg, "loader.shuffle_train", True))

    frac_train = float(_get(cfg, "split.frac_train", 0.8))
    frac_skip = float(_get(cfg, "split.frac_skip", 0.0))
    frac_val = float(_get(cfg, "split.frac_val", 0.1))
    frac_test = float(_get(cfg, "split.frac_test", 0.1))

    normalize_x = bool(_get(cfg, "norm.normalize_x", True))
    normalize_y = bool(_get(cfg, "norm.normalize_y", True))

    if mode == "unet":
        train_loader_full, val_loader_full, test_loader_full, meta = prepare_unet_dataloaders(
            X, y,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            split_mode=split_mode,
            frac_train=frac_train,
            frac_skip=frac_skip,
            frac_val=frac_val,
            frac_test=frac_test,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            shuffle_train=shuffle_train,
            device=device,
        )
        use_lstm = False

    elif mode == "lstm":
        T = int(_get(cfg, "window.T", 7))
        stride = int(_get(cfg, "window.stride", 1))
        target = str(_get(cfg, "window.target", "last")).lower()

        train_loader_full, val_loader_full, test_loader_full, meta = prepare_unet_lstm_many_to_one_dataloaders(
            X, y,
            T=T,
            stride=stride,
            target=target,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            split_mode=split_mode,
            frac_train=frac_train,
            frac_skip=frac_skip,
            frac_val=frac_val,
            frac_test=frac_test,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            shuffle_train=shuffle_train,
            device=device,
        )
        use_lstm = True

    else:
        raise ValueError(f"Unknown task.mode={mode}. Use 'unet' or 'lstm'.")

    norm = meta["norm"]

    if distributed:
        train_loader, train_sampler = _make_distributed_loader(
            train_loader_full, batch_size, num_workers, shuffle=shuffle_train, drop_last=False
        )
        val_loader, val_sampler = _make_distributed_loader(
            val_loader_full, batch_size, num_workers, shuffle=False, drop_last=False
        )
        test_loader, test_sampler = _make_distributed_loader(
            test_loader_full, batch_size, num_workers, shuffle=False, drop_last=False
        )
    else:
        train_loader, val_loader, test_loader = train_loader_full, val_loader_full, test_loader_full
        train_sampler, val_sampler, test_sampler = None, None, None

    if _is_main_process():
        print(
            f"Mode={mode} | "
            f"train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}"
        )

    in_channels = _get(cfg, "model.input_channels", None)
    out_channels = _get(cfg, "model.head_channel", None)

    base = int(_get(cfg, "model.base", 32))
    dropout_2D = float(_get(cfg, "model.dropout_2D", 0.0))
    lstm_dropout = float(_get(cfg, "model.lstm_dropout", 0.0))
    kernel_size = int(_get(cfg, "model.kernel_size", 3))
    activation = str(_get(cfg, "model.activation", "SiLU"))
    batch_norm = str(_get(cfg, "model.batch_norm", "bn"))

    model = UNetPhysDepth1(
        input_channels=in_channels,
        head_channel=out_channels,
        base=base,
        kernel_size=kernel_size,
        activation=activation,
        batch_norm=batch_norm,
        dropout_2D=dropout_2D,
        use_lstm=use_lstm,
        lstm_dropout=lstm_dropout,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if _is_main_process():
        print(_unwrap_model(model))
        n_params = sum(p.numel() for p in _unwrap_model(model).parameters())
        n_trainable = sum(p.numel() for p in _unwrap_model(model).parameters() if p.requires_grad)
        print(f"Model params: total={n_params:,} trainable={n_trainable:,}")

    optim_name = str(_get(cfg, "optim.name", "AdamW"))
    lr = float(_get(cfg, "optim.lr", 3e-4))
    weight_decay = float(_get(cfg, "optim.weight_decay", 0.0))
    grad_clip = _get(cfg, "optim.grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    optimizer = _build_optimizer(
        _unwrap_model(model),
        optim_name=optim_name,
        lr=lr,
        weight_decay=weight_decay,
    )

    use_lr_schedule = bool(_get(cfg, "scheduler.use", True))
    scheduler = _build_lr_scheduler(
        optimizer,
        use_scheduler=use_lr_schedule,
        monitor="val_mse",
        factor=float(_get(cfg, "scheduler.factor", 0.5)),
        patience=int(_get(cfg, "scheduler.patience", 5)),
        min_lr=float(_get(cfg, "scheduler.min_lr", 1e-6)),
    )

    loss_name = str(_get(cfg, "loss.name", "huber")).lower()
    if loss_name == "huber":
        beta = float(_get(cfg, "loss.huber_beta", 0.5))
        loss_fn = nn.SmoothL1Loss(beta=beta, reduction="mean")
        if _is_main_process():
            print(f"Using Huber (SmoothL1) loss with beta={beta}")
    elif loss_name == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
        if _is_main_process():
            print("Using MSE loss")
    elif loss_name in ["mae", "l1", "l1loss"]:
        loss_fn = nn.L1Loss(reduction="mean")
        if _is_main_process():
            print("Using MAE/L1 Loss")
    else:
        raise ValueError("loss.name must be 'huber' or 'mse', or 'mae'")

    epochs = int(_get(cfg, "train.epochs", 50))
    patience = int(_get(cfg, "early_stop.patience", 10))
    min_delta = float(_get(cfg, "early_stop.min_delta", 1e-6))

    save_every = int(_get(cfg, "ckpt.save_every", 1))
    resume = bool(_get(cfg, "ckpt.resume", False))
    latest_path = ckpt_dir / str(_get(cfg, "ckpt.latest_name", "latest.pt"))
    best_path = ckpt_dir / str(_get(cfg, "ckpt.best_name", "best.pt"))

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1

    if resume and latest_path.exists():
        ckpt = safe_torch_load(latest_path, device)
        _unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        norm = ckpt.get("norm", norm)
        if _is_main_process():
            print(f"Resumed from {latest_path} at epoch {start_epoch - 1}")

    if distributed:
        dist.barrier(device_ids=[local_rank])

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()

        if distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

        run_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_seen = torch.tensor(0, device=device, dtype=torch.long)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)

            if pred.ndim == 4 and yb.ndim == 3:
                yb = yb.unsqueeze(1)

            loss = loss_fn(pred, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(_unwrap_model(model).parameters(), grad_clip)

            optimizer.step()

            bs = xb.shape[0]
            run_loss += loss.detach().double() * bs
            n_seen += bs

        if distributed:
            dist.all_reduce(run_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_seen, op=dist.ReduceOp.SUM)

        train_loss = float((run_loss / torch.clamp(n_seen, min=1)).item())

        train_r2 = eval_r2_suite(_unwrap_model(model), train_loader, device, norm=norm, denorm_y=False)
        val_r2 = eval_r2_suite(_unwrap_model(model), val_loader, device, norm=norm, denorm_y=False)
        tr_mse = eval_pixel_mse(_unwrap_model(model), train_loader, device)
        val_mse = eval_pixel_mse(_unwrap_model(model), val_loader, device)
        
        tr_mae = eval_pixel_mae(_unwrap_model(model), train_loader, device)
        val_mae = eval_pixel_mae(_unwrap_model(model), val_loader, device)

        if scheduler is not None:
            scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]

        if _is_main_process():
            dt = time.time() - t0
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"loss={train_loss:.6f} | "
                f"TRAIN MSE={tr_mse:.6f} "
                f"TRAIN MAE={tr_mae:.6f} "
                f"pixR2={train_r2['pixel_r2']:+.4f} "
                f"imgR2={train_r2['image_r2_mean']:+.4f} "
                f"climR2={train_r2['climate_r2']:+.4f} | "
                f"VAL MSE={val_mse:.6f} "
                f"VAL MAE={val_mae:.6f} "
                f"pixR2={val_r2['pixel_r2']:+.4f} "
                f"imgR2={val_r2['image_r2_mean']:+.4f} "
                f"climR2={val_r2['climate_r2']:+.4f} | "
                f"lr={current_lr:.2e} | {dt:.1f}s"
            )

            if (epoch % save_every) == 0:
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model": _unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "norm": norm,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "bad_epochs": bad_epochs,
                        "cfg_path": str(cfg_path),
                        "rng_state": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        },
                    },
                    latest_path,
                )

            improved = (best_val - val_mae) > min_delta
            if improved:
                best_val = val_mae
                best_epoch = epoch
                bad_epochs = 0
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model": _unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "norm": norm,
                        "best_val": best_val,
                        "cfg_path": str(cfg_path),
                    },
                    best_path,
                )
                print("  ✓ saved BEST")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(
                        f"Early stopping: no val MSE improvement for {patience} epochs. "
                        f"Best epoch={best_epoch}, best_val={best_val:.6f}"
                    )

        stop_flag = torch.tensor(0, device=device, dtype=torch.int64)
        best_val_t = torch.tensor(best_val if _is_main_process() else 0.0, device=device, dtype=torch.float64)
        best_epoch_t = torch.tensor(best_epoch if _is_main_process() else 0, device=device, dtype=torch.int64)
        bad_epochs_t = torch.tensor(bad_epochs if _is_main_process() else 0, device=device, dtype=torch.int64)

        if _is_main_process() and bad_epochs >= patience:
            stop_flag.fill_(1)

        if distributed:
            dist.broadcast(stop_flag, src=0)
            dist.broadcast(best_val_t, src=0)
            dist.broadcast(best_epoch_t, src=0)
            dist.broadcast(bad_epochs_t, src=0)

        best_val = float(best_val_t.item())
        best_epoch = int(best_epoch_t.item())
        bad_epochs = int(bad_epochs_t.item())

        if stop_flag.item() == 1:
            break

        if distributed:
            dist.barrier(device_ids=[local_rank])

    if best_path.exists():
        ckpt = safe_torch_load(best_path, device)
        _unwrap_model(model).load_state_dict(ckpt["model"])
        norm_best = ckpt.get("norm", norm)
    else:
        if _is_main_process():
            print(f"WARNING: best checkpoint not found at {best_path}. Using current model.")
        norm_best = norm

    if distributed:
        dist.barrier(device_ids=[local_rank])

    test_stats = eval_r2_suite(_unwrap_model(model), test_loader, device, norm=norm_best, denorm_y=True)

    if _is_main_process():
        with open(out_dir / "test_stats.json", "w") as f:
            json.dump(test_stats, f, indent=2)

        save_predictions_npy(
            model=_unwrap_model(model),
            loader=test_loader_full,
            device=device,
            out_dir=out_dir,
            prefix="test",
            norm=norm_best,
            denorm_y=True,
        )

        print("==== TEST (best) ====")
        print(
            f"TEST pixR2={test_stats['pixel_r2']:+.4f} "
            f"imgR2_mean={test_stats['image_r2_mean']:+.4f} "
            f"imgR2_median={test_stats['image_r2_median']:+.4f} "
            f"climR2={test_stats['climate_r2']:+.4f} "
            f"(n_images={test_stats['n_images']}, n_pixels={test_stats['n_pixels']})"
        )

    if distributed:
        dist.barrier(device_ids=[local_rank])

    _cleanup_distributed()


if __name__ == "__main__":
    main()