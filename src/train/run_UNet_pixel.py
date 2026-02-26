# general
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
# data
import json
import yaml
import argparse
from typing import Dict, Any
# user defined 
from src.utils.data_loader import (
    prepare_unet_dataloaders,
    prepare_unet_lstm_many_to_one_dataloaders,
)
from src.models.UNet_ConvLSTM_3En3De_pixel import UNetPhysDepth1
from src.utils._opt import _build_optimizer, _build_lr_scheduler
from src.utils.cnn_device import _seed_everything, _select_device, _set_num_threads
from src.utils.evaluation import eval_r2_suite
from src.utils.save_pred import save_predictions_npy


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/dict. Got: {type(cfg)}")
    return cfg

def _get(cfg: Dict[str, Any], key: str, default=None):
    """Small helper: dot-path getter. _get(cfg,'train.epochs')"""
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
    
@torch.no_grad()
def eval_pixel_mse(model, loader, device):
    model.eval()
    sse = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if yb.ndim == 3:
            yb = yb.unsqueeze(1)
        pred = model(xb)
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        err2 = (pred - yb) ** 2
        sse += float(err2.sum().item())
        n += err2.numel()
    return sse / max(1, n)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_yaml(cfg_path)

    out_dir = Path(_get(cfg, "output.dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- system / seed --------------------
    seed = int(_get(cfg, "seed", 123))
    _set_num_threads()
    _seed_everything(seed, deterministic=bool(_get(cfg, "deterministic", False)))
    device = _select_device()
    print("device:", device)

    # -------------------- data paths --------------------
    data_dir = Path(_get(cfg, "data.data_dir")).expanduser()
    x_path = data_dir / _get(cfg, "data.x_name", "X.npy")
    y_path = data_dir / _get(cfg, "data.y_name", "y.npy")

    X = np.load(x_path).astype(np.float32, copy=False)
    y = np.load(y_path).astype(np.float32, copy=False)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    # -------------------- dataloaders --------------------
    mode = str(_get(cfg, "task.mode", "unet")).lower()  # "unet" or "lstm"
    split_mode = str(_get(cfg, "split.mode", "time_ordered")).lower()

    batch_size = int(_get(cfg, "loader.batch_size", 8))
    num_workers = int(_get(cfg, "loader.num_workers", 0))
    shuffle_train = bool(_get(cfg, "loader.shuffle_train", True))

    frac_train = float(_get(cfg, "split.frac_train", 0.8))
    frac_skip  = float(_get(cfg, "split.frac_skip", 0.0))
    frac_val   = float(_get(cfg, "split.frac_val", 0.1))
    frac_test  = float(_get(cfg, "split.frac_test", 0.1))

    normalize_x = bool(_get(cfg, "norm.normalize_x", True))
    normalize_y = bool(_get(cfg, "norm.normalize_y", True))

    if mode == "unet":
        train_loader, val_loader, test_loader, meta = prepare_unet_dataloaders(
            X, y,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            split_mode=split_mode,            # "time_ordered" or "random"
            frac_train=frac_train,
            frac_skip=frac_skip,
            frac_val=frac_val,
            frac_test=frac_test,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            shuffle_train=shuffle_train,
            device=device,                    # for pin_memory logic
        )
        use_lstm = False

    elif mode == "lstm":
        T = int(_get(cfg, "window.T", 7))
        stride = int(_get(cfg, "window.stride", 1))
        target = str(_get(cfg, "window.target", "last")).lower()  # "last" or "next"

        train_loader, val_loader, test_loader, meta = prepare_unet_lstm_many_to_one_dataloaders(
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

    norm = meta["norm"]  # contains X_mu/X_sd/y_mu/y_sd (train-only)

    print(f"Mode={mode} | train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}")

    # -------------------- model --------------------
    # infer channels if not provided
    in_channels = _get(cfg, "model.input_channels", None)
    out_channels = _get(cfg, "model.head_channel", None)

    base = int(_get(cfg, "model.base", 32))
    dropout_2D = float(_get(cfg, "model.dropout_2D", 0.0))
    lstm_dropout = float(_get(cfg, "model.lstm_dropout", 0.0))
    kernel_size = int(_get(cfg, "model.kernel_size", 3))
    activation = str(_get(cfg, "model.activation", "SiLU"))
    batch_norm = str(_get(cfg, "model.batch_norm", "bn"))
    # model settings
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

    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={n_params:,} trainable={n_trainable:,}")

    # -------------------- optimizer / scheduler --------------------
    optim_name = str(_get(cfg, "optim.name", "AdamW"))
    lr = float(_get(cfg, "optim.lr", 3e-4))
    weight_decay = float(_get(cfg, "optim.weight_decay", 0.0))
    grad_clip = _get(cfg, "optim.grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    optimizer = _build_optimizer(model, optim_name=optim_name, lr=lr, weight_decay=weight_decay)

    use_lr_schedule = bool(_get(cfg, "scheduler.use", True))
    scheduler = _build_lr_scheduler(
        optimizer,
        use_scheduler=use_lr_schedule,
        monitor="val_mse",
        factor=float(_get(cfg, "scheduler.factor", 0.5)),
        patience=int(_get(cfg, "scheduler.patience", 5)),
        min_lr=float(_get(cfg, "scheduler.min_lr", 1e-6)),
    )

    # -------------------- loss --------------------
    loss_name = str(_get(cfg, "loss.name", "huber")).lower()  # "huber" or "mse"
    if loss_name == "huber":
        beta = float(_get(cfg, "loss.huber_beta", 0.5))
        loss_fn = nn.SmoothL1Loss(beta=beta, reduction="mean")
        print(f"Using Huber (SmoothL1) loss with beta={beta}")
    elif loss_name == "mse":
        loss_fn = nn.MSELoss(reduction="mean")
        print("Using MSE loss")
    else:
        raise ValueError("loss.name must be 'huber' or 'mse'")

    # -------------------- training control --------------------
    epochs = int(_get(cfg, "train.epochs", 50))
    patience = int(_get(cfg, "early_stop.patience", 10))
    min_delta = float(_get(cfg, "early_stop.min_delta", 1e-6))

    # checkpoints
    ckpt_dir = Path(_get(cfg, "ckpt.dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(_get(cfg, "ckpt.save_every", 1))
    resume = bool(_get(cfg, "ckpt.resume", False))
    latest_path = ckpt_dir / str(_get(cfg, "ckpt.latest_name", "latest.pt"))
    best_path = ckpt_dir / str(_get(cfg, "ckpt.best_name", "best.pt"))
    

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    start_epoch = 1

    # resume
    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", 0))
        bad_epochs = int(ckpt.get("bad_epochs", 0))
        norm = ckpt.get("norm", norm)
        print(f"Resumed from {latest_path} at epoch {start_epoch - 1}")

    # -------------------- train loop --------------------
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()

        run_loss = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # UNet: (B,C,H,W) | LSTM: (B,T,C,H,W)
            yb = yb.to(device, non_blocking=True)  # (B,1,H,W) typically

            optimizer.zero_grad(set_to_none=True)
            # print("batch shape:", xb.shape)
            pred = model(xb)
            
            # make shapes consistent if y was (B,H,W)
            if pred.ndim == 4 and yb.ndim == 3:
                yb = yb.unsqueeze(1)

            loss = loss_fn(pred, yb)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = xb.shape[0]
            run_loss += float(loss.item()) * bs
            n_seen += bs

        train_loss = run_loss / max(1, n_seen)

        # metrics
        train_r2 = eval_r2_suite(model, train_loader, device, norm=norm, denorm_y=False)
        val_r2   = eval_r2_suite(model, val_loader,   device, norm=norm, denorm_y=False)
        tr_mse  = eval_pixel_mse(model, train_loader, device)
        val_mse = eval_pixel_mse(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]["lr"]

        dt = time.time() - t0
        print(
            f"Epoch {epoch:4d}/{epochs} | "
            f"loss={train_loss:.6f} | "
            f"TRAIN MSE={tr_mse:.6f} "
            f"pixR2={train_r2['pixel_r2']:+.4f} "
            f"imgR2={train_r2['image_r2_mean']:+.4f} "
            f"climR2={train_r2['climate_r2']:+.4f} | "
            f"VAL MSE={val_mse:.6f} "
            f"pixR2={val_r2['pixel_r2']:+.4f} "
            f"imgR2={val_r2['image_r2_mean']:+.4f} "
            f"climR2={val_r2['climate_r2']:+.4f} | "
            f"lr={current_lr:.2e} | {dt:.1f}s"
        )

        # save latest
        if (epoch % save_every) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
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

        # early stop + save best
        improved = (best_val - val_mse) > min_delta
        if improved:
            best_val = val_mse
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
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
                break

    # -------------------- final test on best --------------------
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        norm_best = ckpt.get("norm", norm)
    else:
        print(f"WARNING: best checkpoint not found at {best_path}. Using current model.")
        norm_best = norm

    test_stats = eval_r2_suite(model, test_loader, device, norm=norm_best, denorm_y=True)
    
    with open(out_dir / "test_stats.json", "w") as f:
        json.dump(test_stats, f, indent=2)
    
    save_predictions_npy(
        model=model,
        loader=test_loader,
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
    

if __name__ == "__main__":
    # source ~/.virtualenvs/JC_Ocean/bin/activate
    # python -m src.train.run_UNet_pixel --config /scratch/jczhang/submeso_data_driven/configs/test_model/cfg_7.yaml

    main()
