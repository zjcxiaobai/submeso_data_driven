#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class NormStats:
    x_mean: np.ndarray  # (C,1,1)
    x_std: np.ndarray   # (C,1,1)
    y_mean: np.ndarray  # (1,1,1)
    y_std: np.ndarray   # (1,1,1)


class TemporalSequenceDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        *,
        temporal_offsets: list[int],
        stats: NormStats,
        normalize_x: bool,
        normalize_y: bool,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"X/y sample count differs: {self.x.shape[0]} vs {self.y.shape[0]}")
        if self.x.ndim != 4 or self.y.ndim != 4:
            raise ValueError(f"Expected X/y as 4D arrays, got X={self.x.shape}, y={self.y.shape}")
        self.temporal_offsets = temporal_offsets
        self.stats = stats
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = np.stack(
            [np.asarray(self.x[index + offset], dtype=np.float32) for offset in self.temporal_offsets],
            axis=0,
        )  # (T,C,H,W)
        y = np.asarray(self.y[index], dtype=np.float32)
        if self.normalize_x:
            seq = (seq - self.stats.x_mean[None, ...]) / self.stats.x_std[None, ...]
        if self.normalize_y:
            y = (y - self.stats.y_mean) / self.stats.y_std
        return torch.from_numpy(seq.copy()), torch.from_numpy(y.copy())


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def split_raw_blocks(n_samples: int, split: dict) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    train_frac = float(split.get("train", 0.8))
    val_frac = float(split.get("val", 0.1))
    test_frac = float(split.get("test", 1.0 - train_frac - val_frac))
    if abs(train_frac + val_frac + test_frac - 1.0) > 1.0e-6:
        raise ValueError("Split fractions must sum to 1.0")
    if not 0.0 < train_frac < 1.0 or not 0.0 <= val_frac < 1.0 or not 0.0 < test_frac < 1.0:
        raise ValueError("Invalid split fractions")
    n_train = int(round(train_frac * n_samples))
    n_val = int(round(val_frac * n_samples))
    if n_train <= 0 or n_train + n_val >= n_samples:
        raise ValueError("Split fractions leave an empty train or test block")
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n_samples)


def centers_in_block(block: tuple[int, int], offsets: list[int]) -> np.ndarray:
    block_start, block_stop = block
    start = max(block_start, block_start - min(offsets))
    stop = min(block_stop, block_stop - max(offsets))
    if stop <= start:
        raise ValueError(f"No valid centers for block={block}, offsets={offsets}")
    return np.arange(start, stop, dtype=np.int64)


def compute_stats(x_path: Path, y_path: Path, train_idx: np.ndarray, offsets: list[int], eps: float = 1.0e-6) -> NormStats:
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    x_sum = np.zeros((x.shape[1],), dtype=np.float64)
    x_sum2 = np.zeros((x.shape[1],), dtype=np.float64)
    y_sum = np.zeros((y.shape[1],), dtype=np.float64)
    y_sum2 = np.zeros((y.shape[1],), dtype=np.float64)
    x_count = 0
    y_count = 0
    for idx in train_idx:
        for offset in offsets:
            sample_x = np.asarray(x[int(idx) + offset], dtype=np.float64)
            x_sum += sample_x.sum(axis=(1, 2))
            x_sum2 += np.square(sample_x).sum(axis=(1, 2))
            x_count += sample_x.shape[1] * sample_x.shape[2]
        sample_y = np.asarray(y[int(idx)], dtype=np.float64)
        y_sum += sample_y.sum(axis=(1, 2))
        y_sum2 += np.square(sample_y).sum(axis=(1, 2))
        y_count += sample_y.shape[1] * sample_y.shape[2]
    x_mean = x_sum / x_count
    x_std = np.sqrt(np.maximum(x_sum2 / x_count - x_mean**2, eps**2))
    y_mean = y_sum / y_count
    y_std = np.sqrt(np.maximum(y_sum2 / y_count - y_mean**2, eps**2))
    return NormStats(
        x_mean=x_mean.astype(np.float32)[:, None, None],
        x_std=x_std.astype(np.float32)[:, None, None],
        y_mean=y_mean.astype(np.float32)[:, None, None],
        y_std=y_std.astype(np.float32)[:, None, None],
    )


def save_stats(path: Path, stats: NormStats, offsets: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x_mean=stats.x_mean,
        x_std=stats.x_std,
        y_mean=stats.y_mean,
        y_std=stats.y_std,
        temporal_offsets=np.asarray(offsets, dtype=np.int64),
    )


def load_stats(path: Path) -> NormStats:
    data = np.load(path)
    return NormStats(x_mean=data["x_mean"], x_std=data["x_std"], y_mean=data["y_mean"], y_std=data["y_std"])


def make_loaders(config: dict, *, recompute_stats: bool = False):
    root = project_dir()
    data_cfg = config["data"]
    x_path = resolve(root, data_cfg["x_path"])
    y_path = resolve(root, data_cfg["y_path"])
    offsets = [int(v) for v in data_cfg.get("temporal_offsets", [0])]
    n_samples = np.load(x_path, mmap_mode="r").shape[0]
    blocks = split_raw_blocks(n_samples, data_cfg["split"])
    train_idx, val_idx, test_idx = [centers_in_block(block, offsets) for block in blocks]
    stats_path = resolve(root, config["output"].get("stats", "analysis/temporal_bottleneck/outputs/norm_stats.npz"))
    if recompute_stats or not stats_path.exists():
        stats = compute_stats(x_path, y_path, train_idx, offsets)
        save_stats(stats_path, stats, offsets)
    else:
        stats = load_stats(stats_path)
    dataset = TemporalSequenceDataset(
        x_path,
        y_path,
        temporal_offsets=offsets,
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=bool(data_cfg.get("normalize_y", True)),
    )
    loader_kwargs = dict(
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, stats


class TemporalBottleneck(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        latent_channels: int,
        spatial_channels: list[int],
        temporal_channels: int,
        temporal_layers: int,
        decoder_channels: list[int],
        dropout: float,
        temporal_reduce: str = "final",
    ) -> None:
        super().__init__()
        spatial = []
        channels = [input_channels] + spatial_channels
        for c0, c1 in zip(channels[:-1], channels[1:]):
            spatial.extend([
                nn.Conv2d(c0, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.SiLU(),
                nn.Dropout2d(dropout),
            ])
        self.spatial_encoder = nn.Sequential(*spatial)

        temporal = []
        in_ch = spatial_channels[-1]
        for _ in range(temporal_layers):
            temporal.extend([
                nn.Conv3d(in_ch, temporal_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(temporal_channels),
                nn.SiLU(),
                nn.Dropout3d(dropout),
            ])
            in_ch = temporal_channels
        self.temporal_encoder = nn.Sequential(*temporal)
        self.to_latent = nn.Conv2d(temporal_channels, latent_channels, kernel_size=1)

        decoder = []
        channels = [latent_channels] + decoder_channels
        for c0, c1 in zip(channels[:-1], channels[1:]):
            decoder.extend([
                nn.Conv2d(c0, c1, kernel_size=3, padding=1),
                nn.BatchNorm2d(c1),
                nn.SiLU(),
                nn.Dropout2d(dropout),
            ])
        decoder.append(nn.Conv2d(channels[-1], 1, kernel_size=1))
        self.decoder = nn.Sequential(*decoder)
        if temporal_reduce not in {"final", "mean"}:
            raise ValueError("temporal_reduce must be 'final' or 'mean'")
        self.temporal_reduce = temporal_reduce

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C,H,W)
        b, t, c, h, w = x.shape
        spatial = self.spatial_encoder(x.reshape(b * t, c, h, w))
        _, f, hh, ww = spatial.shape
        spatial = spatial.reshape(b, t, f, hh, ww).permute(0, 2, 1, 3, 4)  # (B,F,T,H,W)
        temporal = self.temporal_encoder(spatial)
        if self.temporal_reduce == "final":
            summary = temporal[:, :, -1]
        else:
            summary = temporal.mean(dim=2)
        return self.to_latent(summary)

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_features(self.encode_features(x))


def build_model(config: dict) -> TemporalBottleneck:
    model_cfg = config["model"]
    return TemporalBottleneck(
        input_channels=int(model_cfg["input_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        spatial_channels=[int(v) for v in model_cfg["spatial_channels"]],
        temporal_channels=int(model_cfg.get("temporal_channels", model_cfg["spatial_channels"][-1])),
        temporal_layers=int(model_cfg.get("temporal_layers", 2)),
        decoder_channels=[int(v) for v in model_cfg.get("decoder_channels", [32, 16])],
        dropout=float(model_cfg.get("dropout", 0.0)),
        temporal_reduce=str(model_cfg.get("temporal_reduce", "final")),
    )


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name in {"mae", "l1"}:
        return nn.L1Loss()
    if name in {"huber", "smooth_l1"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {name}")


def latent_smoothness(features: torch.Tensor) -> torch.Tensor:
    dx = features[..., :, 1:] - features[..., :, :-1]
    dy = features[..., 1:, :] - features[..., :-1, :]
    return torch.mean(dx**2) + torch.mean(dy**2)


def latent_decorrelation(features: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    channels = features.shape[1]
    if channels <= 1:
        return features.new_tensor(0.0)
    flat = features.permute(1, 0, 2, 3).reshape(channels, -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    flat = flat / flat.std(dim=1, keepdim=True).clamp_min(eps)
    corr = flat @ flat.t() / flat.shape[1]
    eye = torch.eye(channels, dtype=corr.dtype, device=corr.device)
    return torch.mean((corr - eye) ** 2)


def latent_regularization(features: torch.Tensor, *, smooth_weight: float, corr_weight: float) -> torch.Tensor:
    reg = features.new_tensor(0.0)
    if smooth_weight > 0.0:
        reg = reg + smooth_weight * latent_smoothness(features)
    if corr_weight > 0.0:
        reg = reg + corr_weight * latent_decorrelation(features)
    return reg


def evaluate_metrics(model, loader, loss_fn, device):
    model.eval()
    totals = torch.zeros(8, dtype=torch.float64, device=device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            batch = x.shape[0]
            residual = (pred - y).double()
            target = y.double()
            totals[0] += loss.detach().double() * batch
            totals[1] += torch.mean((pred - y) ** 2).double() * batch
            totals[2] += torch.mean(torch.abs(pred - y)).double() * batch
            totals[3] += batch
            totals[4] += torch.sum(residual**2)
            totals[5] += torch.sum(target)
            totals[6] += torch.sum(target**2)
            totals[7] += target.numel()
    denom = totals[3].clamp_min(1.0)
    sst = totals[6] - totals[5] ** 2 / totals[7].clamp_min(1.0)
    r2 = 1.0 - totals[4] / sst if sst > 0 else torch.tensor(float("nan"), device=device)
    return {
        "loss": float((totals[0] / denom).item()),
        "mse": float((totals[1] / denom).item()),
        "mae": float((totals[2] / denom).item()),
        "r2": float(r2.item()),
    }


def train_epoch(model, loader, optimizer, loss_fn, device, *, smooth_weight: float, corr_weight: float):
    model.train()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        features = model.encode_features(x)
        pred = model.decode_features(features)
        pred_loss = loss_fn(pred, y)
        reg_loss = latent_regularization(features, smooth_weight=smooth_weight, corr_weight=corr_weight)
        loss = pred_loss + reg_loss
        loss.backward()
        optimizer.step()
        batch = x.shape[0]
        totals[0] += loss.detach().double() * batch
        totals[1] += pred_loss.detach().double() * batch
        totals[2] += torch.mean((pred - y) ** 2).detach().double() * batch
        totals[3] += torch.mean(torch.abs(pred - y)).detach().double() * batch
        totals[4] += batch
    denom = totals[4].clamp_min(1.0)
    return {
        "loss": float((totals[0] / denom).item()),
        "pred_loss": float((totals[1] / denom).item()),
        "mse": float((totals[2] / denom).item()),
        "mae": float((totals[3] / denom).item()),
    }


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(config: dict) -> None:
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
    checkpoint = resolve(root, train_cfg["checkpoint"])
    history_csv = resolve(root, train_cfg["history_csv"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0
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
        )
        train_eval = evaluate_metrics(model, train_loader, loss_fn, device)
        val_stats = evaluate_metrics(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
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
        append_csv(history_csv, row)
        print(
            f"epoch={epoch} train_r2={train_eval['r2']:.4f} val_r2={val_stats['r2']:.4f} "
            f"train_loss={train_stats['loss']:.6g} val_loss={val_stats['loss']:.6g}",
            flush=True,
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_val": best_val, "config": config}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best_val={best_val:.6g}", flush=True)
                break


def denormalize_y(values: np.ndarray, stats: NormStats) -> np.ndarray:
    return values * stats.y_std[None, ...] + stats.y_mean[None, ...]


def r2_score(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    residual = pred - target
    sample_axes = tuple(range(1, target.ndim))
    sample_sse = np.sum(residual**2, axis=sample_axes)
    sample_mean = np.mean(target, axis=sample_axes, keepdims=True)
    sample_sst = np.sum((target - sample_mean) ** 2, axis=sample_axes)
    sample_r2 = np.full(sample_sse.shape, np.nan, dtype=np.float64)
    np.divide(sample_sse, sample_sst, out=sample_r2, where=sample_sst > 0)
    sample_r2[sample_sst > 0] = 1.0 - sample_r2[sample_sst > 0]
    overall = 1.0 - float(np.sum(residual**2)) / float(np.sum((target - np.mean(target)) ** 2))
    return overall, sample_r2


def evaluate(config: dict) -> None:
    root = project_dir()
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    _, val_loader, test_loader, stats = make_loaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(resolve(root, config["train"]["checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    loss_fn = build_loss(str(config["train"].get("loss", "huber")))
    val_metrics = evaluate_metrics(model, val_loader, loss_fn, device)
    model.eval()
    preds = []
    targets = []
    features = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu().numpy()
            latent = model.encode_features(x).cpu().numpy()
            preds.append(denormalize_y(pred, stats))
            targets.append(denormalize_y(y.numpy(), stats))
            features.append(latent)
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    latent_features = np.concatenate(features, axis=0)
    overall_r2, sample_r2 = r2_score(pred, target)
    mse = np.mean((pred - target) ** 2, axis=(1, 2, 3))
    mae = np.mean(np.abs(pred - target), axis=(1, 2, 3))
    sample_indices = np.asarray(test_loader.dataset.indices, dtype=np.int64)
    pred_path = resolve(root, config["output"]["predictions"])
    csv_path = resolve(root, config["output"]["r2_series"])
    summary_path = resolve(root, config["output"].get("summary", "analysis/temporal_bottleneck/outputs/evaluation_summary.txt"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_path,
        sample_index=sample_indices,
        latent_features=latent_features,
        prediction=pred,
        target=target,
        r2=sample_r2,
        mse=mse,
        mae=mae,
        overall_r2=np.asarray(overall_r2),
        val_r2=np.asarray(val_metrics["r2"]),
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "r2", "mse", "mae"])
        writer.writeheader()
        for pos, (idx, r2, mse_i, mae_i) in enumerate(zip(sample_indices, sample_r2, mse, mae)):
            writer.writerow({"test_position": pos, "sample_index": int(idx), "r2": float(r2), "mse": float(mse_i), "mae": float(mae_i)})
    summary = {
        "val_r2": val_metrics["r2"],
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "test_r2": float(overall_r2),
        "test_mse": float(np.mean(mse)),
        "test_mae": float(np.mean(mae)),
        "latent_feature_shape": list(latent_features.shape),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        for key, value in summary.items():
            handle.write(f"{key}: {value}\n")
    print({key: round(value, 8) for key, value in summary.items() if isinstance(value, float)}, flush=True)
    print(f"Saved {pred_path}", flush=True)
    print(f"Saved {csv_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final model_analysis temporal bottleneck heat-flux experiment.")
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--config", type=Path, default=Path("analysis/model_analysis/config_split.yaml"))
    args = parser.parse_args()
    config = load_config(resolve(project_dir(), args.config))
    if args.command == "train":
        train(config)
    elif args.command == "evaluate":
        evaluate(config)


if __name__ == "__main__":
    main()
