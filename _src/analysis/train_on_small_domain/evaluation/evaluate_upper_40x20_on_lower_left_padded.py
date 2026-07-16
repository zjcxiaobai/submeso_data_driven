#!/usr/bin/env python3
"""Evaluate the upper 40x20 model on lower-left data duplicated to 40x20."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils.npy_dataset import NormStats  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402


HERE = Path(__file__).resolve().parent


class DuplicatedLowerLeftDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        *,
        stats: NormStats,
        normalize_x: bool,
        normalize_y: bool,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.x.shape[-2:] != (20, 20) or self.y.shape[-2:] != (20, 20):
            raise ValueError(f"Expected lower_left 20x20 arrays, got X={self.x.shape}, y={self.y.shape}")
        self.stats = stats
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self.x[index], dtype=np.float32)
        y = np.asarray(self.y[index], dtype=np.float32)
        x = np.concatenate([x, x], axis=-1)
        y = np.concatenate([y, y], axis=-1)
        if x.shape[-2:] != (20, 40) or y.shape[-2:] != (20, 40):
            raise ValueError(f"Duplicated arrays should be 20x40, got X={x.shape}, y={y.shape}")
        if self.normalize_x:
            x = (x - self.stats.x_mean) / self.stats.x_std
        if self.normalize_y:
            y = (y - self.stats.y_mean) / self.stats.y_std
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate upper_40x20 model on lower_left duplicated horizontally to 40x20."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/unet_no_shuffle_upper_40x20.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "upper_40x20_model_lower_left_padded_r2.png",
    )
    return parser.parse_args()


def load_stats(path: Path) -> NormStats:
    with np.load(path) as data:
        return NormStats(
            x_mean=np.asarray(data["x_mean"], dtype=np.float32),
            x_std=np.asarray(data["x_std"], dtype=np.float32),
            y_mean=np.asarray(data["y_mean"], dtype=np.float32),
            y_std=np.asarray(data["y_std"], dtype=np.float32),
        )


def denormalize_y(values: torch.Tensor, stats: NormStats, enabled: bool, device: torch.device) -> torch.Tensor:
    if not enabled:
        return values
    y_mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=device)
    y_std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=device)
    return values * y_std + y_mean


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = prediction - target
    axes = tuple(range(1, target.ndim))
    sse = np.sum(residual**2, axis=axes)
    target_mean = np.mean(target, axis=axes, keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=axes)
    r2 = np.full(sse.shape, np.nan, dtype=np.float64)
    valid = sst > 0.0
    r2[valid] = 1.0 - sse[valid] / sst[valid]
    return r2


def overall_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    sse = float(np.sum((prediction - target) ** 2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - sse / sst if sst > 0.0 else float("nan")


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    stats: NormStats,
    normalize_y: bool,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            pred = denormalize_y(pred, stats, normalize_y, device)
            y = denormalize_y(y, stats, normalize_y, device)
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    return sample_r2(prediction, target), overall_r2(prediction, target)


def plot_r2(r2: np.ndarray, output: Path, overall: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    sample_index = np.arange(r2.size, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.plot(sample_index, r2, color="#125A9C", linewidth=0.75)
    ax.axhline(0.0, color="#777777", linewidth=0.7)
    ax.set_title(f"Lower-left duplicated to 40x20 | overall R2={overall:.3f}")
    ax.set_xlabel("Time/sample index")
    ax.set_ylabel("Per-sample R2")
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.6)
    finite = r2[np.isfinite(r2)]
    if finite.size:
        low = float(np.nanpercentile(finite, 1))
        high = float(np.nanpercentile(finite, 99))
        pad = max(0.05 * (high - low), 0.05)
        ax.set_ylim(low - pad, high + pad)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(args.device or config.get("device", "auto"))
    batch_size = int(args.batch_size or config["data"].get("batch_size", 64))
    normalize_y = bool(config["data"].get("normalize_y", True))

    stats = load_stats(PROJECT_DIR / config["data"]["stats_path"])
    dataset = DuplicatedLowerLeftDataset(
        PROJECT_DIR / "data/train_on_small_domain/lower_left/X.npy",
        PROJECT_DIR / "data/train_on_small_domain/lower_left/y.npy",
        stats=stats,
        normalize_x=bool(config["data"].get("normalize_x", True)),
        normalize_y=normalize_y,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = args.checkpoint or Path(config["train"]["best_checkpoint"])
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    r2, total_r2 = evaluate(model, loader, stats, normalize_y, device)
    print(f"Lower-left duplicated all data: n={r2.size}, overall_r2={total_r2:.6f}", flush=True)
    plot_r2(r2, args.output, total_r2)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
