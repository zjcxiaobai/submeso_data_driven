#!/usr/bin/env python3
"""Evaluate the upper 40x20 model on upper-test, sparse lower-train, and lower-test data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils.npy_dataset import (  # noqa: E402
    NormStats,
    NpyRegressionDataset,
    parse_temporal_offsets,
    split_indices,
    valid_center_indices,
)
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EvalCase:
    label: str
    sample_indices: np.ndarray
    r2: np.ndarray
    overall_r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot R2 for the upper_40x20 model on upper_40x20 test, "
            "lower_40x20 train snapshots sampled every 10 days, and lower_40x20 test."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/unet_no_shuffle_upper_40x20.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate. Default: train.best_checkpoint from --config.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--lower-train-stride",
        type=int,
        default=20,
        help="Stride through the lower_40x20 training split. Default 20 equals 10 days for 12-hour data.",
    )
    parser.add_argument(
        "--lower-train-start",
        type=int,
        default=0,
        help="Offset into the lower_40x20 training split. Default starts with the first training snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "upper_40x20_model_upper_test_lower_train_every10days_lower_test_r2_vs_time.png",
        help="PNG output path.",
    )
    return parser.parse_args()


def load_stats(path: Path) -> NormStats:
    if not path.exists():
        raise FileNotFoundError(f"Missing normalization stats: {path}")
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
    reduce_axes = tuple(range(1, target.ndim))
    sse = np.sum(residual**2, axis=reduce_axes)
    target_mean = np.mean(target, axis=reduce_axes, keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=reduce_axes)
    r2 = np.full(sse.shape, np.nan, dtype=np.float64)
    np.divide(sse, sst, out=r2, where=sst > 0.0)
    r2[sst > 0.0] = 1.0 - r2[sst > 0.0]
    return r2


def overall_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    sse = float(np.sum((prediction - target) ** 2))
    sst = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - sse / sst if sst > 0.0 else float("nan")


def split_indices_for_config(config: dict, root: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    x_path = root / data_cfg["x_path"]
    probe = np.load(x_path, mmap_mode="r")
    temporal_offsets = parse_temporal_offsets(data_cfg)
    centers = valid_center_indices(probe.shape[0], temporal_offsets)
    train_pos, val_pos, test_pos = split_indices(centers.shape[0], data_cfg["split"], seed)
    return centers[train_pos], centers[val_pos], centers[test_pos]


def dataset_for_domain(config: dict, root: Path, domain: str, stats: NormStats) -> NpyRegressionDataset:
    data_cfg = config["data"]
    data_dir = root / "data/train_on_small_domain" / domain
    return NpyRegressionDataset(
        data_dir / "X.npy",
        data_dir / "y.npy",
        stats=stats,
        normalize_x=bool(data_cfg.get("normalize_x", True)),
        normalize_y=bool(data_cfg.get("normalize_y", True)),
        temporal_offsets=parse_temporal_offsets(data_cfg),
    )


def evaluate_case(
    *,
    model: torch.nn.Module,
    dataset: NpyRegressionDataset,
    sample_indices: np.ndarray,
    stats: NormStats,
    normalize_y: bool,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, float]:
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
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


def plot_cases(cases: list[EvalCase], output: Path) -> None:
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
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=False, constrained_layout=True)
    for ax, case in zip(axes, cases):
        finite = case.r2[np.isfinite(case.r2)]
        ax.plot(case.sample_indices, case.r2, color="#125A9C", linewidth=0.75)
        ax.axhline(0.0, color="#777777", linewidth=0.7)
        ax.set_title(f"{case.label} | overall R2={case.overall_r2:.3f}")
        ax.set_xlabel("Time/sample index")
        ax.set_ylabel("Per-sample R2")
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.6)
        if finite.size:
            low = float(np.nanpercentile(finite, 1))
            high = float(np.nanpercentile(finite, 99))
            pad = max(0.05 * (high - low), 0.05)
            ax.set_ylim(low - pad, high + pad)
    fig.suptitle("Upper 40x20 model evaluation", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = get_device(args.device or config.get("device", "auto"))
    batch_size = int(args.batch_size or config["data"].get("batch_size", 64))
    normalize_y = bool(config["data"].get("normalize_y", True))

    checkpoint_path = args.checkpoint or Path(config["train"]["best_checkpoint"])
    stats = load_stats(PROJECT_DIR / config["data"]["stats_path"])

    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    upper_dataset = dataset_for_domain(config, PROJECT_DIR, "upper_40x20", stats)
    lower_dataset = dataset_for_domain(config, PROJECT_DIR, "lower_40x20", stats)
    train_indices, _, test_indices = split_indices_for_config(config, PROJECT_DIR, seed)
    if args.lower_train_start < 0:
        raise ValueError("--lower-train-start must be nonnegative")
    if args.lower_train_stride <= 0:
        raise ValueError("--lower-train-stride must be positive")
    lower_train_indices = train_indices[args.lower_train_start :: args.lower_train_stride]
    if lower_train_indices.size == 0:
        raise ValueError("--lower-train-start is outside the training split")

    cases: list[EvalCase] = []
    for label, dataset, sample_indices in (
        ("Upper 40x20 test", upper_dataset, test_indices),
        (
            f"Lower 40x20 train every {args.lower_train_stride} snapshots",
            lower_dataset,
            lower_train_indices,
        ),
        ("Lower 40x20 test", lower_dataset, test_indices),
    ):
        r2, case_overall_r2 = evaluate_case(
            model=model,
            dataset=dataset,
            sample_indices=sample_indices,
            stats=stats,
            normalize_y=normalize_y,
            device=device,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        print(f"{label}: n={sample_indices.size}, overall_r2={case_overall_r2:.6f}", flush=True)
        cases.append(EvalCase(label, sample_indices, r2, case_overall_r2))

    plot_cases(cases, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
