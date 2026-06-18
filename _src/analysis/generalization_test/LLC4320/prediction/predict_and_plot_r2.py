#!/usr/bin/env python3
"""Predict LLC4320 heat flux and plot per-snapshot R2 through time."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
LLC4320_DIR = SCRIPT_DIR.parent
PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models import build_model  # noqa: E402
from utils import get_device, load_config  # noqa: E402


DEFAULT_DATA_DIR = LLC4320_DIR / "data"
DEFAULT_CONFIG = PROJECT_DIR / "outputs_no_shuffle_overfitting_test/config.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "outputs_no_shuffle_overfitting_test/checkpoints/best.pt"
)
DEFAULT_STATS = PROJECT_DIR / "outputs_no_shuffle_overfitting_test/norm_stats.npz"

COLOR = "#1F5A85"
ACCENT = "#B64A3B"
NEUTRAL = "#3A3A3A"
GRID = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the no_shuffle_overfitting_test U-Net to all LLC4320 X "
            "snapshots, compare with y, and plot the time series of R2."
        )
    )
    parser.add_argument("--x", type=Path, default=DEFAULT_DATA_DIR / "X.npy")
    parser.add_argument("--y", type=Path, default=DEFAULT_DATA_DIR / "y.npy")
    parser.add_argument(
        "--time-index",
        type=Path,
        default=DEFAULT_DATA_DIR / "time_index.npy",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size; 1 minimizes memory use for the large U-Net.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device, for example cpu or cuda.",
    )
    return parser.parse_args()


def validate_inputs(
    x: np.ndarray,
    y: np.ndarray,
    time_index: np.ndarray,
    expected_channels: int,
) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected 4D channel-first arrays; got X={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0] or x.shape[0] != time_index.size:
        raise ValueError(
            "Sample count mismatch: "
            f"X={x.shape[0]}, y={y.shape[0]}, time_index={time_index.size}"
        )
    if x.shape[1] != expected_channels:
        raise ValueError(
            f"Model expects {expected_channels} input channels but X has {x.shape[1]}"
        )
    if y.shape[1] != 1 or x.shape[-2:] != y.shape[-2:]:
        raise ValueError(f"Incompatible X/y shapes: X={x.shape}, y={y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("X or y contains NaN/inf values")


def predict(
    x: np.ndarray,
    model: torch.nn.Module,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    normalize_x: bool,
    normalize_y: bool,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            batch = np.asarray(x[start:stop], dtype=np.float32)
            if normalize_x:
                batch = (batch - x_mean) / x_std
            tensor = torch.from_numpy(batch).to(device)
            original_height, original_width = tensor.shape[-2:]
            pad_height = max(0, 16 - original_height)
            pad_width = max(0, 16 - original_width)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            if pad_left or pad_right:
                tensor = F.pad(
                    tensor,
                    (pad_left, pad_right, 0, 0),
                    mode="circular",
                )
            if pad_top or pad_bottom:
                tensor = F.pad(
                    tensor,
                    (0, 0, pad_top, pad_bottom),
                    mode="reflect",
                )
            prediction = model(tensor)
            prediction = prediction[
                ...,
                pad_top : pad_top + original_height,
                pad_left : pad_left + original_width,
            ]
            prediction = prediction.cpu().numpy()
            if normalize_y:
                prediction = prediction * y_std + y_mean
            outputs.append(prediction.astype(np.float32, copy=False))
            print(f"Predicted snapshots {start + 1}-{stop} of {x.shape[0]}", flush=True)
    return np.concatenate(outputs, axis=0)


def per_snapshot_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(range(1, target.ndim))
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    sse = np.sum(residual**2, axis=axes)
    target_mean = np.mean(target, axis=axes, keepdims=True, dtype=np.float64)
    sst = np.sum((target.astype(np.float64) - target_mean) ** 2, axis=axes)
    r2 = np.full(target.shape[0], np.nan, dtype=np.float64)
    valid = sst > 0.0
    r2[valid] = 1.0 - sse[valid] / sst[valid]
    mse = np.mean(residual**2, axis=axes)
    mae = np.mean(np.abs(residual), axis=axes)
    return r2, mse, mae


def overall_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction64 = prediction.astype(np.float64)
    target64 = target.astype(np.float64)
    residual = prediction64 - target64
    sse = float(np.sum(residual**2))
    sst = float(np.sum((target64 - np.mean(target64)) ** 2))
    return {
        "r2": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
    }


def parse_timestamps(time_index: np.ndarray) -> list[datetime]:
    timestamps: list[datetime] = []
    for value in time_index:
        text = str(int(value))
        if len(text) != 10:
            raise ValueError(
                f"Expected YYYYMMDDHH timestamps in time_index.npy; got {text!r}"
            )
        timestamps.append(datetime.strptime(text, "%Y%m%d%H"))
    return timestamps


def save_metrics_csv(
    path: Path,
    time_index: np.ndarray,
    timestamps: list[datetime],
    r2: np.ndarray,
    mse: np.ndarray,
    mae: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "time_index", "datetime", "r2", "mse", "mae"],
        )
        writer.writeheader()
        for index, values in enumerate(zip(time_index, timestamps, r2, mse, mae)):
            time_value, timestamp, r2_value, mse_value, mae_value = values
            writer.writerow(
                {
                    "sample_index": index,
                    "time_index": int(time_value),
                    "datetime": timestamp.isoformat(sep=" "),
                    "r2": float(r2_value),
                    "mse": float(mse_value),
                    "mae": float(mae_value),
                }
            )


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.labelcolor": NEUTRAL,
            "xtick.color": NEUTRAL,
            "ytick.color": NEUTRAL,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_r2(
    timestamps: list[datetime],
    r2: np.ndarray,
    overall_r2: float,
    output_prefix: Path,
) -> None:
    finite = r2[np.isfinite(r2)]
    if finite.size == 0:
        raise ValueError("No finite per-snapshot R2 values to plot")

    apply_figure_style()
    fig, ax = plt.subplots(figsize=(5.2, 2.7), constrained_layout=True)
    ax.plot(
        timestamps,
        r2,
        color=COLOR,
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.25,
        zorder=3,
    )
    mean_r2 = float(np.mean(finite))
    ax.axhline(
        mean_r2,
        color=ACCENT,
        linewidth=0.9,
        linestyle=(0, (4, 2.5)),
        label=fr"Mean per-snapshot $R^2$ = {mean_r2:.3f}",
        zorder=2,
    )
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.65, alpha=0.7, zorder=1)
    ax.set_xlabel("LLC4320 snapshot time (UTC)")
    ax.set_ylabel(r"Spatial $R^2$")
    ax.set_title("No-shuffle model generalization to LLC4320", fontsize=8, pad=5)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(0, 12)))
    ax.grid(axis="y", color=GRID, linewidth=0.45, alpha=0.65)
    ax.legend(loc="best", fontsize=6.5)
    ax.text(
        0.99,
        0.03,
        fr"Overall $R^2$ = {overall_r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=NEUTRAL,
        fontsize=6.5,
    )

    span = float(np.max(finite) - np.min(finite))
    pad = max(0.08, 0.12 * span)
    ax.set_ylim(float(np.min(finite)) - pad, min(1.0, float(np.max(finite)) + pad))

    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    x_shape: tuple[int, ...],
    y_shape: tuple[int, ...],
    metrics: dict[str, float],
    r2: np.ndarray,
) -> None:
    finite = r2[np.isfinite(r2)]
    with path.open("w") as handle:
        handle.write("LLC4320 no_shuffle_overfitting_test prediction summary\n")
        handle.write(f"X_shape: {x_shape}\n")
        handle.write(f"y_shape: {y_shape}\n")
        handle.write(f"snapshot_count: {len(r2)}\n")
        handle.write(f"overall_r2: {metrics['r2']:.10g}\n")
        handle.write(f"overall_mse: {metrics['mse']:.10g}\n")
        handle.write(f"overall_mae: {metrics['mae']:.10g}\n")
        handle.write(f"mean_per_snapshot_r2: {np.mean(finite):.10g}\n")
        handle.write(f"median_per_snapshot_r2: {np.median(finite):.10g}\n")
        handle.write(f"min_per_snapshot_r2: {np.min(finite):.10g}\n")
        handle.write(f"max_per_snapshot_r2: {np.max(finite):.10g}\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config.expanduser().resolve())
    data_cfg = config["data"]
    device = get_device(args.device or config.get("device", "auto"))

    x = np.load(args.x.expanduser().resolve(), mmap_mode="r")
    y = np.load(args.y.expanduser().resolve(), mmap_mode="r")
    time_index = np.load(args.time_index.expanduser().resolve())
    stats_file = np.load(args.stats.expanduser().resolve())
    x_mean = stats_file["x_mean"]
    x_std = stats_file["x_std"]
    y_mean = stats_file["y_mean"]
    y_std = stats_file["y_std"]

    expected_channels = int(config["model"]["in_channels"])
    validate_inputs(x, y, time_index, expected_channels)
    if x_mean.shape != (expected_channels, 1, 1):
        raise ValueError(
            f"Unexpected x_mean shape {x_mean.shape}; expected {(expected_channels, 1, 1)}"
        )

    print(f"Device: {device}", flush=True)
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = build_model(config["model"]).to(device)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location=device,
        mmap=True,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])

    prediction = predict(
        x,
        model,
        x_mean,
        x_std,
        y_mean,
        y_std,
        bool(data_cfg.get("normalize_x", True)),
        bool(data_cfg.get("normalize_y", True)),
        args.batch_size,
        device,
    )
    if prediction.shape != y.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match y shape {y.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains NaN/inf values")

    target = np.asarray(y, dtype=np.float32)
    r2, mse, mae = per_snapshot_metrics(prediction, target)
    metrics = overall_metrics(prediction, target)
    timestamps = parse_timestamps(time_index)

    np.save(output_dir / "prediction.npy", prediction)
    np.savez_compressed(
        output_dir / "prediction_and_metrics.npz",
        sample_index=np.arange(x.shape[0], dtype=np.int64),
        time_index=time_index,
        prediction=prediction,
        target=target,
        r2=r2,
        mse=mse,
        mae=mae,
        overall_r2=np.asarray(metrics["r2"]),
        overall_mse=np.asarray(metrics["mse"]),
        overall_mae=np.asarray(metrics["mae"]),
    )
    save_metrics_csv(
        output_dir / "r2_time_series.csv",
        time_index,
        timestamps,
        r2,
        mse,
        mae,
    )
    write_summary(
        output_dir / "prediction_summary.txt",
        tuple(x.shape),
        tuple(y.shape),
        metrics,
        r2,
    )
    plot_r2(
        timestamps,
        r2,
        metrics["r2"],
        output_dir / "r2_time_series",
    )

    print(f"Saved outputs in {output_dir}", flush=True)
    print(f"Overall R2: {metrics['r2']:.6f}", flush=True)
    print(f"Mean per-snapshot R2: {np.nanmean(r2):.6f}", flush=True)


if __name__ == "__main__":
    main()
