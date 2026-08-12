#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils import build_dataloaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402


DEFAULT_CONFIG = PROJECT_DIR / "configs/unet_encoder_convlstm_no_shuffle_upper_40x20.yaml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_DIR
    / "analysis/explanability/decoder_channel_ablation"
    / "layerwise_feature_map_ablation_sample_9523"
)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    module_name: str
    family: str
    description: str


LAYER_SPECS = [
    LayerSpec("encoder_lstm_20x40", "convlstm1", "encoder", "Post-ConvLSTM encoder level 1"),
    LayerSpec("encoder_lstm_10x20", "convlstm2", "encoder", "Post-ConvLSTM encoder level 2"),
    LayerSpec("encoder_lstm_5x10", "convlstm3", "encoder", "Post-ConvLSTM encoder level 3"),
    LayerSpec(
        "bottleneck_lstm_2x5",
        "bottleneck_convlstm",
        "bottleneck",
        "Post-ConvLSTM bottleneck",
    ),
    LayerSpec("decoder_up3_5x10", "up3", "decoder", "Decoder level 3 output"),
    LayerSpec("decoder_up2_10x20", "up2", "decoder", "Decoder level 2 output"),
    LayerSpec("decoder_up1_20x40", "up1", "decoder", "Final decoder feature output"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank channels at temporally aggregated encoder, bottleneck, and decoder stages "
            "by prediction degradation after channel ablation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-index", type=int, default=9523)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--plot-columns", type=int, default=10)
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=[spec.name for spec in LAYER_SPECS],
        default=[spec.name for spec in LAYER_SPECS],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


def denormalize_y(values: torch.Tensor, stats, normalize_y: bool) -> torch.Tensor:
    if not normalize_y:
        return values
    mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=values.device)
    return values * std + mean


def snapshot_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = prediction - target
    sse = float(np.sum(residual * residual, dtype=np.float64))
    centered = target - np.mean(target, dtype=np.float64)
    sst = float(np.sum(centered * centered, dtype=np.float64))
    return float("nan") if sst <= 0.0 else 1.0 - sse / sst


def select_sample(test_subset, sample_index: int) -> tuple[int, torch.Tensor, torch.Tensor]:
    test_indices = np.asarray(test_subset.indices, dtype=np.int64)
    matches = np.flatnonzero(test_indices == sample_index)
    if matches.size == 0:
        raise ValueError(f"sample_index={sample_index} is not in the configured test split")
    position = int(matches[0])
    x, y = test_subset[position]
    return position, x.unsqueeze(0), y.unsqueeze(0)


def output_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


class FeatureCapture:
    def __init__(self) -> None:
        self.feature: torch.Tensor | None = None

    def __call__(self, _module, _inputs, output):
        self.feature = output_tensor(output).detach()


def capture_feature(
    model: torch.nn.Module,
    module: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    capture = FeatureCapture()
    handle = module.register_forward_hook(capture)
    try:
        with torch.no_grad():
            prediction = model(x)
    finally:
        handle.remove()
    if capture.feature is None:
        raise RuntimeError("Layer hook did not capture a feature tensor")
    return prediction, capture.feature


def ablation_hook(channel: int):
    def hook(_module, _inputs, output):
        if torch.is_tensor(output):
            modified = output.clone()
            modified[:, channel] = 0.0
            return modified
        if isinstance(output, tuple) and len(output) == 2:
            hidden, cell = output
            hidden_modified = hidden.clone()
            cell_modified = cell.clone()
            hidden_modified[:, channel] = 0.0
            cell_modified[:, channel] = 0.0
            return hidden_modified, cell_modified
        raise TypeError(f"Unsupported layer output type: {type(output)!r}")

    return hook


def evaluate_layer(
    model: torch.nn.Module,
    module: torch.nn.Module,
    x: torch.Tensor,
    target_physical: np.ndarray,
    stats,
    normalize_y: bool,
    baseline_prediction: np.ndarray,
    baseline_r2: float,
    channel_count: int,
) -> dict[str, np.ndarray]:
    ablated_r2 = np.empty(channel_count, dtype=np.float64)
    rms_prediction_change = np.empty(channel_count, dtype=np.float64)
    mean_abs_prediction_change = np.empty(channel_count, dtype=np.float64)
    for channel in range(channel_count):
        handle = module.register_forward_hook(ablation_hook(channel))
        try:
            with torch.no_grad():
                prediction = denormalize_y(model(x), stats, normalize_y)
        finally:
            handle.remove()
        prediction_np = prediction.detach().cpu().numpy()[0, 0]
        delta = baseline_prediction - prediction_np
        ablated_r2[channel] = snapshot_r2(prediction_np, target_physical)
        rms_prediction_change[channel] = float(np.sqrt(np.mean(delta * delta)))
        mean_abs_prediction_change[channel] = float(np.mean(np.abs(delta)))
        if (channel + 1) % 32 == 0 or channel + 1 == channel_count:
            print(f"    ablated {channel + 1}/{channel_count} channels", flush=True)
    return {
        "ablated_r2": ablated_r2,
        "r2_drop": baseline_r2 - ablated_r2,
        "rms_prediction_change": rms_prediction_change,
        "mean_abs_prediction_change": mean_abs_prediction_change,
    }


def write_rank_csv(
    path: Path,
    metrics: dict[str, np.ndarray],
) -> np.ndarray:
    order = np.argsort(metrics["r2_drop"])[::-1]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "channel",
                "ablated_r2",
                "r2_drop",
                "rms_prediction_change",
                "mean_abs_prediction_change",
            ],
        )
        writer.writeheader()
        for rank, channel in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "channel": int(channel),
                    "ablated_r2": float(metrics["ablated_r2"][channel]),
                    "r2_drop": float(metrics["r2_drop"][channel]),
                    "rms_prediction_change": float(
                        metrics["rms_prediction_change"][channel]
                    ),
                    "mean_abs_prediction_change": float(
                        metrics["mean_abs_prediction_change"][channel]
                    ),
                }
            )
    return order


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6,
            "axes.linewidth": 0.6,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_top_feature_maps(
    output: Path,
    features: np.ndarray,
    channels: np.ndarray,
    r2_drop: np.ndarray,
    layer_title: str,
    columns: int,
) -> None:
    apply_style()
    count = len(channels)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, max(1.4, rows * 1.0)),
        constrained_layout=True,
        squeeze=False,
    )
    for rank, (channel, ax) in enumerate(zip(channels, axes.ravel()), start=1):
        image = features[int(channel)]
        vmax = float(np.nanpercentile(np.abs(image), 99.0))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = None
        ax.imshow(
            image,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=None if vmax is None else -vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"#{rank} ch {int(channel)}\n"
            + r"$\Delta R^2$"
            + f"={r2_drop[int(channel)]:.3g}",
            fontsize=5.5,
            pad=1.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[count:]:
        ax.axis("off")
    fig.suptitle(layer_title, fontsize=9)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.plot_columns < 1:
        raise ValueError("--top-k and --plot-columns must be positive")
    config_path = resolve_path(args.config, PROJECT_DIR)
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(args.device or config.get("device", "auto"))
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False
    _, _, test_loader, stats = build_dataloaders(
        config, PROJECT_DIR, int(config.get("seed", 42))
    )
    test_position, x_cpu, y_cpu = select_sample(
        test_loader.dataset, args.sample_index
    )
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    normalize_y = bool(config["data"].get("normalize_y", True))
    target_physical = (
        denormalize_y(y, stats, normalize_y).detach().cpu().numpy()[0, 0]
    )

    model = build_model(config["model"]).to(device)
    checkpoint_path = resolve_path(
        args.checkpoint or Path(config["train"]["best_checkpoint"]), PROJECT_DIR
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        baseline_tensor = denormalize_y(model(x), stats, normalize_y)
    baseline_prediction = baseline_tensor.detach().cpu().numpy()[0, 0]
    baseline_r2 = snapshot_r2(baseline_prediction, target_physical)

    output_root = args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_specs = [spec for spec in LAYER_SPECS if spec.name in args.layers]
    all_summaries = []
    for layer_number, spec in enumerate(selected_specs, start=1):
        layer_dir = output_root / spec.name
        if layer_dir.exists() and not args.overwrite:
            raise FileExistsError(f"{layer_dir} exists; pass --overwrite")
        layer_dir.mkdir(parents=True, exist_ok=True)
        module = model.get_submodule(spec.module_name)
        _, captured = capture_feature(model, module, x)
        features = captured.detach().cpu().numpy()[0]
        channel_count = int(features.shape[0])
        print(
            f"[{layer_number}/{len(selected_specs)}] {spec.name}: "
            f"shape={features.shape}, evaluating {channel_count} channels",
            flush=True,
        )
        metrics = evaluate_layer(
            model,
            module,
            x,
            target_physical,
            stats,
            normalize_y,
            baseline_prediction,
            baseline_r2,
            channel_count,
        )
        order = write_rank_csv(layer_dir / "channel_ablation_rank.csv", metrics)
        top_count = min(args.top_k, channel_count)
        top_channels = order[:top_count]
        np.savez_compressed(
            layer_dir / "top_feature_maps_and_ablation.npz",
            sample_index=np.asarray(args.sample_index, dtype=np.int64),
            feature_maps=features[top_channels].astype(np.float32),
            top_channels=top_channels.astype(np.int64),
            all_channels=np.arange(channel_count, dtype=np.int64),
            baseline_r2=np.asarray(baseline_r2),
            **metrics,
        )
        plot_top_feature_maps(
            layer_dir / f"top{top_count}_feature_maps.png",
            features,
            top_channels,
            metrics["r2_drop"],
            f"{spec.description}: top {top_count} channels by prediction degradation",
            args.plot_columns,
        )
        summary = {
            "layer": spec.name,
            "module": spec.module_name,
            "family": spec.family,
            "description": spec.description,
            "feature_shape_channels_y_x": list(features.shape),
            "sample_index": args.sample_index,
            "test_position": test_position,
            "baseline_r2": baseline_r2,
            "ranking_metric": "baseline snapshot R2 minus channel-ablated snapshot R2",
            "top_k": top_count,
            "top_channels": top_channels.tolist(),
            "largest_r2_drop": float(metrics["r2_drop"][top_channels[0]]),
            "median_r2_drop": float(np.nanmedian(metrics["r2_drop"])),
            "convlstm_ablation": (
                "Hidden and cell-state channel are zeroed at every recurrent step"
                if "lstm" in spec.name
                else None
            ),
        }
        (layer_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        all_summaries.append(summary)

    root_summary = {
        "sample_index": args.sample_index,
        "test_position": test_position,
        "baseline_r2": baseline_r2,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "top_k": args.top_k,
        "layers": all_summaries,
        "correlations_calculated": False,
    }
    (output_root / "layerwise_ablation_summary.json").write_text(
        json.dumps(root_summary, indent=2) + "\n"
    )
    print(f"Saved layer-wise feature-map ablation under {output_root}", flush=True)


if __name__ == "__main__":
    main()
