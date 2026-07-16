#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils import build_dataloaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DEFAULT_CONFIG = PROJECT_DIR / "configs/unet_encoder_convlstm_no_shuffle_upper_40x20.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis/explanability/integrated_gradients_encoder_convlstm"
DEFAULT_ALL_TEST_RESULT = DEFAULT_OUTPUT_DIR / "all_test_mean_ig_mean_abs_zero_steps64.npz"
DEFAULT_FEATURE_NAMES = [
    "U velocity",
    "V velocity",
    "W velocity",
    "Vorticity",
    "Strain",
    "Potential temperature",
    "Horizontal speed",
    "KPP boundary layer depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated Gradients attribution for the encoder-ConvLSTM upper 40x20 model. "
            "The default scalar diagnostic is mean(abs(predicted heat flux)) over the output map."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Override device, for example cpu or cuda.")
    parser.add_argument("--steps", type=int, default=64, help="Integrated Gradients path steps.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Original dataset sample index to explain. If omitted, use --test-position.",
    )
    parser.add_argument(
        "--test-position",
        type=int,
        default=0,
        help="Position inside the test split when --sample-index is omitted.",
    )
    parser.add_argument(
        "--diagnostic",
        choices=["mean_abs", "mean_signed", "pixel"],
        default="mean_abs",
        help="Scalar J to attribute: mean absolute predicted flux, signed spatial mean, or one pixel.",
    )
    parser.add_argument("--pixel-y", type=int, default=None, help="Output y index for --diagnostic pixel.")
    parser.add_argument("--pixel-x", type=int, default=None, help="Output x index for --diagnostic pixel.")
    parser.add_argument(
        "--baseline",
        choices=["zero", "sample_mean"],
        default="zero",
        help=(
            "IG baseline in normalized input space. zero is training-mean input when normalize_x=true; "
            "sample_mean keeps each input channel spatial mean."
        ),
    )
    parser.add_argument(
        "--feature-names",
        nargs="+",
        default=DEFAULT_FEATURE_NAMES,
        help="Names for the frame channels. Default matches the 8-variable upper 40x20 data.",
    )
    parser.add_argument(
        "--all-test",
        action="store_true",
        help="Average Integrated Gradients summaries over every sample in the held-out test split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for testing --all-test before running the complete split.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Progress-report interval in all-test mode.",
    )
    parser.add_argument(
        "--flux-subset", choices=["all", "high", "low"], default="all",
        help="Restrict --all-test to a flux quantile selected from a previous all-test result.",
    )
    parser.add_argument(
        "--flux-quantile", type=float, default=0.25,
        help="Fraction retained for --flux-subset high or low.",
    )
    parser.add_argument(
        "--selection-npz", type=Path, default=DEFAULT_ALL_TEST_RESULT,
        help="Previous all-test NPZ containing sample_indices and objective_values.",
    )
    return parser.parse_args()


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def resolve_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


def denormalize_y(values: torch.Tensor, stats, normalize_y: bool) -> torch.Tensor:
    if not normalize_y:
        return values
    y_mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=values.device)
    y_std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=values.device)
    return values * y_std + y_mean


def select_test_sample(test_subset, sample_index: int | None, test_position: int) -> tuple[int, torch.Tensor, torch.Tensor]:
    test_indices = np.asarray(test_subset.indices, dtype=np.int64)
    if sample_index is None:
        if not 0 <= test_position < len(test_indices):
            raise IndexError(f"test_position={test_position} outside test split length {len(test_indices)}")
        position = int(test_position)
    else:
        matches = np.flatnonzero(test_indices == int(sample_index))
        if matches.size == 0:
            raise ValueError(f"sample_index={sample_index} is not in the test split")
        position = int(matches[0])

    x, y = test_subset[position]
    return int(test_indices[position]), x.unsqueeze(0), y.unsqueeze(0)


def make_baseline(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "sample_mean":
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x).clone()
    raise ValueError(f"Unknown baseline mode: {mode}")


def scalar_diagnostic(prediction_physical: torch.Tensor, diagnostic: str, pixel_y: int | None, pixel_x: int | None) -> torch.Tensor:
    field = prediction_physical[:, 0]
    if diagnostic == "mean_abs":
        return field.abs().mean()
    if diagnostic == "mean_signed":
        return field.mean()
    if diagnostic == "pixel":
        height, width = field.shape[-2:]
        y = height // 2 if pixel_y is None else int(pixel_y)
        x = width // 2 if pixel_x is None else int(pixel_x)
        if not 0 <= y < height or not 0 <= x < width:
            raise IndexError(f"pixel ({y}, {x}) outside output shape {(height, width)}")
        return field[0, y, x]
    raise ValueError(f"Unknown diagnostic: {diagnostic}")


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    stats,
    normalize_y: bool,
    steps: int,
    diagnostic: str,
    pixel_y: int | None,
    pixel_x: int | None,
    verbose: bool = True,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    if steps < 1:
        raise ValueError("--steps must be positive")

    delta = x - baseline
    total_grad = torch.zeros_like(x)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, dtype=x.dtype, device=x.device)

    for step_number, alpha in enumerate(alphas, start=1):
        path_input = (baseline + alpha * delta).detach().requires_grad_(True)
        pred = model(path_input)
        pred_physical = denormalize_y(pred, stats, normalize_y)
        objective = scalar_diagnostic(pred_physical, diagnostic, pixel_y, pixel_x)
        grad = torch.autograd.grad(objective, path_input, retain_graph=False, create_graph=False)[0]
        total_grad += grad.detach()
        if verbose and (step_number == 1 or step_number == steps or step_number % max(1, steps // 8) == 0):
            print(f"  IG step {step_number}/{steps}", flush=True)

    attribution = delta * total_grad / float(steps)
    with torch.no_grad():
        pred = denormalize_y(model(x), stats, normalize_y)
        objective_value = float(scalar_diagnostic(pred, diagnostic, pixel_y, pixel_x).detach().cpu())
    return attribution.detach(), objective_value, pred.detach()


def reshape_attribution(attribution: np.ndarray, sequence_length: int, frame_channels: int) -> np.ndarray:
    if attribution.shape[1] != sequence_length * frame_channels:
        raise ValueError(
            f"Attribution channel count {attribution.shape[1]} does not match "
            f"sequence_length*frame_channels={sequence_length * frame_channels}"
        )
    return attribution[0].reshape(sequence_length, frame_channels, attribution.shape[-2], attribution.shape[-1])


def summarize_attribution(attr: np.ndarray) -> dict[str, np.ndarray]:
    abs_attr = np.abs(attr)
    return {
        "feature_importance": abs_attr.sum(axis=(0, 2, 3)),
        "lag_importance": abs_attr.sum(axis=(1, 2, 3)),
        "spatial_importance": abs_attr.sum(axis=(0, 1)),
        "lag_feature_importance": abs_attr.sum(axis=(2, 3)),
        "feature_spatial_importance": abs_attr.sum(axis=0),
        "signed_feature_importance": attr.sum(axis=(0, 2, 3)),
        "signed_lag_importance": attr.sum(axis=(1, 2, 3)),
        "signed_lag_feature_importance": attr.sum(axis=(2, 3)),
    }


def write_csvs(output_prefix: Path, feature_names: list[str], temporal_offsets: list[int], summary: dict[str, np.ndarray]) -> None:
    feature_total = summary["feature_importance"]
    lag_total = summary["lag_importance"]
    feature_fraction = feature_total / feature_total.sum() if feature_total.sum() > 0 else np.full_like(feature_total, np.nan)
    lag_fraction = lag_total / lag_total.sum() if lag_total.sum() > 0 else np.full_like(lag_total, np.nan)

    with output_prefix.with_name(output_prefix.name + "_feature_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_index", "feature_name", "abs_attribution", "fraction"])
        writer.writeheader()
        for rank, feature_index in enumerate(np.argsort(feature_total)[::-1], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "feature_name": feature_names[int(feature_index)],
                    "abs_attribution": float(feature_total[int(feature_index)]),
                    "fraction": float(feature_fraction[int(feature_index)]),
                }
            )

    with output_prefix.with_name(output_prefix.name + "_lag_importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "lag_index", "temporal_offset", "abs_attribution", "fraction"])
        writer.writeheader()
        for rank, lag_index in enumerate(np.argsort(lag_total)[::-1], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "lag_index": int(lag_index),
                    "temporal_offset": int(temporal_offsets[int(lag_index)]),
                    "abs_attribution": float(lag_total[int(lag_index)]),
                    "fraction": float(lag_fraction[int(lag_index)]),
                }
            )


def imshow_with_colorbar(
    fig, ax, image: np.ndarray, *, cmap: str, title: str, symmetric: bool = False, aspect: str = "auto"
):
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(image), 99.0))
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(image, 1.0))
        vmax = float(np.nanpercentile(image, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None
    mesh = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)


def save_summary_plot(
    output_prefix: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    attr: np.ndarray,
    feature_names: list[str],
    temporal_offsets: list[int],
    summary: dict[str, np.ndarray],
    diagnostic: str,
    objective_value: float,
    sample_index: int,
) -> None:
    apply_plot_style()
    feature_total = summary["feature_importance"]
    lag_total = summary["lag_importance"]
    spatial = summary["spatial_importance"]

    fig = plt.figure(figsize=(7.1, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.95, 1.15])
    ax_pred = fig.add_subplot(grid[0, 0])
    ax_target = fig.add_subplot(grid[0, 1])
    ax_spatial = fig.add_subplot(grid[0, 2])
    ax_feature = fig.add_subplot(grid[1, :2])
    ax_lag = fig.add_subplot(grid[1, 2])
    channel_axes = [fig.add_subplot(grid[2, col]) for col in range(3)]

    imshow_with_colorbar(fig, ax_pred, prediction[0, 0], cmap="RdBu_r", title="Prediction", symmetric=True)
    imshow_with_colorbar(fig, ax_target, target[0, 0], cmap="RdBu_r", title="Truth", symmetric=True)
    imshow_with_colorbar(fig, ax_spatial, spatial, cmap="magma", title="IG |attribution|")

    order = np.argsort(feature_total)
    ax_feature.barh(np.arange(len(order)), feature_total[order], color="#0F4D92", height=0.72)
    ax_feature.set_yticks(np.arange(len(order)))
    ax_feature.set_yticklabels([feature_names[index] for index in order])
    ax_feature.set_xlabel("Summed absolute attribution")
    ax_feature.set_title("Attribution by variable", fontsize=8, pad=3)
    ax_feature.grid(axis="x", color="#D0D0D0", linewidth=0.35, alpha=0.65)

    ax_lag.plot(temporal_offsets, lag_total, color="#B64342", marker="o", linewidth=1.0, markersize=3)
    ax_lag.set_xlabel("Temporal offset")
    ax_lag.set_ylabel("Summed |IG|")
    ax_lag.set_title("Attribution by lag", fontsize=8, pad=3)
    ax_lag.grid(axis="y", color="#D0D0D0", linewidth=0.35, alpha=0.65)

    top_features = np.argsort(feature_total)[::-1][:3]
    for ax, feature_index in zip(channel_axes, top_features):
        image = np.abs(attr[:, feature_index]).sum(axis=0)
        imshow_with_colorbar(fig, ax, image, cmap="magma", title=feature_names[int(feature_index)])

    fig.suptitle(
        f"Integrated Gradients sample {sample_index}: J={diagnostic}, value={objective_value:.3g}",
        fontsize=9,
    )
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_prefix.with_name(output_prefix.name + "_summary").with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_channel_maps(output_prefix: Path, attr: np.ndarray, feature_names: list[str]) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.2), constrained_layout=True)
    for feature_index, ax in enumerate(axes.ravel()):
        image = np.abs(attr[:, feature_index]).sum(axis=0)
        imshow_with_colorbar(fig, ax, image, cmap="magma", title=feature_names[feature_index])
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_prefix.with_name(output_prefix.name + "_channel_maps").with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)



def save_average_summary_plot(
    output_prefix: Path,
    *,
    summary: dict[str, np.ndarray],
    feature_names: list[str],
    temporal_offsets: list[int],
    sample_count: int,
) -> None:
    apply_plot_style()
    feature_total = summary["feature_importance"]
    lag_total = summary["lag_importance"]
    lag_feature = summary["lag_feature_importance"]

    fig = plt.figure(figsize=(7.1, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.1))
    ax_feature = fig.add_subplot(grid[0, :2])
    ax_lag = fig.add_subplot(grid[0, 2])
    ax_spatial = fig.add_subplot(grid[1, 0])
    ax_lag_feature = fig.add_subplot(grid[1, 1:])

    order = np.argsort(feature_total)
    ax_feature.barh(np.arange(len(order)), feature_total[order], color="#0F4D92", height=0.72)
    ax_feature.set_yticks(np.arange(len(order)))
    ax_feature.set_yticklabels([feature_names[index] for index in order])
    ax_feature.set_xlabel("Mean summed |IG| per sample")
    ax_feature.set_title("Attribution by variable", fontsize=8, pad=3)
    ax_feature.grid(axis="x", color="#D0D0D0", linewidth=0.35, alpha=0.65)

    ax_lag.plot(temporal_offsets, lag_total, color="#B64342", marker="o", linewidth=1.0, markersize=3)
    ax_lag.set_xlabel("Temporal offset")
    ax_lag.set_ylabel("Mean summed |IG|")
    ax_lag.set_title("Attribution by lag", fontsize=8, pad=3)
    ax_lag.grid(axis="y", color="#D0D0D0", linewidth=0.35, alpha=0.65)

    imshow_with_colorbar(
        fig, ax_spatial, summary["spatial_importance"], cmap="magma",
        title="Mean spatial |IG|", aspect="equal",
    )
    ax_spatial.set_xticks([0, 20, 39])
    ax_spatial.set_yticks([0, 10, 19])
    ax_spatial.set_xlabel("x index")
    ax_spatial.set_ylabel("y index")
    mesh = ax_lag_feature.imshow(lag_feature, origin="lower", cmap="magma", aspect="auto")
    ax_lag_feature.set_xticks(np.arange(len(feature_names)))
    ax_lag_feature.set_xticklabels(feature_names, rotation=35, ha="right")
    ax_lag_feature.set_yticks(np.arange(len(temporal_offsets)))
    ax_lag_feature.set_yticklabels(temporal_offsets)
    ax_lag_feature.set_xlabel("Input variable")
    ax_lag_feature.set_ylabel("Temporal offset")
    ax_lag_feature.set_title("Mean attribution by lag and variable", fontsize=8, pad=3)
    fig.colorbar(mesh, ax=ax_lag_feature, fraction=0.046, pad=0.02, label="Mean summed |IG|")

    fig.suptitle(f"Test-set mean Integrated Gradients (n={sample_count})", fontsize=9)
    fig.savefig(output_prefix.with_name(output_prefix.name + "_summary").with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_lag_feature_csv(
    path: Path,
    feature_names: list[str],
    temporal_offsets: list[int],
    values: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["temporal_offset", *feature_names])
        for offset, row in zip(temporal_offsets, values):
            writer.writerow([offset, *[float(value) for value in row]])


def run_all_test(
    args: argparse.Namespace,
    *,
    config: dict,
    config_path: Path,
    checkpoint_path: Path,
    model: torch.nn.Module,
    test_subset,
    stats,
    device: torch.device,
) -> None:
    sequence_length = int(config["model"]["sequence_length"])
    frame_channels = int(config["model"]["frame_channels"])
    feature_names = list(args.feature_names)
    if len(feature_names) != frame_channels:
        raise ValueError(f"Expected {frame_channels} feature names, got {len(feature_names)}")
    temporal_offsets = [int(v) for v in config["data"].get("temporal_offsets", range(sequence_length))]
    if len(temporal_offsets) != sequence_length:
        raise ValueError("temporal_offsets length does not match sequence_length")

    test_indices = np.asarray(test_subset.indices, dtype=np.int64)
    selected_positions = np.arange(len(test_indices), dtype=np.int64)
    flux_threshold: float | None = None
    if args.flux_subset != "all":
        if not 0.0 < args.flux_quantile < 0.5:
            raise ValueError("--flux-quantile must be between 0 and 0.5")
        selection_path = resolve_path(args.selection_npz, PROJECT_DIR)
        with np.load(selection_path, allow_pickle=True) as selection:
            source_indices = np.asarray(selection["sample_indices"], dtype=np.int64)
            source_objective = np.asarray(selection["objective_values"], dtype=np.float64)
        if args.flux_subset == "high":
            flux_threshold = float(np.quantile(source_objective, 1.0 - args.flux_quantile))
            selected_indices = source_indices[source_objective >= flux_threshold]
        else:
            flux_threshold = float(np.quantile(source_objective, args.flux_quantile))
            selected_indices = source_indices[source_objective <= flux_threshold]
        selected_positions = np.flatnonzero(np.isin(test_indices, selected_indices))
        if len(selected_positions) != len(selected_indices):
            raise ValueError(
                "Flux-selection NPZ and current test split do not contain identical selected sample indices"
            )
    if args.max_samples is not None:
        selected_positions = selected_positions[: args.max_samples]
    sample_count = len(selected_positions)
    if sample_count < 1:
        raise ValueError("No test samples selected")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    sums: dict[str, np.ndarray] | None = None
    objective_values = np.empty(sample_count, dtype=np.float64)
    processed_indices = np.empty(sample_count, dtype=np.int64)
    sample_feature_importance = np.empty((sample_count, frame_channels), dtype=np.float64)
    sample_signed_feature_importance = np.empty((sample_count, frame_channels), dtype=np.float64)
    normalize_y = bool(config["data"].get("normalize_y", True))
    print(
        f"Running {args.flux_subset}-flux mean Integrated Gradients over "
        f"{sample_count}/{len(test_indices)} held-out test samples; "
        f"diagnostic={args.diagnostic}, steps={args.steps}, baseline={args.baseline}",
        flush=True,
    )

    for position in range(sample_count):
        test_position = int(selected_positions[position])
        x_cpu, _ = test_subset[test_position]
        x = x_cpu.unsqueeze(0).to(device)
        baseline = make_baseline(x, args.baseline)
        attribution, objective_value, _ = integrated_gradients(
            model, x, baseline, stats=stats, normalize_y=normalize_y, steps=args.steps,
            diagnostic=args.diagnostic, pixel_y=args.pixel_y, pixel_x=args.pixel_x, verbose=False,
        )
        attr_np = reshape_attribution(attribution.cpu().numpy(), sequence_length, frame_channels)
        sample_summary = summarize_attribution(attr_np)
        if sums is None:
            sums = {key: np.zeros_like(value, dtype=np.float64) for key, value in sample_summary.items()}
        for key, value in sample_summary.items():
            sums[key] += value
        objective_values[position] = objective_value
        processed_indices[position] = test_indices[test_position]
        sample_feature_importance[position] = sample_summary["feature_importance"]
        sample_signed_feature_importance[position] = sample_summary["signed_feature_importance"]
        if (position + 1) % max(1, args.progress_every) == 0 or position + 1 == sample_count:
            print(f"  processed {position + 1}/{sample_count} test samples", flush=True)

    assert sums is not None
    mean_summary = {key: value / sample_count for key, value in sums.items()}
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_prefix = "all_test" if args.flux_subset == "all" else f"{args.flux_subset}_flux_test"
    output_prefix = output_dir / (
        f"{subset_prefix}_mean_ig_{args.diagnostic}_{args.baseline}_steps{args.steps}"
    )

    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        sample_indices=processed_indices,
        sample_count=np.asarray(sample_count, dtype=np.int64),
        objective_values=objective_values,
        sample_feature_importance=sample_feature_importance,
        sample_signed_feature_importance=sample_signed_feature_importance,
        feature_importance=mean_summary["feature_importance"],
        lag_importance=mean_summary["lag_importance"],
        spatial_importance=mean_summary["spatial_importance"].astype(np.float32),
        lag_feature_importance=mean_summary["lag_feature_importance"],
        feature_spatial_importance=mean_summary["feature_spatial_importance"].astype(np.float32),
        signed_feature_importance=mean_summary["signed_feature_importance"],
        signed_lag_importance=mean_summary["signed_lag_importance"],
        signed_lag_feature_importance=mean_summary["signed_lag_feature_importance"],
        feature_names=np.asarray(feature_names),
        temporal_offsets=np.asarray(temporal_offsets, dtype=np.int64),
        diagnostic=np.asarray(args.diagnostic),
        checkpoint=np.asarray(str(checkpoint_path)),
        config=np.asarray(str(config_path)),
    )
    write_csvs(output_prefix, feature_names, temporal_offsets, mean_summary)
    write_lag_feature_csv(
        output_prefix.with_name(output_prefix.name + "_lag_feature_importance.csv"),
        feature_names, temporal_offsets, mean_summary["lag_feature_importance"],
    )
    with output_prefix.with_name(output_prefix.name + "_signed_feature_importance.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "feature_name", "mean_signed_attribution"])
        for feature_index, feature_name in enumerate(feature_names):
            writer.writerow(
                [feature_index, feature_name, float(mean_summary["signed_feature_importance"][feature_index])]
            )
    save_average_summary_plot(
        output_prefix, summary=mean_summary, feature_names=feature_names,
        temporal_offsets=temporal_offsets, sample_count=sample_count,
    )
    metadata = {
        "mode": f"{args.flux_subset}_flux_test_mean",
        "sample_count": sample_count,
        "available_test_samples": len(test_indices),
        "flux_subset": args.flux_subset,
        "flux_quantile": args.flux_quantile,
        "flux_threshold": flux_threshold,
        "diagnostic": args.diagnostic,
        "objective_mean": float(np.mean(objective_values)),
        "objective_std": float(np.std(objective_values)),
        "baseline": args.baseline,
        "steps": args.steps,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "outputs_prefix": str(output_prefix),
    }
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved test-set mean Integrated Gradients outputs with prefix: {output_prefix}", flush=True)

def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_DIR)
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(args.device or config.get("device", "auto"))

    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False

    _, _, test_loader, stats = build_dataloaders(config, PROJECT_DIR, int(config.get("seed", 42)))
    model = build_model(config["model"]).to(device)
    checkpoint_path = resolve_path(args.checkpoint or Path(config["train"]["best_checkpoint"]), PROJECT_DIR)
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.all_test:
        run_all_test(
            args, config=config, config_path=config_path, checkpoint_path=checkpoint_path,
            model=model, test_subset=test_loader.dataset, stats=stats, device=device,
        )
        return

    sample_index, x_cpu, y_cpu = select_test_sample(
        test_loader.dataset, args.sample_index, args.test_position
    )
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    baseline = make_baseline(x, args.baseline)
    print(
        f"Running Integrated Gradients for sample_index={sample_index}, "
        f"diagnostic={args.diagnostic}, steps={args.steps}, baseline={args.baseline}",
        flush=True,
    )
    attribution, objective_value, prediction = integrated_gradients(
        model,
        x,
        baseline,
        stats=stats,
        normalize_y=bool(config["data"].get("normalize_y", True)),
        steps=args.steps,
        diagnostic=args.diagnostic,
        pixel_y=args.pixel_y,
        pixel_x=args.pixel_x,
    )

    target = denormalize_y(y, stats, bool(config["data"].get("normalize_y", True))).detach()
    sequence_length = int(config["model"]["sequence_length"])
    frame_channels = int(config["model"]["frame_channels"])
    feature_names = list(args.feature_names)
    if len(feature_names) != frame_channels:
        raise ValueError(f"Expected {frame_channels} feature names, got {len(feature_names)}")

    attr_np = reshape_attribution(attribution.cpu().numpy(), sequence_length, frame_channels)
    summary = summarize_attribution(attr_np)
    temporal_offsets = [int(v) for v in config["data"].get("temporal_offsets", list(range(sequence_length)))]
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / f"sample_{sample_index}_ig_{args.diagnostic}_{args.baseline}_steps{args.steps}"

    np.savez_compressed(
        output_prefix.with_suffix(".npz"),
        sample_index=np.asarray(sample_index, dtype=np.int64),
        attribution=attr_np.astype(np.float32),
        prediction=prediction.cpu().numpy().astype(np.float32),
        target=target.cpu().numpy().astype(np.float32),
        input=x_cpu.numpy().astype(np.float32),
        baseline=baseline.detach().cpu().numpy().astype(np.float32),
        feature_importance=summary["feature_importance"].astype(np.float64),
        lag_importance=summary["lag_importance"].astype(np.float64),
        signed_feature_importance=summary["signed_feature_importance"].astype(np.float64),
        signed_lag_importance=summary["signed_lag_importance"].astype(np.float64),
        spatial_importance=summary["spatial_importance"].astype(np.float32),
        feature_names=np.asarray(feature_names),
        temporal_offsets=np.asarray(temporal_offsets, dtype=np.int64),
        diagnostic=np.asarray(args.diagnostic),
        objective_value=np.asarray(objective_value, dtype=np.float64),
        checkpoint=np.asarray(str(checkpoint_path)),
        config=np.asarray(str(config_path)),
    )
    write_csvs(output_prefix, feature_names, temporal_offsets, summary)
    save_summary_plot(
        output_prefix,
        prediction=prediction.cpu().numpy(),
        target=target.cpu().numpy(),
        attr=attr_np,
        feature_names=feature_names,
        temporal_offsets=temporal_offsets,
        summary=summary,
        diagnostic=args.diagnostic,
        objective_value=objective_value,
        sample_index=sample_index,
    )
    save_channel_maps(output_prefix, attr_np, feature_names)

    metadata = {
        "sample_index": sample_index,
        "diagnostic": args.diagnostic,
        "objective_value": objective_value,
        "baseline": args.baseline,
        "steps": args.steps,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "outputs_prefix": str(output_prefix),
    }
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved Integrated Gradients outputs with prefix: {output_prefix}", flush=True)


if __name__ == "__main__":
    main()
