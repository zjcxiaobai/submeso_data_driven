#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data_utils import build_dataloaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402


DEFAULT_FEATURE_NAMES = [
    "U velocity",
    "V velocity",
    "W velocity",
    "Vorticity",
    "Strain",
    "Potential temperature",
    "Density",
    "KPP boundary layer depth",
]


def resolve_path(path: Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def denormalize_y(values: torch.Tensor, stats, normalize_y: bool) -> torch.Tensor:
    if not normalize_y:
        return values
    mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=values.device)
    return values * std + mean


def make_baseline(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)
    if mode == "sample_mean":
        return x.mean(dim=(-2, -1), keepdim=True).expand_as(x).clone()
    raise ValueError(f"Unknown baseline mode: {mode}")


def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    stats,
    normalize_y: bool,
    steps: int,
    diagnostic: str,
    pixel_y: int | None = None,
    pixel_x: int | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    if diagnostic != "pixel":
        raise ValueError("This grid-cell script currently supports diagnostic='pixel' only.")
    if pixel_y is None or pixel_x is None:
        raise ValueError("pixel_y and pixel_x are required for pixel Integrated Gradients.")

    delta = x - baseline
    total_grad = torch.zeros_like(x)
    prediction = None
    objective_value = float("nan")
    for step_number in range(1, steps + 1):
        alpha = float(step_number) / float(steps)
        scaled = (baseline + alpha * delta).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        pred_norm = model(scaled)
        pred = denormalize_y(pred_norm, stats, normalize_y)
        objective = pred[:, 0, int(pixel_y), int(pixel_x)].sum()
        objective.backward()
        if scaled.grad is None:
            raise RuntimeError("Integrated Gradients gradient is None")
        total_grad += scaled.grad.detach()
        if step_number == steps:
            prediction = pred.detach()
            objective_value = float(objective.detach().cpu().item())
        if verbose and (step_number == 1 or step_number == steps or step_number % max(1, steps // 4) == 0):
            print(f"  IG step {step_number}/{steps}", flush=True)

    attribution = delta * total_grad / float(steps)
    if prediction is None:
        with torch.no_grad():
            prediction = denormalize_y(model(x), stats, normalize_y)
    return attribution.detach(), objective_value, prediction.detach()


def reshape_attribution(attribution: np.ndarray, sequence_length: int, frame_channels: int) -> np.ndarray:
    if attribution.ndim != 4 or attribution.shape[0] != 1:
        raise ValueError(f"Expected attribution shape (1, C, H, W), got {attribution.shape}")
    channels = attribution.shape[1]
    if channels != sequence_length * frame_channels:
        raise ValueError(
            f"Attribution channel count {channels} does not match sequence_length*frame_channels "
            f"{sequence_length * frame_channels}"
        )
    return attribution[0].reshape(sequence_length, frame_channels, attribution.shape[2], attribution.shape[3])


DEFAULT_CONFIG = HERE / "config_model_analysis.yaml"
DEFAULT_SELECTION = PROJECT_DIR / "analysis/model_analysis/outputs/test_predictions_denorm.npz"
DEFAULT_OUTPUT_DIR = HERE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-cell Integrated Gradients for temporally distributed top-flux test snapshots, "
            "including geographical, target-centered, lag, distance, and signed diagnostics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--selection-npz", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--baseline", choices=["zero", "sample_mean"], default="zero")
    parser.add_argument("--flux-quantile", type=float, default=0.10)
    parser.add_argument("--num-snapshots", type=int, default=90)
    parser.add_argument("--dx-km", type=float, default=50.0)
    parser.add_argument("--dy-km", type=float, default=50.0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--feature-names", nargs="+", default=DEFAULT_FEATURE_NAMES)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def select_snapshots(
    selection_path: Path,
    test_indices: np.ndarray,
    flux_quantile: float,
    num_snapshots: int,
) -> tuple[np.ndarray, float]:
    if not 0.0 < flux_quantile < 0.5:
        raise ValueError("--flux-quantile must be between 0 and 0.5")
    with np.load(selection_path, allow_pickle=True) as selection:
        if {"sample_indices", "objective_values"}.issubset(selection.files):
            indices = np.asarray(selection["sample_indices"], dtype=np.int64)
            objective = np.asarray(selection["objective_values"], dtype=np.float64)
        elif {"sample_index", "prediction"}.issubset(selection.files):
            indices = np.asarray(selection["sample_index"], dtype=np.int64)
            prediction = np.asarray(selection["prediction"], dtype=np.float64)
            objective = np.nanmean(np.abs(prediction), axis=(1, 2, 3))
        else:
            raise ValueError(
                f"{selection_path} must contain either sample_indices/objective_values "
                "or sample_index/prediction arrays"
            )
    keep = np.isin(indices, test_indices)
    indices = indices[keep]
    objective = objective[keep]
    threshold = float(np.quantile(objective, 1.0 - flux_quantile))
    candidates = np.sort(indices[objective >= threshold])
    if len(candidates) < num_snapshots:
        raise ValueError(
            f"Only {len(candidates)} top-flux candidates are available; "
            f"cannot select {num_snapshots} snapshots"
        )
    positions = np.rint(np.linspace(0, len(candidates) - 1, num_snapshots)).astype(np.int64)
    selected = np.unique(candidates[positions])
    if len(selected) != num_snapshots:
        raise RuntimeError("Evenly spaced selection produced duplicate snapshots")
    return selected, threshold


def target_cells(field: np.ndarray) -> list[tuple[str, int, int, float]]:
    height, width = field.shape
    choices = [
        ("strong_positive", int(np.nanargmax(field))),
        ("strong_negative", int(np.nanargmin(field))),
        ("weak", int(np.nanargmin(np.abs(field)))),
    ]
    return [
        (name, flat_index // width, flat_index % width, float(field.ravel()[flat_index]))
        for name, flat_index in choices
    ]


def recenter_map(image: np.ndarray, target_y: int, target_x: int) -> np.ndarray:
    height, width = image.shape
    center_y, center_x = height // 2, width // 2
    rolled = np.roll(image, center_x - target_x, axis=1)
    centered = np.full_like(rolled, np.nan, dtype=np.float64)
    shift_y = center_y - target_y
    source_start = max(0, -shift_y)
    source_stop = min(height, height - shift_y)
    destination_start = source_start + shift_y
    destination_stop = source_stop + shift_y
    centered[destination_start:destination_stop] = rolled[source_start:source_stop]
    return centered


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def map_grid(
    values: np.ndarray,
    feature_names: list[str],
    output: Path,
    *,
    title: str,
    cmap: str,
    signed: bool = False,
    extent: tuple[float, float, float, float] | None = None,
    xlabel: str = "x index",
    ylabel: str = "y index",
) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), constrained_layout=True)
    finite = np.abs(values[np.isfinite(values)])
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    vmin = -vmax if signed else 0.0
    for feature_index, ax in enumerate(axes.ravel()):
        mesh = ax.imshow(
            values[feature_index],
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(feature_names[feature_index], fontsize=7, pad=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title, fontsize=9)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_lag_diagnostics(
    lag_feature: np.ndarray,
    temporal_offsets: np.ndarray,
    feature_names: list[str],
    output: Path,
) -> None:
    apply_style()
    normalized = safe_divide(lag_feature, lag_feature.sum(axis=0, keepdims=True))
    total_fraction = lag_feature.sum(axis=1) / lag_feature.sum()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    axes[0].plot(temporal_offsets, total_fraction, color="#1F5A85", marker="o", linewidth=1.2)
    axes[0].set_xlabel("Temporal offset")
    axes[0].set_ylabel("Fraction of total |IG|")
    axes[0].set_title("Total attribution by lag", fontsize=8)
    axes[0].grid(axis="y", color="#D0D0D0", linewidth=0.4)
    raw_mesh = axes[1].imshow(lag_feature, origin="lower", aspect="auto", cmap="magma")
    norm_mesh = axes[2].imshow(
        normalized,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=float(np.nanpercentile(normalized, 99)),
    )
    for ax, panel_title in zip(axes[1:], ["Mean |IG|", "Within-feature lag fraction"]):
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(temporal_offsets)))
        ax.set_yticklabels(temporal_offsets)
        ax.set_xlabel("Input feature")
        ax.set_ylabel("Temporal offset")
        ax.set_title(panel_title, fontsize=8)
    fig.colorbar(raw_mesh, ax=axes[1], fraction=0.046, pad=0.02)
    fig.colorbar(norm_mesh, ax=axes[2], fraction=0.046, pad=0.02)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def distance_diagnostics(
    centered: np.ndarray,
    dx_km: float,
    dy_km: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    _, height, width = centered.shape
    delta_y = np.arange(height) - height // 2
    delta_x = np.arange(width) - width // 2
    distance = np.sqrt(
        (delta_y[:, None] * dy_km) ** 2 + (delta_x[None, :] * dx_km) ** 2
    )
    curves = []
    r50 = np.full(centered.shape[0], np.nan)
    r90 = np.full(centered.shape[0], np.nan)
    for feature_index, image in enumerate(centered):
        valid = np.isfinite(image) & (image >= 0)
        radii = distance[valid]
        weights = image[valid]
        order = np.argsort(radii)
        radii = radii[order]
        weights = weights[order]
        cumulative = np.cumsum(weights)
        cumulative = cumulative / cumulative[-1]
        curves.append((radii, cumulative))
        r50[feature_index] = radii[np.searchsorted(cumulative, 0.5)]
        r90[feature_index] = radii[np.searchsorted(cumulative, 0.9)]
    return curves, r50, r90


def plot_distance(
    curves: list[tuple[np.ndarray, np.ndarray]],
    r50: np.ndarray,
    r90: np.ndarray,
    feature_names: list[str],
    output: Path,
) -> None:
    apply_style()
    colors = plt.get_cmap("tab10")(np.arange(len(feature_names)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for feature_index, (radii, cumulative) in enumerate(curves):
        axes[0].plot(
            radii,
            cumulative,
            color=colors[feature_index],
            linewidth=1.1,
            label=feature_names[feature_index],
        )
    axes[0].axhline(0.5, color="#888888", linewidth=0.7, linestyle="--")
    axes[0].axhline(0.9, color="#888888", linewidth=0.7, linestyle="--")
    axes[0].set_xlabel("Periodic target distance (km)")
    axes[0].set_ylabel("Cumulative attribution")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(color="#D0D0D0", linewidth=0.35)
    axes[0].legend(fontsize=6, ncol=2)
    y = np.arange(len(feature_names))
    axes[1].barh(y - 0.18, r50, height=0.34, color="#6B8EAD", label=r"$r_{50}$")
    axes[1].barh(y + 0.18, r90, height=0.34, color="#C75B4A", label=r"$r_{90}$")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(feature_names)
    axes[1].set_xlabel("Distance (km)")
    axes[1].grid(axis="x", color="#D0D0D0", linewidth=0.35)
    axes[1].legend()
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
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
    test_subset = test_loader.dataset
    test_indices = np.asarray(test_subset.indices, dtype=np.int64)
    selected_indices, flux_threshold = select_snapshots(
        resolve_path(args.selection_npz, PROJECT_DIR),
        test_indices,
        args.flux_quantile,
        args.num_snapshots,
    )
    index_to_position = {int(index): position for position, index in enumerate(test_indices)}

    model = build_model(config["model"]).to(device)
    checkpoint_path = resolve_path(
        args.checkpoint or Path(config["train"]["best_checkpoint"]), PROJECT_DIR
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    sequence_length = int(config["model"]["sequence_length"])
    frame_channels = int(config["model"]["frame_channels"])
    feature_names = list(args.feature_names)
    if len(feature_names) != frame_channels:
        raise ValueError(f"Expected {frame_channels} feature names, got {len(feature_names)}")
    temporal_offsets = np.asarray(
        config["data"].get("temporal_offsets", range(sequence_length)), dtype=np.int64
    )
    normalize_y = bool(config["data"].get("normalize_y", True))
    first_x, _ = test_subset[index_to_position[int(selected_indices[0])]]
    height, width = first_x.shape[-2:]

    geographic_sum = np.zeros((frame_channels, height, width), dtype=np.float64)
    centered_sum = np.zeros_like(geographic_sum)
    centered_count = np.zeros_like(geographic_sum)
    lag_feature_sum = np.zeros((sequence_length, frame_channels), dtype=np.float64)
    signed_target_sum = np.zeros_like(geographic_sum)
    signed_target_count = np.zeros((height, width), dtype=np.int64)
    target_records: list[dict[str, object]] = []
    target_total = args.num_snapshots * 3

    print(
        f"Selected {args.num_snapshots} temporally distributed snapshots from the "
        f"top {args.flux_quantile:.0%}; threshold={flux_threshold:.6g}; "
        f"targets={target_total}",
        flush=True,
    )
    target_number = 0
    for snapshot_number, sample_index in enumerate(selected_indices, start=1):
        position = index_to_position[int(sample_index)]
        x_cpu, _ = test_subset[position]
        x = x_cpu.unsqueeze(0).to(device)
        baseline = make_baseline(x, args.baseline)
        with torch.no_grad():
            prediction = denormalize_y(model(x), stats, normalize_y)
        field = prediction[0, 0].detach().cpu().numpy()
        for target_type, target_y, target_x, target_flux in target_cells(field):
            target_number += 1
            attribution, _, _ = integrated_gradients(
                model,
                x,
                baseline,
                stats=stats,
                normalize_y=normalize_y,
                steps=args.steps,
                diagnostic="pixel",
                pixel_y=target_y,
                pixel_x=target_x,
                verbose=False,
            )
            attr = reshape_attribution(
                attribution.cpu().numpy(), sequence_length, frame_channels
            )
            abs_attr = np.abs(attr)
            feature_maps = abs_attr.sum(axis=0)
            geographic_sum += feature_maps
            lag_feature_sum += abs_attr.sum(axis=(2, 3))
            signed_feature = attr.sum(axis=(0, 2, 3))
            signed_target_sum[:, target_y, target_x] += signed_feature
            signed_target_count[target_y, target_x] += 1
            for feature_index in range(frame_channels):
                centered = recenter_map(
                    feature_maps[feature_index], target_y, target_x
                )
                finite = np.isfinite(centered)
                centered_sum[feature_index][finite] += centered[finite]
                centered_count[feature_index][finite] += 1
            target_records.append(
                {
                    "sample_index": int(sample_index),
                    "snapshot_rank": snapshot_number,
                    "target_type": target_type,
                    "target_y": target_y,
                    "target_x": target_x,
                    "predicted_signed_flux": target_flux,
                }
            )
        if snapshot_number % max(1, args.progress_every) == 0:
            print(
                f"  processed {snapshot_number}/{args.num_snapshots} snapshots "
                f"({target_number}/{target_total} targets)",
                flush=True,
            )

    geographic = geographic_sum / target_total
    centered = safe_divide(centered_sum, centered_count)
    lag_feature = lag_feature_sum / target_total
    signed_maps = safe_divide(
        signed_target_sum,
        signed_target_count[None, :, :],
    )
    curves, r50, r90 = distance_diagnostics(centered, args.dx_km, args.dy_km)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "top10_gridcell_ig"
    x_half = width // 2
    y_half = height // 2
    centered_extent = (
        -x_half * args.dx_km,
        (width - x_half) * args.dx_km,
        -y_half * args.dy_km,
        (height - y_half) * args.dy_km,
    )
    map_grid(
        geographic,
        feature_names,
        prefix.with_name(prefix.name + "_geographical_feature_maps.png"),
        title="Geographical feature attribution: top-10% flux snapshots",
        cmap="magma",
    )
    map_grid(
        centered,
        feature_names,
        prefix.with_name(prefix.name + "_target_centered_feature_maps.png"),
        title="Target-centered feature attribution",
        cmap="magma",
        extent=centered_extent,
        xlabel=r"$\Delta x$ (km)",
        ylabel=r"$\Delta y$ (km)",
    )
    plot_lag_diagnostics(
        lag_feature,
        temporal_offsets,
        feature_names,
        prefix.with_name(prefix.name + "_time_lag_sensitivity.png"),
    )
    plot_distance(
        curves,
        r50,
        r90,
        feature_names,
        prefix.with_name(prefix.name + "_distance_sensitivity.png"),
    )
    map_grid(
        signed_maps,
        feature_names,
        prefix.with_name(prefix.name + "_signed_contribution_maps.png"),
        title="Signed contribution at explained target cells",
        cmap="RdBu_r",
        signed=True,
    )

    np.savez_compressed(
        prefix.with_suffix(".npz"),
        selected_sample_indices=selected_indices,
        flux_threshold=np.asarray(flux_threshold),
        geographic_feature_attribution=geographic.astype(np.float32),
        target_centered_feature_attribution=centered.astype(np.float32),
        lag_feature_attribution=lag_feature,
        signed_target_feature_attribution=signed_maps.astype(np.float32),
        signed_target_count=signed_target_count,
        r50_km=r50,
        r90_km=r90,
        feature_names=np.asarray(feature_names),
        temporal_offsets=temporal_offsets,
        dx_km=np.asarray(args.dx_km),
        dy_km=np.asarray(args.dy_km),
    )
    with prefix.with_name(prefix.name + "_targets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(target_records[0]))
        writer.writeheader()
        writer.writerows(target_records)
    with prefix.with_name(prefix.name + "_radii.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "feature_name", "r50_km", "r90_km"])
        for feature_index, feature_name in enumerate(feature_names):
            writer.writerow(
                [feature_index, feature_name, r50[feature_index], r90[feature_index]]
            )
    metadata = {
        "selection": "temporally distributed snapshots from top predicted mean-absolute-flux decile",
        "flux_quantile": args.flux_quantile,
        "flux_threshold": flux_threshold,
        "num_snapshots": args.num_snapshots,
        "targets_per_snapshot": ["strong_positive", "strong_negative", "weak"],
        "target_count": target_total,
        "steps": args.steps,
        "baseline": args.baseline,
        "dx_km": args.dx_km,
        "dy_km": args.dy_km,
        "periodic_x": True,
        "periodic_y": False,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "output_prefix": str(prefix),
    }
    prefix.with_name(prefix.name + "_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved grid-cell IG outputs with prefix: {prefix}", flush=True)


if __name__ == "__main__":
    main()
