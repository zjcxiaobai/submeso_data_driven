#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
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


PALETTE = {
    "normal": "#B64342",
    "ablated": "#0F4D92",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}

CASES = {
    "no_shuffle": PROJECT_DIR / "configs/unet_no_shuffle.yaml",
    "shuffle_train_val": PROJECT_DIR / "configs/unet_shuffle_train_val.yaml",
    "shuffle_all": PROJECT_DIR / "configs/unet_shuffle_all.yaml",
}

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


@dataclass
class CaseResult:
    case: str
    sample_index: np.ndarray
    baseline_r2: np.ndarray
    feature_r2: np.ndarray
    feature_drop: np.ndarray
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feature-importance analysis by replacing each input feature with its training mean "
            "and measuring the R2 drop on the test set using trained checkpoints only."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        default=["no_shuffle", "shuffle_train_val", "shuffle_all"],
        help="Model cases to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "analysis/feature_importance",
        help="Directory for feature-importance outputs.",
    )
    parser.add_argument(
        "--feature-names",
        nargs="+",
        default=None,
        help="Optional names for the input channels. Must match model in_channels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even when a case result .npz already exists.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch-size override. By default, use the YAML config value.",
    )
    return parser.parse_args()


def apply_publication_style() -> None:
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


def sample_r2_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = pred - target
    reduce_dims = tuple(range(1, target.ndim))
    sse = torch.sum(residual * residual, dim=reduce_dims)
    target_mean = torch.mean(target, dim=reduce_dims, keepdim=True)
    sst = torch.sum((target - target_mean) ** 2, dim=reduce_dims)
    r2 = torch.full_like(sse, float("nan"), dtype=torch.float64)
    valid = sst > 0.0
    r2[valid] = 1.0 - sse[valid].double() / sst[valid].double()
    return r2


def replace_feature_with_training_mean(
    x: torch.Tensor,
    feature_index: int,
    *,
    stats,
    normalize_x: bool,
    device: torch.device,
) -> torch.Tensor:
    x_ablated = x.clone()
    if normalize_x:
        replacement = torch.tensor(0.0, dtype=x.dtype, device=device)
    else:
        replacement = torch.as_tensor(stats.x_mean[feature_index], dtype=x.dtype, device=device)
    x_ablated[:, feature_index, :, :] = replacement
    return x_ablated


def run_case(
    case: str,
    config_path: Path,
    feature_names: list[str] | None,
    output_dir: Path,
    device_override: str | None,
    batch_size: int,
) -> CaseResult:
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(device_override or config.get("device", "auto"))
    checkpoint_path = Path(config["train"]["best_checkpoint"])
    if batch_size is not None:
        config["data"]["batch_size"] = batch_size

    _, _, test_loader, stats = build_dataloaders(config, PROJECT_DIR, int(config.get("seed", 42)))
    n_features = int(config["model"]["in_channels"])
    names = feature_names or DEFAULT_FEATURE_NAMES[:n_features]
    if len(names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, got {len(names)}")

    model = build_model(config["model"]).to(device)
    print(f"Loading trained checkpoint for {format_case_name(case)}: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalize_x = bool(config["data"].get("normalize_x", True))
    sample_indices: list[np.ndarray] = []
    baseline_batches: list[np.ndarray] = []
    feature_batches: list[list[np.ndarray]] = [[] for _ in range(n_features)]

    offset = 0
    dataset_indices = np.asarray(test_loader.dataset.indices, dtype=np.int64)
    effective_batch_size = int(config["data"].get("batch_size", 1))
    print(
        f"Running inference-only feature ablation for {format_case_name(case)} "
        f"on {len(dataset_indices)} test samples with batch_size={effective_batch_size}...",
        flush=True,
    )
    with torch.no_grad():
        for batch_number, (x, y) in enumerate(test_loader, start=1):
            batch_size = x.size(0)
            sample_indices.append(dataset_indices[offset : offset + batch_size])
            offset += batch_size

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            baseline_pred = model(x)
            baseline_batches.append(sample_r2_torch(baseline_pred, y).cpu().numpy())

            for feature_index in range(n_features):
                x_ablated = replace_feature_with_training_mean(
                    x,
                    feature_index,
                    stats=stats,
                    normalize_x=normalize_x,
                    device=device,
                )
                ablated_pred = model(x_ablated)
                feature_batches[feature_index].append(sample_r2_torch(ablated_pred, y).cpu().numpy())
            if batch_number % 25 == 0 or offset == len(dataset_indices):
                print(f"  {format_case_name(case)}: processed {offset}/{len(dataset_indices)} samples", flush=True)

    sample_index = np.concatenate(sample_indices)
    baseline_r2 = np.concatenate(baseline_batches)
    feature_r2 = np.vstack([np.concatenate(batches) for batches in feature_batches])
    order = np.argsort(sample_index)
    sample_index = sample_index[order]
    baseline_r2 = baseline_r2[order]
    feature_r2 = feature_r2[:, order]
    feature_drop = np.nanmean(baseline_r2[None, :] - feature_r2, axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{case}_feature_importance.npz",
        sample_index=sample_index,
        baseline_r2=baseline_r2,
        feature_r2=feature_r2,
        feature_drop=feature_drop,
        feature_names=np.asarray(names),
    )
    write_case_csv(output_dir / f"{case}_feature_importance.csv", names, feature_drop)
    return CaseResult(case, sample_index, baseline_r2, feature_r2, feature_drop, names)


def load_case_result(case: str, path: Path) -> CaseResult:
    data = np.load(path, allow_pickle=True)
    return CaseResult(
        case=case,
        sample_index=data["sample_index"],
        baseline_r2=data["baseline_r2"],
        feature_r2=data["feature_r2"],
        feature_drop=data["feature_drop"],
        feature_names=[str(name) for name in data["feature_names"]],
    )


def write_case_csv(path: Path, feature_names: list[str], feature_drop: np.ndarray) -> None:
    order = np.argsort(feature_drop)[::-1]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "feature_index", "feature_name", "mean_r2_drop"])
        writer.writeheader()
        for rank, feature_index in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "feature_name": feature_names[int(feature_index)],
                    "mean_r2_drop": float(feature_drop[int(feature_index)]),
                }
            )


def plot_case_timeseries(result: CaseResult, output_dir: Path) -> None:
    apply_publication_style()
    n_features = len(result.feature_names)
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 6.6), sharex=True, constrained_layout=True)
    axes_flat = axes.ravel()
    sorted_feature_indices = np.argsort(result.feature_drop)[::-1]

    for rank, (feature_index, ax) in enumerate(zip(sorted_feature_indices, axes_flat[:n_features]), start=1):
        ax.plot(
            result.sample_index,
            result.baseline_r2,
            color=PALETTE["normal"],
            linewidth=0.9,
            label="Normal",
        )
        ax.plot(
            result.sample_index,
            result.feature_r2[feature_index],
            color=PALETTE["ablated"],
            linewidth=0.9,
            label="Feature replaced",
        )
        ax.set_title(
            f"{rank}. {result.feature_names[feature_index]}: mean R2 drop={result.feature_drop[feature_index]:.3f}",
            fontsize=7,
            pad=3,
        )
        ax.axhline(0.0, color=PALETTE["neutral_light"], linewidth=0.6, zorder=0)
        ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
        ax.tick_params(labelsize=6, colors=PALETTE["neutral_dark"])

    for ax in axes[-1, :]:
        ax.set_xlabel("Test sample index")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$R^2$")
    axes_flat[0].legend(loc="best", fontsize=6, handlelength=2.0)
    fig.suptitle(f"{format_case_name(result.case)} feature ablation R2 time series", fontsize=9)
    save_figure(fig, output_dir / f"{result.case}_feature_ablation_timeseries")


def plot_sorted_importance(results: list[CaseResult], output_dir: Path) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, len(results), figsize=(7.1, 2.4), sharey=True, constrained_layout=True)
    if len(results) == 1:
        axes = np.asarray([axes])

    for ax, result in zip(axes, results):
        order = np.argsort(result.feature_drop)
        y = np.arange(len(order))
        ax.barh(y, result.feature_drop[order], color=PALETTE["ablated"], height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([result.feature_names[index] for index in order])
        ax.set_xlabel(r"Mean $R^2$ drop")
        ax.set_title(format_case_name(result.case), fontsize=8)
        ax.axvline(0.0, color=PALETTE["neutral_mid"], linewidth=0.7)
        ax.grid(axis="x", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.55)
        ax.tick_params(labelsize=6, colors=PALETTE["neutral_dark"])

    axes[0].set_ylabel("Feature")
    fig.suptitle("Sorted feature importance from training-mean replacement", fontsize=9)
    save_figure(fig, output_dir / "feature_importance_sorted")


def save_figure(fig, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def format_case_name(case: str) -> str:
    return case.replace("_", " ").title().replace("Train Val", "Train/Val")


def main() -> None:
    args = parse_args()
    results: list[CaseResult] = []
    for case in args.cases:
        result_path = args.output_dir / f"{case}_feature_importance.npz"
        if result_path.exists() and not args.overwrite:
            result = load_case_result(case, result_path)
        else:
            result = run_case(case, CASES[case], args.feature_names, args.output_dir, args.device, args.batch_size)
        results.append(result)
        plot_case_timeseries(result, args.output_dir)

    plot_sorted_importance(results, args.output_dir)
    for result in results:
        print(f"{format_case_name(result.case)}:", flush=True)
        for feature_index in np.argsort(result.feature_drop)[::-1]:
            print(
                f"  {result.feature_names[int(feature_index)]}: mean R2 drop={result.feature_drop[int(feature_index)]:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
