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

PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils import build_dataloaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402


DEFAULT_CONFIG = PROJECT_DIR / "analysis/model_analysis/outputs/config.yaml"
HERE = Path(__file__).resolve().parent
DEFAULT_PHYSICS = HERE / "physics_variables_top90/physics_variables.npz"
DEFAULT_OUTPUT_DIR = HERE / "last_decoder_pca"
DEFAULT_FEATURE_CACHE = DEFAULT_OUTPUT_DIR / "last_decoder_all_channel_feature_maps.npz"
DEFAULT_CHECKPOINT = PROJECT_DIR / "analysis/model_analysis/outputs/checkpoints/best.pt"

PHYSICS_VARIABLES = [
    ("divergence", "Divergence"),
    ("dTdx", r"$\partial T/\partial x$"),
    ("dTdy", r"$\partial T/\partial y$"),
    ("frontogenesis_FT", r"Frontogenesis $F_T$"),
    ("gradT", r"$|\nabla T|$"),
    ("KPPhbl", "KPPHBL"),
    ("Rossby_number", r"$Ro$"),
    ("shear_strain", "Shear strain"),
    ("rho", "Density"),
    ("strain", "Strain"),
    ("T", r"$T$"),
    ("U", r"$U$"),
    ("V", r"$V$"),
    ("W", r"$W$"),
    ("strain_times_gradT", r"Strain $\times |\nabla T|$"),
]


class TensorStore:
    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def hook(self, _module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            raise TypeError(f"Expected tensor output from model.up1, got {type(output)!r}")
        self.tensor = output.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA across the channel dimension of the final decoder feature maps "
            "and correlate the first PCs with selected physics diagnostics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-index", type=int, default=None, help="Optional single sample. If omitted, read sample indices from --physics.")
    parser.add_argument("--n-pcs", type=int, default=8)
    parser.add_argument(
        "--reuse-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the all-channel decoder feature cache when it exists.",
    )
    return parser.parse_args()


def resolve_path(path: Path, root: Path = PROJECT_DIR) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


def resolve_workflow_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else HERE / path


def sample_indices_from_physics(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing physics file with selected sample_index values: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "sample_index" not in data.files:
            raise ValueError(f"{path} is missing sample_index")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64).reshape(-1)
    if sample_index.size == 0:
        raise ValueError(f"No sample_index values found in {path}")
    return sample_index


def select_samples(test_subset, sample_indices: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    test_indices = np.asarray(test_subset.indices, dtype=np.int64)
    position_lookup = {int(sample): pos for pos, sample in enumerate(test_indices)}
    positions: list[int] = []
    tensors: list[torch.Tensor] = []
    missing: list[int] = []
    for sample_index in sample_indices:
        position = position_lookup.get(int(sample_index))
        if position is None:
            missing.append(int(sample_index))
            continue
        x, _ = test_subset[position]
        positions.append(int(position))
        tensors.append(x)
    if missing:
        raise ValueError(f"{len(missing)} sample_index values are not in the configured test split. First missing: {missing[:8]}")
    return np.asarray(positions, dtype=np.int64), torch.stack(tensors, dim=0)


def capture_last_decoder_features(
    config_path: Path,
    checkpoint_path: Path | None,
    sample_indices: np.ndarray,
    device_name: str | None,
) -> dict[str, np.ndarray | int | str]:
    config = load_config(config_path)
    # The saved model_analysis config was copied from the training run and can
    # still point at the old shuffle_* folder. Force this standalone workflow
    # to use the self-contained model_analysis dataset and normalization stats.
    config["data"]["x_path"] = "analysis/model_analysis/data/X.npy"
    config["data"]["y_path"] = "analysis/model_analysis/data/y.npy"
    config["data"]["stats_path"] = "analysis/model_analysis/outputs/norm_stats.npz"
    seed_everything(int(config.get("seed", 42)))
    device = get_device(device_name or config.get("device", "auto"))
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False
    _, _, test_loader, _stats = build_dataloaders(
        config, PROJECT_DIR, int(config.get("seed", 42))
    )
    test_positions, x_cpu = select_samples(test_loader.dataset, sample_indices)
    x = x_cpu.to(device)
    model = build_model(config["model"]).to(device)
    checkpoint = resolve_path(
        checkpoint_path or Path(config["train"]["best_checkpoint"]), PROJECT_DIR
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    store = TensorStore()
    handle = model.up1.register_forward_hook(store.hook)
    try:
        with torch.no_grad():
            prediction = model(x)
    finally:
        handle.remove()
    if store.tensor is None:
        raise RuntimeError("Failed to capture the final decoder feature map")

    return {
        "feature_maps": store.tensor.detach().cpu().numpy().astype(np.float32),
        "prediction": prediction.detach().cpu().numpy().astype(np.float32),
        "sample_index": sample_indices.astype(np.int64),
        "test_position": test_positions.astype(np.int64),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
    }


def load_or_capture_features(args: argparse.Namespace) -> dict[str, np.ndarray | int | str]:
    cache = resolve_workflow_path(args.feature_cache)
    if args.reuse_cache and cache.exists():
        with np.load(cache, allow_pickle=True) as data:
            cached = {key: data[key].item() if data[key].shape == () else data[key] for key in data.files}
        cached_sample = np.asarray(cached.get("sample_index", []), dtype=np.int64).reshape(-1)
        requested_sample = (
            np.asarray([int(args.sample_index)], dtype=np.int64)
            if args.sample_index is not None
            else sample_indices_from_physics(resolve_workflow_path(args.physics))
        )
        if np.array_equal(cached_sample, requested_sample):
            return cached
        print(
            f"Ignoring feature cache for {cached_sample.size} cached samples; "
            f"requested {requested_sample.size} samples",
            flush=True,
        )

    requested_sample = (
        np.asarray([int(args.sample_index)], dtype=np.int64)
        if args.sample_index is not None
        else sample_indices_from_physics(resolve_workflow_path(args.physics))
    )
    result = capture_last_decoder_features(
        resolve_path(args.config),
        args.checkpoint,
        requested_sample,
        args.device,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **result)
    return result


def pca_channel_space(feature_maps: np.ndarray, n_pcs: int) -> dict[str, np.ndarray]:
    features = np.asarray(feature_maps, dtype=np.float64)
    if features.ndim == 3:
        features = features[None, ...]
    if features.ndim != 4:
        raise ValueError(f"Expected feature maps with shape [N,C,H,W], got {features.shape}")

    n_samples, n_channels, height, width = features.shape
    samples_by_channels = np.moveaxis(features, 1, -1).reshape(-1, n_channels)
    mean = np.mean(samples_by_channels, axis=0, keepdims=True)
    centered = samples_by_channels - mean
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_pcs = min(n_pcs, vt.shape[0])
    scores = u[:, :n_pcs] * singular_values[:n_pcs]
    pc_maps = scores.reshape(n_samples, height, width, n_pcs).transpose(0, 3, 1, 2)
    explained_variance = singular_values**2 / max(centered.shape[0] - 1, 1)
    total_variance = np.sum(explained_variance)
    explained_ratio = explained_variance[:n_pcs] / total_variance if total_variance > 0 else np.zeros(n_pcs)
    return {
        "pc_maps": pc_maps.astype(np.float32),
        "components": vt[:n_pcs].astype(np.float32),
        "channel_mean": mean.ravel().astype(np.float32),
        "singular_values": singular_values[:n_pcs].astype(np.float64),
        "explained_variance_ratio": explained_ratio.astype(np.float64),
    }


def pearsonr_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    x = x[valid] - np.mean(x[valid])
    y = y[valid] - np.mean(y[valid])
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float("nan") if denom == 0.0 else float(np.sum(x * y) / denom)


def load_physics(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {name: np.asarray(data[name], dtype=np.float64) for name, _label in PHYSICS_VARIABLES}


def align_pc_signs(pc_maps: np.ndarray, components: np.ndarray, correlations: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aligned_maps = pc_maps.copy()
    aligned_components = components.copy()
    aligned_correlations = correlations.copy()
    signs = np.ones(pc_maps.shape[:2], dtype=np.float32)
    reference = aligned_maps[0]
    for sample in range(1, aligned_maps.shape[0]):
        for pc in range(aligned_maps.shape[1]):
            dot = float(np.nansum(reference[pc] * aligned_maps[sample, pc]))
            if dot < 0.0:
                aligned_maps[sample, pc] *= -1.0
                aligned_components[sample, pc] *= -1.0
                aligned_correlations[sample, :, pc] *= -1.0
                signs[sample, pc] = -1.0
    return aligned_maps, aligned_components, aligned_correlations, signs


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def symmetric_limits(values: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def plot_pc_maps(path: Path, pc_maps: np.ndarray, explained_ratio: np.ndarray) -> None:
    style()
    pcs = np.nanmean(pc_maps, axis=0)
    count = pcs.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 2.8), constrained_layout=True)
    vmin, vmax = symmetric_limits(pcs)
    for pc_index, ax in enumerate(axes.ravel()):
        if pc_index >= count:
            ax.axis("off")
            continue
        mesh = ax.imshow(
            pcs[pc_index],
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"PC{pc_index + 1} ({100.0 * explained_ratio[pc_index]:.1f}%)",
            fontsize=8,
            pad=3,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.025, pad=0.012)
    colorbar.set_label("PC score")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    path: Path,
    correlations: np.ndarray,
    labels: list[str],
    *,
    absolute: bool = False,
) -> None:
    style()
    values = np.abs(correlations) if absolute else correlations
    cmap = (
        mpl.colors.LinearSegmentedColormap.from_list(
            "light_abs_r", ["#ffffff", "#fff7bc", "#fec44f", "#f97316"]
        )
        if absolute
        else "RdBu_r"
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    mesh = ax.imshow(
        values,
        origin="upper",
        aspect="auto",
        cmap=cmap,
        vmin=0 if absolute else -1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([f"PC{i}" for i in range(1, values.shape[1] + 1)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Last-decoder PCA mode")
    title = (
        "Absolute spatial correlation between PCA modes and physical diagnostics"
        if absolute
        else "Spatial correlation between PCA modes and physical diagnostics"
    )
    ax.set_title(title, fontsize=9, pad=5)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "#222222" if absolute else ("white" if np.isfinite(value) and value >= 0.55 else "#222222")
            ax.text(
                column,
                row,
                f"{value:.2f}" if np.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=5.5,
                color=text_color,
            )
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.018)
    colorbar.set_label(r"$|r|$" if absolute else "Spatial Pearson r")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_correlation_csv(
    path: Path,
    correlations: np.ndarray,
    explained_ratio: np.ndarray,
    *,
    absolute: bool = False,
) -> None:
    values = np.abs(correlations) if absolute else correlations
    value_column = "abs_spatial_pearson_r" if absolute else "spatial_pearson_r"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "physics_variable",
                "display_label",
                "pc",
                "explained_variance_ratio",
                value_column,
            ],
        )
        writer.writeheader()
        for row, (variable, label) in enumerate(PHYSICS_VARIABLES):
            for pc in range(values.shape[1]):
                writer.writerow(
                    {
                        "physics_variable": variable,
                        "display_label": label,
                        "pc": pc + 1,
                        "explained_variance_ratio": float(explained_ratio[pc]),
                        value_column: float(values[row, pc]),
                    }
                )


def main() -> None:
    args = parse_args()
    output_dir = resolve_workflow_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_data = load_or_capture_features(args)
    feature_maps = np.asarray(feature_data["feature_maps"], dtype=np.float32)
    physics = load_physics(resolve_workflow_path(args.physics))

    n_samples = feature_maps.shape[0]
    pc_maps_list: list[np.ndarray] = []
    components_list: list[np.ndarray] = []
    channel_mean_list: list[np.ndarray] = []
    singular_values_list: list[np.ndarray] = []
    explained_list: list[np.ndarray] = []
    correlations_list: list[np.ndarray] = []
    for sample in range(n_samples):
        pca = pca_channel_space(feature_maps[sample : sample + 1], args.n_pcs)
        pc_maps_sample = pca["pc_maps"][0]
        pc_maps_list.append(pc_maps_sample)
        components_list.append(pca["components"])
        channel_mean_list.append(pca["channel_mean"])
        singular_values_list.append(pca["singular_values"])
        explained_list.append(pca["explained_variance_ratio"])

        sample_correlations = np.empty((len(PHYSICS_VARIABLES), pc_maps_sample.shape[0]), dtype=np.float64)
        for row, (variable, _label) in enumerate(PHYSICS_VARIABLES):
            field = physics[variable]
            field_sample = field[sample] if field.ndim == 3 else field
            for pc in range(pc_maps_sample.shape[0]):
                sample_correlations[row, pc] = pearsonr_flat(pc_maps_sample[pc], field_sample)
        correlations_list.append(sample_correlations)
        if sample == 0 or (sample + 1) % 10 == 0 or sample + 1 == n_samples:
            print(f"Calculated per-sample PCA {sample + 1}/{n_samples}", flush=True)

    pc_maps = np.stack(pc_maps_list, axis=0).astype(np.float32)
    components = np.stack(components_list, axis=0).astype(np.float32)
    channel_mean = np.stack(channel_mean_list, axis=0).astype(np.float32)
    singular_values = np.stack(singular_values_list, axis=0).astype(np.float64)
    explained_variance_ratio = np.stack(explained_list, axis=0).astype(np.float64)
    physics_correlations_by_sample = np.stack(correlations_list, axis=0).astype(np.float64)
    pc_maps, components, physics_correlations_by_sample, pc_sign = align_pc_signs(
        pc_maps, components, physics_correlations_by_sample
    )
    mean_pc_maps = np.nanmean(pc_maps, axis=0).astype(np.float32)
    mean_components = np.nanmean(components, axis=0).astype(np.float32)
    mean_channel_mean = np.nanmean(channel_mean, axis=0).astype(np.float32)
    mean_singular_values = np.nanmean(singular_values, axis=0).astype(np.float64)
    mean_explained_variance_ratio = np.nanmean(explained_variance_ratio, axis=0).astype(np.float64)
    correlations = np.nanmean(physics_correlations_by_sample, axis=0).astype(np.float64)

    np.savez_compressed(
        output_dir / "last_decoder_pca_results.npz",
        feature_cache=str(resolve_workflow_path(args.feature_cache)),
        sample_index=np.asarray(feature_data["sample_index"], dtype=np.int64),
        test_position=np.asarray(feature_data.get("test_position", -1), dtype=np.int64),
        pc_maps=pc_maps,
        mean_pc_maps=mean_pc_maps,
        components=components,
        mean_components=mean_components,
        channel_mean=channel_mean,
        mean_channel_mean=mean_channel_mean,
        singular_values=singular_values,
        mean_singular_values=mean_singular_values,
        explained_variance_ratio=mean_explained_variance_ratio,
        explained_variance_ratio_by_sample=explained_variance_ratio,
        pc_sign=pc_sign,
        physics_variables=np.asarray([name for name, _label in PHYSICS_VARIABLES]),
        physics_labels=np.asarray([label for _name, label in PHYSICS_VARIABLES]),
        physics_correlations=correlations,
        physics_correlations_by_sample=physics_correlations_by_sample,
    )
    plot_pc_maps(
        output_dir / "last_decoder_pca_first8_maps.png",
        mean_pc_maps[None, ...],
        mean_explained_variance_ratio,
    )
    plot_correlation_heatmap(
        output_dir / "last_decoder_pca_physics_correlations.png",
        correlations,
        [label for _name, label in PHYSICS_VARIABLES],
    )
    plot_correlation_heatmap(
        output_dir / "last_decoder_pca_abs_physics_correlations.png",
        correlations,
        [label for _name, label in PHYSICS_VARIABLES],
        absolute=True,
    )
    plot_correlation_heatmap(
        output_dir / "last_decoder_negative_pca_physics_correlations.png",
        -correlations,
        [label for _name, label in PHYSICS_VARIABLES],
    )
    write_correlation_csv(
        output_dir / "last_decoder_pca_physics_correlations.csv",
        correlations,
        mean_explained_variance_ratio,
    )
    write_correlation_csv(
        output_dir / "last_decoder_pca_abs_physics_correlations.csv",
        correlations,
        mean_explained_variance_ratio,
        absolute=True,
    )
    write_correlation_csv(
        output_dir / "last_decoder_negative_pca_physics_correlations.csv",
        -correlations,
        mean_explained_variance_ratio,
    )
    summary = {
        "sample_index": np.asarray(feature_data["sample_index"], dtype=np.int64).reshape(-1).astype(int).tolist(),
        "test_position": np.asarray(feature_data.get("test_position", []), dtype=np.int64).reshape(-1).astype(int).tolist(),
        "n_selected_samples": int(np.asarray(feature_data["sample_index"]).size),
        "feature_maps_shape": list(feature_maps.shape),
        "pca_method": "PCA is calculated independently for each sample on [H*W, C], then PC maps/correlations are sign-aligned and averaged across samples.",
        "pca_input_matrix_per_sample": "[H*W, C]",
        "n_pcs": int(pc_maps.shape[1]),
        "mean_explained_variance_ratio": mean_explained_variance_ratio.tolist(),
        "physics_variables": [name for name, _label in PHYSICS_VARIABLES],
        "model_analysis_note": "Defaults point to analysis/model_analysis and use the top-90 test samples listed in the physics file.",
    }
    (output_dir / "last_decoder_pca_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved PCA outputs under {output_dir}")


if __name__ == "__main__":
    main()
