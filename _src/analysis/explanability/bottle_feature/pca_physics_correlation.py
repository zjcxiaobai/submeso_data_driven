#!/usr/bin/env python3
"""PCA/correlation analysis for the best-sample bottleneck feature maps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PREDICTIONS = PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz"
DEFAULT_PHYSICS = HERE / "physics_variables_best_sample/physics_variables.npz"
DEFAULT_MANIFEST = HERE / "physics_variables_best_sample/physics_variable_manifest.csv"
DEFAULT_OUTPUT_DIR = HERE / "best_sample_pca_physics"
BOTTLENECK_FEATURE_NAMES = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-index", type=int, default=None, help="Optional explicit sample_index. Default is best test R2 sample.")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


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


def symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def best_sample_from_predictions(path: Path) -> tuple[int, int, float, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
        latent = np.asarray(data["latent_features"], dtype=np.float64)
    valid = np.where(np.isfinite(r2))[0]
    if valid.size == 0:
        raise ValueError("No finite test-sample R2 values found")
    pos = int(valid[np.argmax(r2[valid])])
    return int(sample_index[pos]), pos, float(r2[pos]), latent


def sample_position(path: Path, requested_sample: int | None) -> tuple[int, int, float, np.ndarray]:
    best_sample, best_pos, best_r2, latent = best_sample_from_predictions(path)
    if requested_sample is None:
        return best_sample, best_pos, best_r2, latent
    with np.load(path, allow_pickle=True) as data:
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
    matches = np.where(sample_index == int(requested_sample))[0]
    if matches.size == 0:
        raise ValueError(f"sample_index {requested_sample} was not found in {path}")
    pos = int(matches[0])
    return int(requested_sample), pos, float(r2[pos]), latent


def pca_channel_space(feature_maps: np.ndarray) -> dict[str, np.ndarray]:
    if feature_maps.ndim != 3:
        raise ValueError(f"Expected feature maps shape (4,H,W), got {feature_maps.shape}")
    channels, height, width = feature_maps.shape
    matrix = np.moveaxis(feature_maps, 0, -1).reshape(-1, channels)
    mean = np.mean(matrix, axis=0, keepdims=True)
    centered = matrix - mean
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    n_pcs = min(channels, vt.shape[0])
    scores = u[:, :n_pcs] * singular_values[:n_pcs]
    pc_maps = scores.reshape(height, width, n_pcs).transpose(2, 0, 1)
    explained_variance = singular_values**2 / max(centered.shape[0] - 1, 1)
    total = float(np.sum(explained_variance))
    ratio = explained_variance[:n_pcs] / total if total > 0 else np.zeros(n_pcs)
    return {
        "pc_maps": pc_maps.astype(np.float32),
        "components": vt[:n_pcs].astype(np.float32),
        "channel_mean": mean.ravel().astype(np.float32),
        "singular_values": singular_values[:n_pcs].astype(np.float64),
        "explained_variance_ratio": ratio.astype(np.float64),
    }


def load_variable_names(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback
    names: list[str] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("variable", "")
            if name:
                names.append(name)
    return names or fallback


def load_physics(path: Path, manifest: Path) -> tuple[np.ndarray, list[str], int]:
    with np.load(path, allow_pickle=True) as data:
        sample_index = np.asarray(data["sample_index"], dtype=np.int64).reshape(-1)
        candidates = [name for name in data.files if name not in {"sample_index", "test_position", "time_index", "selected_r2"}]
        variable_names = [name for name in load_variable_names(manifest, candidates) if name in data.files]
        fields = []
        for name in variable_names:
            arr = np.asarray(data[name], dtype=np.float64)
            if arr.ndim == 3:
                fields.append(arr[0])
            elif arr.ndim == 2:
                fields.append(arr)
            else:
                raise ValueError(f"Physics variable {name} has unsupported shape {arr.shape}")
    if sample_index.size != 1:
        raise ValueError(f"Expected one physics sample, got {sample_index.size}")
    return np.stack(fields, axis=0), variable_names, int(sample_index[0])


def plot_pca_results(path: Path, features: np.ndarray, pca: dict[str, np.ndarray], sample_index: int, r2: float, dpi: int) -> None:
    style()
    pc_maps = pca["pc_maps"]
    ratios = pca["explained_variance_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), constrained_layout=True)
    feature_limit = symmetric_limit(features)
    pc_limit = symmetric_limit(pc_maps)
    feature_norm = TwoSlopeNorm(vmin=-feature_limit, vcenter=0.0, vmax=feature_limit)
    pc_norm = TwoSlopeNorm(vmin=-pc_limit, vcenter=0.0, vmax=pc_limit)
    meshes = []
    for i in range(4):
        meshes.append(axes[0, i].imshow(features[i], origin="lower", aspect="equal", cmap="RdBu_r", norm=feature_norm, interpolation="nearest"))
        axes[0, i].set_title(f"Feature {i + 1}", fontsize=8, pad=3)
        meshes.append(axes[1, i].imshow(pc_maps[i], origin="lower", aspect="equal", cmap="RdBu_r", norm=pc_norm, interpolation="nearest"))
        axes[1, i].set_title(f"PC{i + 1} ({100 * ratios[i]:.1f}%)", fontsize=8, pad=3)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(meshes[0], ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.012, label="Latent activation")
    fig.colorbar(meshes[1], ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.012, label="PC score")
    fig.suptitle(f"Best-sample bottleneck PCA: sample {sample_index}, R2={r2:.3f}", fontsize=9, y=1.02)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_corr_heatmap(path: Path, corr: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, dpi: int) -> None:
    style()
    height = max(6.0, 0.16 * len(row_labels) + 1.4)
    width = max(4.5, 0.42 * len(col_labels) + 2.7)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    mesh = ax.imshow(corr, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.8)
    ax.set_title(title, fontsize=9, pad=5)
    ax.tick_params(length=2)
    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = corr[row, col]
            if not np.isfinite(value):
                label = "NA"
                color = "#222222"
            else:
                label = f"{value:.2f}"
                color = "white" if abs(value) >= 0.55 else "#222222"
            ax.text(col, row, label, ha="center", va="center", fontsize=4.8, color=color)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.026, pad=0.014)
    cbar.set_label("Spatial Pearson r", fontsize=7)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_corr_csv(path: Path, corr: np.ndarray, row_labels: list[str], col_labels: list[str], row_name: str, col_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[row_name, col_name, "spatial_pearson_r", "abs_spatial_pearson_r"])
        writer.writeheader()
        for row, rname in enumerate(row_labels):
            for col, cname in enumerate(col_labels):
                value = float(corr[row, col])
                writer.writerow({row_name: rname, col_name: cname, "spatial_pearson_r": value, "abs_spatial_pearson_r": abs(value)})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index, test_position, r2, latent = sample_position(args.predictions, args.sample_index)
    feature_maps = np.asarray(latent[test_position], dtype=np.float64)
    if feature_maps.shape[0] != 4:
        raise ValueError(f"Expected 4 bottleneck features, got {feature_maps.shape}")

    physics_maps, physics_names, physics_sample = load_physics(args.physics, args.manifest)
    if physics_sample != sample_index:
        raise ValueError(f"Physics sample {physics_sample} does not match bottleneck sample {sample_index}. Re-run prepare_physics_variables.py.")

    pca = pca_channel_space(feature_maps)
    pc_maps = pca["pc_maps"]

    physics_feature_corr = np.empty((len(physics_names), len(BOTTLENECK_FEATURE_NAMES)), dtype=np.float64)
    physics_pc_corr = np.empty((len(physics_names), pc_maps.shape[0]), dtype=np.float64)
    for row, physics in enumerate(physics_maps):
        for col, feature_map in enumerate(feature_maps):
            physics_feature_corr[row, col] = pearsonr_flat(physics, feature_map)
        for col, pc_map in enumerate(pc_maps):
            physics_pc_corr[row, col] = pearsonr_flat(physics, pc_map)

    np.savez_compressed(
        output_dir / "bottleneck_pca_physics_correlations.npz",
        sample_index=np.asarray([sample_index], dtype=np.int64),
        test_position=np.asarray([test_position], dtype=np.int64),
        r2=np.asarray([r2], dtype=np.float64),
        bottleneck_features=feature_maps.astype(np.float32),
        pc_maps=pc_maps.astype(np.float32),
        pca_components=pca["components"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        physics_variables=np.asarray(physics_names),
        bottleneck_feature_names=np.asarray(BOTTLENECK_FEATURE_NAMES),
        physics_bottleneck_feature_correlations=physics_feature_corr,
        physics_pca_correlations=physics_pc_corr,
    )

    plot_pca_results(output_dir / "bottleneck_4_features_pca_best_sample.png", feature_maps, pca, sample_index, r2, args.dpi)
    plot_corr_heatmap(
        output_dir / "physics_vs_bottleneck_features_correlation.png",
        physics_feature_corr,
        physics_names,
        BOTTLENECK_FEATURE_NAMES,
        f"Physics variables vs bottleneck feature maps: sample {sample_index}",
        args.dpi,
    )
    plot_corr_heatmap(
        output_dir / "physics_vs_bottleneck_pca_correlation.png",
        physics_pc_corr,
        physics_names,
        [f"PC{i}" for i in range(1, pc_maps.shape[0] + 1)],
        f"Physics variables vs bottleneck PCA components: sample {sample_index}",
        args.dpi,
    )
    write_corr_csv(
        output_dir / "physics_vs_bottleneck_features_correlation.csv",
        physics_feature_corr,
        physics_names,
        BOTTLENECK_FEATURE_NAMES,
        "physics_variable",
        "bottleneck_feature",
    )
    write_corr_csv(
        output_dir / "physics_vs_bottleneck_pca_correlation.csv",
        physics_pc_corr,
        physics_names,
        [f"PC{i}" for i in range(1, pc_maps.shape[0] + 1)],
        "physics_variable",
        "pca_component",
    )
    print(f"Saved bottleneck PCA/correlation outputs under {output_dir}")


if __name__ == "__main__":
    main()
