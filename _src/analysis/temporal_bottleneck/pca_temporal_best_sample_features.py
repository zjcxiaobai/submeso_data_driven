#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def choose_best_position(r2: np.ndarray) -> int:
    valid = np.where(np.isfinite(r2))[0]
    if valid.size == 0:
        raise ValueError("No finite per-sample R2 values found.")
    return int(valid[np.nanargmax(r2[valid])])


def pca_spatial_features(feature_maps: np.ndarray) -> dict[str, np.ndarray]:
    """PCA on one sample's latent maps with pixels as observations and features as variables."""
    if feature_maps.ndim != 3:
        raise ValueError(f"Expected feature_maps shape (C,H,W), got {feature_maps.shape}")
    channels, height, width = feature_maps.shape
    x = feature_maps.reshape(channels, height * width).T.astype(np.float64)  # (pixels, C)
    mean = x.mean(axis=0, keepdims=True)
    x0 = x - mean
    u, s, vt = np.linalg.svd(x0, full_matrices=False)
    scores = u * s
    explained_variance = (s**2) / max(1, x0.shape[0] - 1)
    explained_ratio = explained_variance / explained_variance.sum()
    return {
        "scores": scores,
        "score_maps": scores.T.reshape(channels, height, width),
        "components": vt,
        "mean": mean.ravel(),
        "singular_values": s,
        "explained_variance": explained_variance,
        "explained_ratio": explained_ratio,
    }


def robust_limits(arr: np.ndarray, q: float = 99.0) -> tuple[float, float]:
    vmax = float(np.nanpercentile(np.abs(arr), q))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    return -vmax, vmax


def plot_pc_maps(score_maps: np.ndarray, explained_ratio: np.ndarray, output: Path) -> None:
    n_pc = min(6, score_maps.shape[0])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
    vmin, vmax = robust_limits(score_maps[:n_pc])
    for i, ax in enumerate(axes.flat):
        if i >= n_pc:
            ax.axis("off")
            continue
        im = ax.imshow(score_maps[i], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"PC{i + 1} ({explained_ratio[i] * 100:.1f}%)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="PC score")
    fig.suptitle("PCA score maps of best-sample temporal latent features", fontsize=12)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_loadings(components: np.ndarray, explained_ratio: np.ndarray, output: Path) -> None:
    n_pc = min(6, components.shape[0])
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    im = ax.imshow(components[:n_pc], vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(components.shape[1]), [f"feature {i + 1}" for i in range(components.shape[1])], rotation=35, ha="right")
    ax.set_yticks(range(n_pc), [f"PC{i + 1} ({explained_ratio[i] * 100:.1f}%)" for i in range(n_pc)])
    ax.set_xlabel("learned latent feature map")
    ax.set_ylabel("principal component")
    fig.colorbar(im, ax=ax, label="loading")
    ax.set_title("PCA loadings for best-sample temporal latent features")
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_variance(explained_ratio: np.ndarray, output: Path) -> None:
    x = np.arange(1, explained_ratio.size + 1)
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    ax.bar(x, explained_ratio * 100, color="#4C78A8")
    ax.plot(x, np.cumsum(explained_ratio) * 100, color="#D55E00", marker="o", linewidth=1.4)
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.set_title("Best-sample temporal latent PCA variance")
    fig.savefig(output, dpi=240)
    plt.close(fig)


def save_summary(path: Path, position: int, sample_index: int | None, sample_r2: float | None, pca: dict[str, np.ndarray]) -> None:
    with path.open("w") as handle:
        handle.write(f"best_test_position: {position}\n")
        if sample_index is not None:
            handle.write(f"sample_index: {sample_index}\n")
        if sample_r2 is not None:
            handle.write(f"sample_r2: {sample_r2}\n")
        handle.write("explained_variance_ratio:\n")
        for i, value in enumerate(pca["explained_ratio"], start=1):
            handle.write(f"  PC{i}: {value:.10f}\n")
        handle.write("\nloadings_rows_PC_columns_feature:\n")
        for i, row in enumerate(pca["components"], start=1):
            vals = ", ".join(f"{v:.8f}" for v in row)
            handle.write(f"  PC{i}: [{vals}]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA analysis of the best test sample's 6 temporal-bottleneck feature maps.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("analysis/temporal_bottleneck/outputs/test_predictions_denorm.npz"),
        help="NPZ produced by run_bottleneck.py evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/temporal_bottleneck/feature_plots/pca_best_sample"),
        help="Output directory for PCA figures and arrays.",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=None,
        help="Optional test position. Defaults to the sample with highest per-sample R2.",
    )
    args = parser.parse_args()

    data = np.load(args.predictions)
    if "latent_features" not in data.files:
        raise KeyError("NPZ does not contain latent_features. Re-run bottleneck evaluation first.")
    features = np.asarray(data["latent_features"])
    if features.ndim != 4:
        raise ValueError(f"Expected latent_features shape (N,C,H,W), got {features.shape}")
    r2 = np.asarray(data["r2"]) if "r2" in data.files else None
    position = choose_best_position(r2) if args.position is None and r2 is not None else int(args.position or 0)
    sample_index = int(data["sample_index"][position]) if "sample_index" in data.files else None
    sample_r2 = float(r2[position]) if r2 is not None else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pca = pca_spatial_features(features[position])
    np.savez_compressed(
        args.output_dir / "best_sample_latent_pca.npz",
        test_position=np.asarray(position),
        sample_index=np.asarray(-1 if sample_index is None else sample_index),
        sample_r2=np.asarray(np.nan if sample_r2 is None else sample_r2),
        feature_maps=features[position],
        pc_score_maps=pca["score_maps"],
        components=pca["components"],
        mean=pca["mean"],
        explained_variance=pca["explained_variance"],
        explained_ratio=pca["explained_ratio"],
    )
    plot_pc_maps(pca["score_maps"], pca["explained_ratio"], args.output_dir / "best_sample_pc_score_maps.png")
    plot_loadings(pca["components"], pca["explained_ratio"], args.output_dir / "best_sample_pc_loadings.png")
    plot_variance(pca["explained_ratio"], args.output_dir / "best_sample_explained_variance.png")
    save_summary(args.output_dir / "best_sample_pca_summary.txt", position, sample_index, sample_r2, pca)

    print(f"best test position: {position}")
    if sample_index is not None:
        print(f"sample index: {sample_index}")
    if sample_r2 is not None:
        print(f"sample R2: {sample_r2:.6f}")
    print("explained variance ratio:", ", ".join(f"PC{i+1}={v:.4f}" for i, v in enumerate(pca["explained_ratio"])))
    print(f"saved PCA outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
