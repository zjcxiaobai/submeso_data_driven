#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml


def robust_limits(arr: np.ndarray, q: float = 99.0) -> tuple[float, float]:
    vmax = float(np.nanpercentile(np.abs(arr), q))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(arr))) if np.isfinite(np.nanmax(np.abs(arr))) else 1.0
    return -vmax, vmax


def sample_label(sample_index: np.ndarray, position: int) -> str:
    if sample_index is None:
        return f"test position {position}"
    return f"test position {position}, sample index {int(sample_index[position])}"


def plot_feature_grid(features: np.ndarray, sample_index: np.ndarray | None, positions: list[int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples, n_features, height, width = features.shape
    for pos in positions:
        if pos < 0 or pos >= n_samples:
            raise IndexError(f"Position {pos} is outside available range 0..{n_samples - 1}")
        maps = features[pos]
        vmin, vmax = robust_limits(maps)
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
        for k, ax in enumerate(axes.flat):
            im = ax.imshow(maps[k], origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"feature {k + 1}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="activation")
        fig.suptitle(sample_label(sample_index, pos), fontsize=12)
        fig.savefig(output_dir / f"features_sample_{pos:04d}.png", dpi=220)
        plt.close(fig)


def plot_feature_means(features: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_maps = np.nanmean(features, axis=0)
    std_maps = np.nanstd(features, axis=0)

    for name, maps, cmap in (("mean", mean_maps, "RdBu_r"), ("std", std_maps, "viridis")):
        if name == "mean":
            vmin, vmax = robust_limits(maps)
        else:
            vmin, vmax = float(np.nanmin(maps)), float(np.nanpercentile(maps, 99.0))
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), constrained_layout=True)
        for k, ax in enumerate(axes.flat):
            im = ax.imshow(maps[k], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"feature {k + 1}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label=f"activation {name}")
        fig.suptitle(f"Latent feature {name} across test set", fontsize=12)
        fig.savefig(output_dir / f"features_test_{name}.png", dpi=220)
        plt.close(fig)


def plot_feature_correlations(features: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = features.transpose(1, 0, 2, 3).reshape(features.shape[1], -1)
    corr = np.corrcoef(flat)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(features.shape[1]), [str(i + 1) for i in range(features.shape[1])])
    ax.set_yticks(range(features.shape[1]), [str(i + 1) for i in range(features.shape[1])])
    ax.set_xlabel("feature")
    ax.set_ylabel("feature")
    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Latent feature correlation")
    fig.savefig(output_dir / "features_correlation.png", dpi=220)
    plt.close(fig)


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def load_feature_names(config: dict, metadata_path: Path | None) -> list[str]:
    if metadata_path is not None and metadata_path.exists():
        with metadata_path.open("r") as handle:
            metadata = json.load(handle)
        names = metadata.get("features")
        if names:
            return [str(name) for name in names]
    input_channels = int(config["model"]["input_channels"])
    n_offsets = len(config["data"].get("temporal_offsets", [0]))
    channels_per_time = input_channels // max(1, n_offsets)
    return [f"ch{i}" for i in range(channels_per_time)]


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    if mask.sum() < 2:
        return float("nan")
    av = av[mask] - np.mean(av[mask])
    bv = bv[mask] - np.mean(bv[mask])
    denom = np.sqrt(np.sum(av**2) * np.sum(bv**2))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(av * bv) / denom)


def input_snapshot_for_sample(
    x: np.ndarray,
    center_index: int,
    temporal_offsets: list[int],
    input_mode: str,
) -> tuple[np.ndarray, list[int]]:
    if input_mode == "last":
        return np.asarray(x[center_index], dtype=np.float32), [0]
    if input_mode == "all-offsets":
        frames = [np.asarray(x[center_index + offset], dtype=np.float32) for offset in temporal_offsets]
        return np.concatenate(frames, axis=0), temporal_offsets
    raise ValueError("input_mode must be 'last' or 'all-offsets'")


def expanded_feature_names(base_names: list[str], offsets_used: list[int]) -> list[str]:
    if offsets_used == [0]:
        return base_names
    names = []
    for offset in offsets_used:
        names.extend([f"{name}@t{offset:+d}" for name in base_names])
    return names


def latent_input_corr_matrix(latent_maps: np.ndarray, input_maps: np.ndarray) -> np.ndarray:
    out = np.empty((latent_maps.shape[0], input_maps.shape[0]), dtype=np.float64)
    for i in range(latent_maps.shape[0]):
        for j in range(input_maps.shape[0]):
            out[i, j] = spatial_corr(latent_maps[i], input_maps[j])
    return out


def save_corr_csv(path: Path, corr: np.ndarray, input_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("latent_feature," + ",".join(input_names) + "\n")
        for i, row in enumerate(corr, start=1):
            handle.write(f"feature_{i}," + ",".join(f"{value:.8g}" for value in row) + "\n")


def plot_latent_input_correlations(
    *,
    features: np.ndarray,
    sample_index: np.ndarray,
    positions: list[int],
    config: dict,
    metadata_path: Path | None,
    input_mode: str,
    output_dir: Path,
) -> None:
    root = project_dir()
    x_path = resolve(root, config["data"]["x_path"])
    x = np.load(x_path, mmap_mode="r")
    temporal_offsets = [int(v) for v in config["data"].get("temporal_offsets", [0])]
    base_names = load_feature_names(config, metadata_path)
    corr_dir = output_dir / "input_correlations"
    corr_dir.mkdir(parents=True, exist_ok=True)

    for pos in positions:
        center_index = int(sample_index[pos])
        input_maps, offsets_used = input_snapshot_for_sample(x, center_index, temporal_offsets, input_mode)
        input_names = expanded_feature_names(base_names, offsets_used)
        corr = latent_input_corr_matrix(features[pos], input_maps)
        save_corr_csv(corr_dir / f"latent_input_corr_sample_{pos:04d}.csv", corr, input_names)

        fig_width = max(7.0, 0.5 * len(input_names) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(input_names)), input_names, rotation=45, ha="right")
        ax.set_yticks(range(features.shape[1]), [f"feature {i + 1}" for i in range(features.shape[1])])
        ax.set_xlabel("input feature at same sample time" if input_mode == "last" else "input feature and temporal offset")
        ax.set_ylabel("learned latent map")
        ax.set_title(f"Latent-input spatial correlation, test position {pos}, sample index {center_index}")
        fig.colorbar(im, ax=ax, label="Pearson r across grid cells")
        fig.savefig(corr_dir / f"latent_input_corr_sample_{pos:04d}.png", dpi=220)
        plt.close(fig)


def parse_positions(raw: str, n_samples: int, r2: np.ndarray | None) -> list[int]:
    if raw == "first-middle-last":
        return sorted(set([0, n_samples // 2, n_samples - 1]))
    if raw == "best-worst-middle":
        if r2 is None:
            return sorted(set([0, n_samples // 2, n_samples - 1]))
        valid = np.where(np.isfinite(r2))[0]
        best = int(valid[np.nanargmax(r2[valid])])
        worst = int(valid[np.nanargmin(r2[valid])])
        return sorted(set([best, n_samples // 2, worst]))
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learned temporal-bottleneck spatial feature maps.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("analysis/temporal_bottleneck/outputs/test_predictions_denorm.npz"),
        help="NPZ produced by run_bottleneck.py evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/temporal_bottleneck/feature_plots"),
        help="Directory for generated feature plots.",
    )
    parser.add_argument(
        "--samples",
        default="best-worst-middle",
        help="Comma-separated test positions, or first-middle-last, or best-worst-middle.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("analysis/temporal_bottleneck/config.yaml"),
        help="Temporal bottleneck experiment config, used to locate X.npy and temporal offsets.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("analysis/model_analysis/data/metadata.json"),
        help="Metadata JSON containing input feature names.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("last", "all-offsets"),
        default="last",
        help="Use same-time input features only, or all temporal-offset input maps.",
    )
    args = parser.parse_args()

    data = np.load(args.predictions)
    if "latent_features" not in data.files:
        raise KeyError(f"{args.predictions} does not contain latent_features. Re-run evaluation with the spatial bottleneck script.")
    features = np.asarray(data["latent_features"])
    if features.ndim != 4:
        raise ValueError(f"Expected latent_features with shape (N, C, H, W), got {features.shape}")
    sample_index = np.asarray(data["sample_index"]) if "sample_index" in data.files else None
    r2 = np.asarray(data["r2"]) if "r2" in data.files else None

    positions = parse_positions(args.samples, features.shape[0], r2)
    plot_feature_grid(features, sample_index, positions, args.output_dir)
    plot_feature_means(features, args.output_dir)
    plot_feature_correlations(features, args.output_dir)

    if sample_index is not None:
        config = load_config(resolve(project_dir(), args.config))
        metadata_path = resolve(project_dir(), args.metadata)
        plot_latent_input_correlations(
            features=features,
            sample_index=sample_index,
            positions=positions,
            config=config,
            metadata_path=metadata_path,
            input_mode=args.input_mode,
            output_dir=args.output_dir,
        )

    print(f"latent_features shape: {features.shape}")
    print(f"saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
