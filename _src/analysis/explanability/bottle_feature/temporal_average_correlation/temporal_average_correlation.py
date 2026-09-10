#!/usr/bin/env python3
"""Test how rolling averages across test samples affect physics-bottleneck correlations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[4]
SOURCE_SCRIPT = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation/model_analysis_last_decoder_pca/prepare_physics_variables.py"
DEFAULT_X = PROJECT_DIR / "analysis/model_analysis/data/X.npy"
DEFAULT_PREDICTIONS = PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz"
DEFAULT_OUTPUT_DIR = HERE / "outputs"
DEFAULT_CACHE = HERE / "physics_variable_test_series.npz"

spec = importlib.util.spec_from_file_location("last_decoder_prepare_physics_variables", SOURCE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {SOURCE_SCRIPT}")
source_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_module)
FEATURE_NAMES = list(source_module.FEATURE_NAMES)
calculate_fields = source_module.calculate_fields

VARIABLE_SPECS: list[tuple[str, str, str]] = [
    ("T", "Thermal structure", r"$T$"),
    ("rho", "Density structure", r"$\rho$"),
    ("gradT", "Thermal-front strength", r"$|\nabla_h T|$"),
    ("strain", "Deformation strain", r"$S$"),
    ("strain_times_gradT", "Compound frontal deformation", r"$S|\nabla_h T|$"),
    ("frontogenesis_FT", "Frontogenesis", r"$F_T$"),
    ("zeta", "Rotation", r"$\zeta$"),
    ("Wprime_times_Tprime", "Flux-like diagnostic", r"$w'T'$"),
    ("KPPhbl", "Mixed-layer structure", r"$h_{bl}$"),
]
BOTTLENECK_LABELS = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
PCA_LABELS = ["PC1", "PC2", "PC3", "PC4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--variables", nargs="+", default=[name for name, _group, _label in VARIABLE_SPECS])
    parser.add_argument("--rolling-windows", nargs="+", type=int, default=[1, 3, 7, 15, 30, 60, 90, 180, 365])
    parser.add_argument("--dx-m", type=float, default=50_000.0)
    parser.add_argument("--dy-m", type=float, default=50_000.0)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--domain-height-m", type=float, default=1.0e6)
    parser.add_argument("--beta-reference", choices=("center", "south", "north"), default="center")
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
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
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


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


def pca_channel_space(feature_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if feature_maps.ndim != 3:
        raise ValueError(f"Expected bottleneck features shape (4,H,W), got {feature_maps.shape}")
    channels, height, width = feature_maps.shape
    matrix = np.moveaxis(feature_maps, 0, -1).reshape(-1, channels)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :channels] * singular_values[:channels]
    pc_maps = scores.reshape(height, width, channels).transpose(2, 0, 1)
    explained_variance = singular_values**2 / max(centered.shape[0] - 1, 1)
    total = float(np.sum(explained_variance))
    ratio = explained_variance[:channels] / total if total > 0 else np.zeros(channels)
    return pc_maps.astype(np.float64), ratio.astype(np.float64)


def trailing_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError(f"Rolling window must be positive, got {window}")
    if window > values.shape[0]:
        raise ValueError(f"Rolling window {window} exceeds sample count {values.shape[0]}")
    cumsum = np.cumsum(values, axis=0, dtype=np.float64)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / float(window)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        for key in ["sample_index", "r2", "latent_features"]:
            if key not in data.files:
                raise KeyError(f"{path} is missing {key}")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
        latent = np.asarray(data["latent_features"], dtype=np.float64)
    if latent.ndim != 4 or latent.shape[1] != 4:
        raise ValueError(f"Expected latent_features shape (N,4,H,W), got {latent.shape}")
    return sample_index, r2, latent


def calculate_physics_series(
    x_path: Path,
    sample_index: np.ndarray,
    variables: list[str],
    *,
    dx_m: float,
    dy_m: float,
    f0: float,
    beta: float,
    domain_height_m: float,
    beta_reference: str,
) -> dict[str, np.ndarray]:
    x = np.load(x_path, mmap_mode="r")
    if x.ndim != 4 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected X shape (N,{len(FEATURE_NAMES)},H,W), got {x.shape}")
    series = {name: [] for name in variables}
    for n, idx in enumerate(sample_index, start=1):
        source = {name: np.asarray(x[int(idx), channel], dtype=np.float64) for channel, name in enumerate(FEATURE_NAMES)}
        fields, _definitions = calculate_fields(source, dx_m, dy_m, f0, beta, domain_height_m, beta_reference)
        for variable in variables:
            if variable not in fields:
                raise KeyError(f"Calculated physics fields do not include {variable}")
            series[variable].append(np.asarray(fields[variable], dtype=np.float32))
        if n == 1 or n % 100 == 0 or n == sample_index.size:
            print(f"Prepared physics variables for {n}/{sample_index.size} test samples", flush=True)
    return {name: np.stack(values, axis=0).astype(np.float32) for name, values in series.items()}


def load_or_create_physics_cache(args: argparse.Namespace, sample_index: np.ndarray, variables: list[str]) -> dict[str, np.ndarray]:
    cache = args.cache.expanduser()
    if args.reuse_cache and cache.exists():
        with np.load(cache, allow_pickle=True) as data:
            cached_samples = np.asarray(data["sample_index"], dtype=np.int64) if "sample_index" in data.files else np.asarray([], dtype=np.int64)
            if np.array_equal(cached_samples, sample_index) and all(name in data.files for name in variables):
                print(f"Using cached physics variable time series: {cache}", flush=True)
                return {name: np.asarray(data[name], dtype=np.float64) for name in variables}
            print("Ignoring physics cache because samples or variables do not match", flush=True)
    series = calculate_physics_series(
        args.x_path.expanduser(),
        sample_index,
        variables,
        dx_m=args.dx_m,
        dy_m=args.dy_m,
        f0=args.f0,
        beta=args.beta,
        domain_height_m=args.domain_height_m,
        beta_reference=args.beta_reference,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, sample_index=sample_index.astype(np.int64), variables=np.asarray(variables), **series)
    print(f"Saved physics variable time-series cache: {cache}", flush=True)
    return {name: np.asarray(values, dtype=np.float64) for name, values in series.items()}


def summarize_correlations(physics_series: np.ndarray, latent: np.ndarray, windows: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = np.empty((len(windows), 4), dtype=np.float64)
    feature_std = np.empty((len(windows), 4), dtype=np.float64)
    pca_mean = np.empty((len(windows), 4), dtype=np.float64)
    pca_std = np.empty((len(windows), 4), dtype=np.float64)
    for row, window in enumerate(windows):
        physics_roll = trailing_rolling_mean(physics_series, window)
        latent_roll = trailing_rolling_mean(latent, window)
        feature_corr_samples = np.empty((physics_roll.shape[0], 4), dtype=np.float64)
        pca_corr_samples = np.empty((physics_roll.shape[0], 4), dtype=np.float64)
        for sample in range(physics_roll.shape[0]):
            pc_maps, _ratio = pca_channel_space(latent_roll[sample])
            for col in range(4):
                feature_corr_samples[sample, col] = pearsonr_flat(physics_roll[sample], latent_roll[sample, col])
                pca_corr_samples[sample, col] = pearsonr_flat(physics_roll[sample], pc_maps[col])
        feature_mean[row] = np.nanmean(feature_corr_samples, axis=0)
        feature_std[row] = np.nanstd(feature_corr_samples, axis=0)
        pca_mean[row] = np.nanmean(pca_corr_samples, axis=0)
        pca_std[row] = np.nanstd(pca_corr_samples, axis=0)
    return feature_mean, feature_std, pca_mean, pca_std


def mean_pca_ratio_for_largest_window(latent: np.ndarray, window: int) -> np.ndarray:
    latent_roll = trailing_rolling_mean(latent, window)
    ratios = []
    for sample in range(latent_roll.shape[0]):
        _pc_maps, ratio = pca_channel_space(latent_roll[sample])
        ratios.append(ratio)
    return np.nanmean(np.stack(ratios, axis=0), axis=0)


def plot_variable(
    output: Path,
    variable: str,
    group: str,
    label: str,
    windows: list[int],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    pca_mean: np.ndarray,
    pca_std: np.ndarray,
    pca_ratio: np.ndarray,
    n_samples: int,
    dpi: int,
) -> None:
    style()
    fig, ax = plt.subplots(figsize=(5.9, 3.35), constrained_layout=True)
    feature_colors = ["#275D8C", "#3C8DAD", "#66A182", "#D08C60"]
    pc_colors = ["#1B1B1B", "#7A5195", "#BC5090", "#EF5675"]
    x = np.asarray(windows, dtype=float)
    for i, name in enumerate(BOTTLENECK_LABELS):
        ax.plot(x, feature_mean[:, i], marker="o", linewidth=1.25, color=feature_colors[i], label=name)
        ax.fill_between(x, feature_mean[:, i] - feature_std[:, i], feature_mean[:, i] + feature_std[:, i], color=feature_colors[i], alpha=0.08, linewidth=0)
    for i, name in enumerate(PCA_LABELS):
        ax.plot(x, pca_mean[:, i], marker="s", linewidth=1.05, linestyle="--", color=pc_colors[i], label=f"{name} ({100 * pca_ratio[i]:.1f}%)")
        ax.fill_between(x, pca_mean[:, i] - pca_std[:, i], pca_mean[:, i] + pca_std[:, i], color=pc_colors[i], alpha=0.06, linewidth=0)
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xticks(windows)
    ax.set_xticklabels([str(v) for v in windows])
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("Rolling mean window across test samples")
    ax.set_ylabel("Mean spatial Pearson r")
    ax.set_title(f"{group}: {label}\nrolling windows over {n_samples} test samples", fontsize=9, pad=5)
    ax.grid(axis="y", color="#D6D6D6", linewidth=0.45)
    ax.legend(ncol=2, fontsize=6.1, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_lookup = {name: (group, label) for name, group, label in VARIABLE_SPECS}
    missing = [name for name in args.variables if name not in variable_lookup]
    if missing:
        raise ValueError(f"Unknown selected variable(s): {missing}")

    sample_index, r2, latent = load_predictions(args.predictions.expanduser())
    order = np.argsort(sample_index)
    sample_index = sample_index[order]
    r2 = r2[order]
    latent = latent[order]
    windows = sorted({int(v) for v in args.rolling_windows if int(v) > 0 and int(v) <= latent.shape[0]})
    if not windows:
        raise ValueError("No valid rolling windows remain after filtering by test-sample count")

    physics = load_or_create_physics_cache(args, sample_index, args.variables)
    pca_ratio = mean_pca_ratio_for_largest_window(latent, max(windows))

    all_rows: list[dict[str, str | int | float]] = []
    feature_results = []
    pca_results = []
    for variable in args.variables:
        group, label = variable_lookup[variable]
        series = np.asarray(physics[variable], dtype=np.float64)
        if series.shape[0] != latent.shape[0]:
            raise ValueError(f"{variable} has {series.shape[0]} samples but latent has {latent.shape[0]}")
        feature_mean, feature_std, pca_mean, pca_std = summarize_correlations(series, latent, windows)
        feature_results.append(feature_mean)
        pca_results.append(pca_mean)
        for row, window in enumerate(windows):
            n_windows = latent.shape[0] - window + 1
            for col, target in enumerate(BOTTLENECK_LABELS):
                all_rows.append({
                    "variable": variable,
                    "group": group,
                    "target": target,
                    "target_type": "bottleneck_feature",
                    "rolling_window": window,
                    "n_windows": n_windows,
                    "mean_spatial_pearson_r": feature_mean[row, col],
                    "std_spatial_pearson_r": feature_std[row, col],
                    "mean_test_r2": float(np.nanmean(r2)),
                })
            for col, target in enumerate(PCA_LABELS):
                all_rows.append({
                    "variable": variable,
                    "group": group,
                    "target": target,
                    "target_type": "bottleneck_pca",
                    "rolling_window": window,
                    "n_windows": n_windows,
                    "mean_spatial_pearson_r": pca_mean[row, col],
                    "std_spatial_pearson_r": pca_std[row, col],
                    "mean_test_r2": float(np.nanmean(r2)),
                })
        plot_variable(
            output_dir / f"temporal_average_correlation_{slugify(variable)}.png",
            variable,
            group,
            label,
            windows,
            feature_mean,
            feature_std,
            pca_mean,
            pca_std,
            pca_ratio,
            latent.shape[0],
            args.dpi,
        )

    with (output_dir / "temporal_average_correlation_summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "variable",
            "group",
            "target",
            "target_type",
            "rolling_window",
            "n_windows",
            "mean_spatial_pearson_r",
            "std_spatial_pearson_r",
            "mean_test_r2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    np.savez_compressed(
        output_dir / "temporal_average_correlation_results.npz",
        sample_index=sample_index.astype(np.int64),
        test_r2=r2.astype(np.float64),
        rolling_windows=np.asarray(windows, dtype=np.int64),
        variables=np.asarray(args.variables),
        bottleneck_labels=np.asarray(BOTTLENECK_LABELS),
        pca_labels=np.asarray(PCA_LABELS),
        mean_pca_explained_variance_ratio=pca_ratio.astype(np.float64),
        feature_correlations=np.stack(feature_results, axis=0).astype(np.float64),
        pca_correlations=np.stack(pca_results, axis=0).astype(np.float64),
    )
    print(f"Saved temporal-average correlation figures under {output_dir}")


if __name__ == "__main__":
    main()
