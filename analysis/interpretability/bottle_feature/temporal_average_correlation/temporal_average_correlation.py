#!/usr/bin/env python3
"""Test how rolling averages across test samples affect physics-bottleneck correlations."""

from __future__ import annotations

import argparse
import csv
import json
import importlib.util
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[4]
SOURCE_SCRIPT = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation/model_analysis_last_decoder_pca/prepare_physics_variables.py"
DEFAULT_X = PROJECT_DIR / "analysis/model_analysis/data/X.npy"
DEFAULT_PREDICTIONS = PROJECT_DIR / "results/final_model/test_predictions_denorm.npz"
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
    ("shear_strain", "Rotation/deformation", r"$S_s$"),
    ("T", "Background state", r"$T$"),
    ("Rossby_number", "Rotation/deformation", r"$Ro$"),
    ("W", "Vertical exchange", r"$w$"),
    ("gradT", "Frontal structure", r"$|\nabla_h T|$"),
    ("frontogenesis_FT", "Frontal structure", r"$F_T$"),
    ("normal_strain", "Rotation/deformation", r"$S_n$"),
]
BOTTLENECK_LABELS = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]


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
    parser.add_argument("--replot-only", action="store_true", help="Plot saved correlations without recalculating them.")
    parser.add_argument("--snapshot-days", type=float, default=None,
                        help="Physical interval between snapshots in model days; specify explicitly.")
    parser.add_argument("--time-index", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/time_index.npy")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-block", type=int, default=30,
                        help="Minimum block length in correlation-series samples; increased to the rolling window.")
    parser.add_argument("--min-blocks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final-window-days", type=float, default=356.0)
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


def spatial_correlations(physics: np.ndarray, latent: np.ndarray) -> np.ndarray:
    result = np.empty((len(physics), 4), dtype=np.float64)
    for sample in range(len(physics)):
        for feature in range(4):
            result[sample, feature] = pearsonr_flat(physics[sample], latent[sample, feature])
    return result


def mean_abs_correlation(values: np.ndarray) -> np.ndarray:
    result = np.full(4, np.nan)
    for feature in range(4):
        finite = values[:, feature][np.isfinite(values[:, feature])]
        if finite.size:
            result[feature] = np.nanmean(np.abs(finite))
    return result


def averaging_windows(requested, final_days, snapshot_days, n_samples):
    if not np.isfinite(final_days) or final_days <= 0:
        raise ValueError("Final averaging window must be finite and positive")
    count = final_days / snapshot_days
    final = int(round(count))
    if final < 1 or not np.isclose(count, final, rtol=0, atol=1e-8):
        raise ValueError("Final window must be an integer number of snapshots")
    if final > n_samples:
        raise ValueError("Final averaging window exceeds the available test series")
    return sorted({int(w) for w in requested if 0 < w <= final} | {final})


def plot_feature_lines(output, values, variables, days, dpi):
    style()
    lookup = {name: (group, label) for name, group, label in VARIABLE_SPECS}
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#333333", "#7B3294"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    linestyles = ["-", "-", "-", "-", "--", "--", "--", "--"]
    styles = {name: (colors[i], markers[i], linestyles[i]) for i, (name, _, _) in enumerate(VARIABLE_SPECS)}
    if values.shape != (len(variables), len(days), 4):
        raise ValueError("Saved correlations do not match variables, windows, and four features")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    abs_values = np.abs(values)

    # Keep the original one-feature-per-file outputs for detailed inspection.
    for feature in range(4):
        fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
        for vi, variable in enumerate(variables):
            rows.append(dict(feature=feature + 1, variable=variable,
                             final_window_days=float(days[-1]),
                             average_absolute_correlation=float(abs_values[vi, -1, feature])))
            color, marker, linestyle = styles[variable]
            ax.plot(days, abs_values[vi, :, feature], color=color, marker=marker,
                    linestyle=linestyle, linewidth=1.3, markersize=4,
                    label=lookup[variable][1])
        ax.set_xscale("log")
        ax.set_xticks(days, labels=[f"{day:g}" for day in days])
        ax.minorticks_off()
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("Temporal averaging window (model days)")
        ax.set_ylabel(r"Average spatial Pearson $|r|$")
        ax.set_title(f"Feature {feature + 1}", loc="left", fontsize=10)
        ax.grid(axis="y", color="#D6D6D6", linewidth=0.45)
        ax.legend(title="Physical diagnostics", loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=8, title_fontsize=8, handlelength=2.5, borderaxespad=0)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"temporal_average_correlation_feature_{feature + 1}.{extension}",
                        dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)

    # Publication-friendly combined panel: four bottleneck features, one shared legend.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.765, bottom=0.105, top=0.93, wspace=0.17, hspace=0.22)
    legend_handles = []
    legend_labels = []
    panel_labels = ["a", "b", "c", "d"]
    for feature, ax in enumerate(axes.ravel()):
        for vi, variable in enumerate(variables):
            color, marker, linestyle = styles[variable]
            line, = ax.plot(days, abs_values[vi, :, feature], color=color, marker=marker,
                            linestyle=linestyle, linewidth=1.15, markersize=3.2,
                            label=lookup[variable][1])
            if feature == 0:
                legend_handles.append(line)
                legend_labels.append(lookup[variable][1])
        ax.set_xscale("log")
        ax.set_xticks(days, labels=[f"{day:g}" for day in days])
        ax.minorticks_off()
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.42)
        ax.set_title(f"{panel_labels[feature]}, Feature {feature + 1}", loc="left", fontsize=8.8, pad=3)
        ax.tick_params(labelsize=6.7, length=2.3, width=0.65)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Average spatial Pearson $|r|$", fontsize=7.6)
    for ax in axes[1, :]:
        ax.set_xlabel("Temporal averaging window (model days)", fontsize=7.6)
    fig.legend(legend_handles, legend_labels, title="Physical diagnostics",
               loc="center left", bbox_to_anchor=(0.785, 0.52), fontsize=7.2,
               title_fontsize=7.5, frameon=False, handlelength=2.2, borderaxespad=0.0)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"temporal_average_correlation_features_combined.{extension}",
                    dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    with (output / "selected_physical_variables_by_feature.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "variable", "final_window_days", "average_absolute_correlation"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.replot_only:
        with np.load(args.output_dir.expanduser() / "bottleneck_physics_heatmaps.npz") as data:
            variables = [str(value) for value in data["variables"]]
            requested = [name for name, _group, _label in VARIABLE_SPECS]
            missing = [name for name in requested if name not in variables]
            if missing:
                raise ValueError(f"Saved results are missing requested variables: {missing}. Run without --replot-only and use --no-reuse-cache.")
            order = [variables.index(name) for name in requested]
            days = np.asarray(data["window_days"], dtype=float)
            if args.snapshot_days is not None:
                if args.snapshot_days <= 0 or not np.isfinite(args.snapshot_days):
                    raise ValueError("--snapshot-days must be finite and positive")
                days = data["rolling_windows"] * args.snapshot_days
            matches = np.flatnonzero(np.isclose(days, args.final_window_days, rtol=0, atol=1e-8))
            if len(matches) != 1:
                raise ValueError(f"Saved results do not contain {args.final_window_days:g} days. Run without --replot-only and specify --snapshot-days to calculate it.")
            window_order = np.flatnonzero(days <= args.final_window_days)
            window_order = window_order[np.argsort(days[window_order])]
            plot_feature_lines(args.output_dir.expanduser(), data["feature_correlations"][order][:, window_order],
                               [variables[i] for i in order], days[window_order], args.dpi)
        print("Saved four feature-wise correlation plots from existing results", flush=True)
        return
    if args.snapshot_days is None or args.snapshot_days <= 0 or not np.isfinite(args.snapshot_days):
        raise ValueError("--snapshot-days must be finite and positive")
    if args.bootstrap_repeats < 100 or args.bootstrap_block < 1 or args.min_blocks < 2:
        raise ValueError("Use at least 100 bootstrap repeats, a positive block length, and at least two blocks")
    requested = set(args.variables)
    lookup = {name: (group, label) for name, group, label in VARIABLE_SPECS}
    if requested - lookup.keys():
        raise ValueError(f"Unknown variables: {sorted(requested - lookup.keys())}")
    variables = [name for name, _, _ in VARIABLE_SPECS if name in requested]
    if not variables:
        raise ValueError("Select at least one variable")
    sample_index, r2, latent = load_predictions(args.predictions.expanduser())
    order = np.argsort(sample_index)
    sample_index, latent = sample_index[order], latent[order]
    if not np.all(np.diff(sample_index) == 1):
        raise ValueError("Rolling means require unique consecutive test samples")
    times = np.load(args.time_index, mmap_mode="r")[sample_index]
    if len(times) < 2 or not np.all(np.diff(times) == times[1] - times[0]) or times[1] <= times[0]:
        raise ValueError("Snapshot timestamps must be uniformly spaced and increasing")
    windows = averaging_windows(args.rolling_windows, args.final_window_days, args.snapshot_days, len(latent))
    if not windows:
        raise ValueError("No valid rolling windows")
    physics = load_or_create_physics_cache(args, sample_index, variables)
    shape = (len(variables), len(windows), 4)
    mean_abs = np.full(shape, np.nan)
    rows = []
    for wi, window in enumerate(windows):
        latent_roll = trailing_rolling_mean(latent, window)
        block = max(args.bootstrap_block, window)
        for vi, variable in enumerate(variables):
            series = np.asarray(physics[variable], dtype=np.float64)
            if series.shape != (len(latent), *latent.shape[-2:]):
                raise ValueError(f"Unexpected shape for {variable}: {series.shape}")
            correlations = spatial_correlations(trailing_rolling_mean(series, window), latent_roll)
            mean_abs[vi, wi] = mean_abs_correlation(correlations)
            for fi, label in enumerate(BOTTLENECK_LABELS):
                rows.append(dict(variable=variable, group=lookup[variable][0], feature=label,
                                 window_snapshots=window, window_days=window * args.snapshot_days,
                                 n_windows=len(correlations),
                                 mean_abs_spatial_pearson_r=mean_abs[vi, wi, fi]))
        print(f"Finished window {window} snapshots ({window * args.snapshot_days:g} days)", flush=True)
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "bottleneck_physics_heatmaps.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "bottleneck_physics_heatmaps.npz", sample_index=sample_index,
                        variables=np.asarray(variables), rolling_windows=np.asarray(windows),
                        window_days=np.asarray(windows) * args.snapshot_days,
                        feature_correlations=mean_abs,
                        statistic=np.asarray("mean_abs_spatial_pearson_r"))
    metadata = dict(statistic="mean absolute spatial Pearson correlation across trailing rolling windows",
                    snapshot_days=args.snapshot_days, time_index_spacing=int(times[1] - times[0]),
                    aggregation="for each temporal averaging window, compute spatial Pearson r for every trailing window and average abs(r)",
                    predictions=str(args.predictions), x_path=str(args.x_path), pca=False)
    (output / "bottleneck_physics_heatmaps_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_feature_lines(output, mean_abs, variables, np.asarray(windows) * args.snapshot_days, args.dpi)
    print(f"Saved four feature-wise correlation plots and statistics under {output}", flush=True)


if __name__ == "__main__":
    main()
