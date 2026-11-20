#!/usr/bin/env python3
"""Plot bottleneck–physics correlations across temporal averaging windows."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PREDICTIONS = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
PHYSICS = HERE / "physics_variable_test_series.npz"
RESULTS = HERE / "temporal_average_correlation_results.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/interpretability/bottle_feature/temporal_average_correlation/temporal_average_correlation_features_combined.png"

VARIABLES = [
    ("shear_strain", r"$S_s$", "#0072B2", "o", "-"),
    ("T", r"$T$", "#D55E00", "s", "-"),
    ("Rossby_number", r"$Ro$", "#009E73", "^", "-"),
    ("W", r"$w$", "#CC79A7", "D", "-"),
    ("gradT", r"$|\nabla_h T|$", "#E69F00", "v", "--"),
    ("frontogenesis_FT", r"$F_T$", "#56B4E9", "P", "--"),
    ("normal_strain", r"$S_n$", "#333333", "X", "--"),
]
WINDOWS = np.asarray([1, 3, 7, 15, 30, 60, 90, 180, 365, 712])
WINDOW_DAYS = WINDOWS * 0.5


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    }
)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(values, axis=0, dtype=np.float64)
    cumulative[window:] -= cumulative[:-window]
    return cumulative[window - 1 :] / window


def spatial_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    correlations = np.full((len(a), 4), np.nan)
    for sample in range(len(a)):
        x = a[sample].ravel()
        for feature in range(4):
            y = b[sample, feature].ravel()
            valid = np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid] - np.mean(x[valid])
            y_valid = y[valid] - np.mean(y[valid])
            denominator = np.sqrt(np.sum(x_valid**2) * np.sum(y_valid**2))
            if denominator > 0:
                correlations[sample, feature] = np.sum(x_valid * y_valid) / denominator
    return correlations


def calculate_results() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = np.asarray([name for name, *_ in VARIABLES])
    with np.load(PREDICTIONS) as data:
        model_samples = data["sample_index"].astype(int)
        latent = data["latent_features"].astype(float)
    with np.load(PHYSICS) as data:
        physics_samples = data["sample_index"].astype(int)
        physics = {name: data[name].astype(float) for name in names}

    positions = {int(sample): position for position, sample in enumerate(model_samples)}
    latent = latent[[positions[int(sample)] for sample in physics_samples]]
    correlations = np.empty((len(names), len(WINDOWS), 4))

    for window_index, window in enumerate(WINDOWS):
        latent_mean = rolling_mean(latent, int(window))
        for variable_index, name in enumerate(names):
            physics_mean = rolling_mean(physics[name], int(window))
            values = spatial_correlation(physics_mean, latent_mean)
            correlations[variable_index, window_index] = np.nanmean(np.abs(values), axis=0)

    np.savez_compressed(
        RESULTS,
        variables=names,
        rolling_windows=WINDOWS,
        window_days=WINDOW_DAYS,
        feature_correlations=correlations,
    )
    return names, WINDOW_DAYS, correlations


def load_results() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not RESULTS.exists():
        return calculate_results()
    with np.load(RESULTS) as data:
        return (
            data["variables"].astype(str),
            data["window_days"].astype(float),
            data["feature_correlations"].astype(float),
        )


def plot_figure(names: np.ndarray, days: np.ndarray, correlations: np.ndarray) -> None:
    index = {name: position for position, name in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.095, right=0.765, bottom=0.105, top=0.93, wspace=0.17, hspace=0.22)
    handles = []
    labels = []

    for feature, axis in enumerate(axes.ravel()):
        for name, label, color, marker, linestyle in VARIABLES:
            line, = axis.plot(
                days,
                correlations[index[name], :, feature],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.15,
                markersize=3.2,
            )
            if feature == 0:
                handles.append(line)
                labels.append(label)
        axis.set_xscale("log")
        axis.set_xticks(days, labels=[f"{day:g}" for day in days])
        axis.minorticks_off()
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.42)
        axis.set_title(f"{'abcd'[feature]}, Feature {feature + 1}", loc="left", fontsize=8.8, pad=3)
        axis.tick_params(labelsize=6.7, length=2.3, width=0.65)

    for axis in axes[:, 0]:
        axis.set_ylabel(r"Average spatial Pearson $|r|$", fontsize=7.6)
    for axis in axes[1]:
        axis.set_xlabel("Temporal averaging window (model days)", fontsize=7.6)
    fig.legend(
        handles,
        labels,
        title="Physical diagnostics",
        loc="center left",
        bbox_to_anchor=(0.785, 0.52),
        fontsize=7.2,
        title_fontsize=7.5,
        handlelength=2.2,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


def main() -> None:
    plot_figure(*load_results())


if __name__ == "__main__":
    main()
