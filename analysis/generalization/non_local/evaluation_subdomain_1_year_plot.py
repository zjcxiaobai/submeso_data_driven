#!/usr/bin/env python3
"""Plot a selected one-year lower-domain evaluation window."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
R2_SERIES = HERE / "upper_model_on_lower_40x20_model_analysis_r2_series.csv"
PREDICTIONS = HERE / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz"
MLE_PREDICTIONS = HERE / "evaluation_subdomain_fk_prediction.npz"
UPPER_TRUTH = PROJECT_ROOT / "data/processed/upper/y.npy"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/generalization/non_local/evaluation_subdomain_1_year_window.png"

WINDOW = 730
SMOOTH_WINDOW = 41
SNAPSHOT_SEED = 42

PALETTE = {
    "line": "#245E8A",
    "smooth": "#B55345",
    "mle": "#4F8A5B",
    "mean": "#2F2F2F",
    "neutral_dark": "#2F2F2F",
    "neutral_light": "#E3E3E3",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def load_r2_series() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(R2_SERIES, delimiter=",", names=True)
    return data["sample_index"].astype(int), data["time_index"].astype(int), data["r2"]


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or values.size < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        lo = max(0, idx - half)
        hi = min(values.size, idx + half + 1)
        out[idx] = np.nanmedian(values[lo:hi])
    return out


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction.reshape(len(prediction), -1)
    target = target.reshape(len(target), -1)
    sse = np.sum((prediction - target) ** 2, axis=1)
    sst = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


def load_mle_r2(window_time_index: np.ndarray) -> np.ndarray:
    with np.load(PREDICTIONS) as data:
        target_time = np.asarray(data["time_index"], dtype=np.int64)
        target = np.asarray(data["target"], dtype=np.float32)
    with np.load(MLE_PREDICTIONS) as data:
        mle_time = np.asarray(data["time_index"], dtype=np.int64)
        mle_prediction = np.asarray(data["prediction"], dtype=np.float32)

    target_lookup = {int(time): position for position, time in enumerate(target_time)}
    mle_lookup = {int(time): position for position, time in enumerate(mle_time)}
    pred = np.stack([mle_prediction[mle_lookup[int(time)]] for time in window_time_index])
    truth = np.stack([target[target_lookup[int(t)]] for t in window_time_index], axis=0)
    return sample_r2(pred, truth)


def select_window(r2: np.ndarray) -> tuple[int, int, float]:
    means = np.asarray([np.nanmean(r2[start : start + WINDOW]) for start in range(len(r2) - WINDOW + 1)])
    start = int(np.nanargmax(means))
    return start, start + WINDOW, float(means[start])


def load_domain_map(sample: int) -> np.ndarray:
    with np.load(PREDICTIONS) as data:
        lower_sample = np.asarray(data["sample_index"], dtype=np.int64)
        lower_target = np.asarray(data["target"], dtype=np.float32)
    lower_position = np.flatnonzero(lower_sample == sample)[0]
    lower = lower_target[lower_position, 0]
    upper = np.load(UPPER_TRUTH, mmap_mode="r")[sample, 0]
    return np.vstack([lower, upper])


def draw_figure(
    sample_index: np.ndarray,
    r2: np.ndarray,
    start: int,
    stop: int,
    mean_r2: float,
    mle_r2: np.ndarray,
    domain_map: np.ndarray,
) -> None:
    title_fs = 10.5
    axis_label_fs = 10.5
    tick_fs = 7.8
    legend_fs = 8.5
    annotation_fs = 8.5
    cbar_label_fs = 8.5
    x = sample_index[start:stop]
    y = r2[start:stop]
    y_smooth = rolling_median(y, SMOOTH_WINDOW)
    mle_smooth = rolling_median(mle_r2, SMOOTH_WINDOW)

    fig = plt.figure(figsize=(9.8, 3.35))
    ax = fig.add_axes([0.065, 0.16, 0.59, 0.72])
    ax_map = fig.add_axes([0.735, 0.18, 0.18, 0.64])
    cax = fig.add_axes([0.945, 0.18, 0.016, 0.64])

    fluxnet_label = rf"SubmesoFluxNet, mean $R^2$={mean_r2:.3f}"
    ax.plot(x, y, color=PALETTE["line"], linewidth=0.55, alpha=0.42)
    ax.plot(x, y_smooth, color=PALETTE["smooth"], linewidth=1.35, label=fluxnet_label)
    mle_label = rf"MLE parameterization, mean $R^2$={float(np.nanmean(mle_r2)):.3f}"
    ax.plot(x, mle_r2, color=PALETTE["mle"], linewidth=0.55, alpha=0.30)
    ax.plot(x, mle_smooth, color=PALETTE["mle"], linewidth=1.25, label=mle_label)
    ax.axhline(mean_r2, color=PALETTE["mean"], linewidth=0.85, linestyle=(0, (3, 2)))
    ax.axhline(float(np.nanmean(mle_r2)), color=PALETTE["mle"], linewidth=0.85, linestyle=(0, (3, 2)), alpha=0.9)

    finite = np.concatenate([y[np.isfinite(y)], mle_r2[np.isfinite(mle_r2)]])
    low = float(np.nanpercentile(finite, 1))
    high = float(np.nanpercentile(finite, 99))
    pad = max(0.06 * (high - low), 0.04)
    ax.set_xlim(int(x[0]), int(x[-1]))
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Evaluation subdomain data index", fontsize=axis_label_fs)
    ax.set_ylabel(r"Snapshot $R^2$", fontsize=axis_label_fs)
    ax.set_title(r"Non-local $R^2$ vs time", fontsize=title_fs, pad=5)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.65)
    ax.tick_params(labelsize=tick_fs, length=2.5, width=0.7)
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.43, 0.18),
        fontsize=legend_fs,
        handlelength=1.8,
        borderaxespad=0.0,
    )

    vmax = 60.0
    image = ax_map.imshow(domain_map, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_map.axhline(19.5, color="#5F5F5F", linewidth=0.85)
    ax_map.set_xlim(-0.5, 39.5)
    ax_map.set_ylim(-0.5, 39.5)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xticks([0, 20, 40])
    ax_map.set_yticks([0, 20, 40])
    ax_map.tick_params(labelsize=tick_fs, length=2.5, width=0.7)
    ax_map.set_xlabel("x index", fontsize=axis_label_fs)
    ax_map.set_ylabel("y index", fontsize=axis_label_fs)
    ax_map.set_title("Submesoscale VHF", fontsize=title_fs, pad=5)
    ax_map.text(
        -0.05,
        0.75,
        "Training\nsubdomain",
        transform=ax_map.transAxes,
        ha="right",
        va="center",
        fontsize=annotation_fs,
        color=PALETTE["neutral_dark"],
    )
    ax_map.text(
        -0.05,
        0.25,
        "Evaluation\nSubdomain",
        transform=ax_map.transAxes,
        ha="right",
        va="center",
        fontsize=annotation_fs,
        color=PALETTE["neutral_dark"],
    )

    cbar = fig.colorbar(image, cax=cax, ticks=[-60, -40, -20, 0, 20, 40, 60])
    cbar.ax.tick_params(labelsize=tick_fs, length=2.2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(r"W m$^{-2}$", fontsize=cbar_label_fs, color=PALETTE["neutral_dark"], labelpad=4)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def main() -> None:
    sample_index, time_index, r2 = load_r2_series()

    start, stop, mean_r2 = select_window(r2)
    window_r2 = r2[start:stop]
    mle_r2 = load_mle_r2(time_index[start:stop])

    low, high = np.nanquantile(window_r2, [0.4, 0.6])
    candidates = np.flatnonzero((window_r2 >= low) & (window_r2 <= high))
    selected_snapshot_position = start + int(np.random.default_rng(SNAPSHOT_SEED).choice(candidates))
    selected_sample = int(sample_index[selected_snapshot_position])
    domain_map = load_domain_map(selected_sample)

    draw_figure(sample_index, r2, start, stop, mean_r2, mle_r2, domain_map)
    print(f"Selected window positions: {start}-{stop - 1}")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
