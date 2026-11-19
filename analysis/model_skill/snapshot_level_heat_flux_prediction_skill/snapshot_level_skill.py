#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_DIR / "results/model/test_predictions_denorm.npz"
OUTPUT_PATH = (
    PROJECT_DIR
    / "results/analysis_outputs/snapshot_level_heat_flux_prediction_skill/test_snapshot_skill.png"
)
ROLLING_WINDOW = 81

PALETTE = {
    "main": "#1F5A85",
    "smooth": "#163B57",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["axes.labelcolor"] = PALETTE["neutral_dark"]
mpl.rcParams["xtick.color"] = PALETTE["neutral_dark"]
mpl.rcParams["ytick.color"] = PALETTE["neutral_dark"]
mpl.rcParams["xtick.major.width"] = 0.75
mpl.rcParams["ytick.major.width"] = 0.75
mpl.rcParams["xtick.major.size"] = 2.8
mpl.rcParams["ytick.major.size"] = 2.8
mpl.rcParams["legend.frameon"] = False


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    smooth = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        start = max(0, idx - half)
        stop = min(values.size, idx + half + 1)
        smooth[idx] = np.nanmedian(values[start:stop])
    return smooth


def finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def draw_figure(
    sample_index: np.ndarray,
    r2: np.ndarray,
    smooth: np.ndarray,
    summary: dict[str, float],
) -> None:
    order = np.argsort(sample_index)
    sample_sorted = sample_index[order]
    r2_sorted = r2[order]
    smooth_sorted = smooth[order]

    fig, (ax_ts, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(5.6, 2.55),
        width_ratios=(3.35, 1.05),
        constrained_layout=True,
    )

    ax_ts.plot(sample_sorted, r2_sorted, color=PALETTE["main"], linewidth=0.55, alpha=0.30)
    ax_ts.plot(
        sample_sorted,
        smooth_sorted,
        color=PALETTE["smooth"],
        linewidth=1.35,
        label="SubmesoFluxNet",
    )
    ax_ts.set_xlim(int(sample_sorted.min()), int(sample_sorted.max()))
    ymin = max(-0.1, float(np.nanmin(r2_sorted)) - 0.04)
    ymax = min(1.0, float(np.nanmax(r2_sorted)) + 0.04)
    ax_ts.set_ylim(ymin, ymax)
    ax_ts.set_xlabel("Held-out sample index")
    ax_ts.set_ylabel(r"Snapshot-level $R^2$")
    ax_ts.set_title(r" SubmesoFluxNet Test $R^2$", fontsize=7.5, pad=4)
    ax_ts.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.5)
    stats_text = (
        f"Mean snapshot R² = {float(summary['mean']):.3f}\n"
        f"SD = {float(summary['std']):.3f}\n"
        f"Min = {float(summary['min']):.3f}\n"
        f"Max = {float(summary['max']):.3f}"
    )
    ax_ts.text(
        0.78,
        0.045,
        stats_text,
        transform=ax_ts.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=PALETTE["neutral_dark"],
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": PALETTE["neutral_light"],
            "linewidth": 0.55,
            "alpha": 0.88,
        },
    )
    ax_ts.legend(loc="lower left", fontsize=6.4, handlelength=1.5)

    finite = r2_sorted[np.isfinite(r2_sorted)]
    bins = np.linspace(float(np.min(finite)) - 0.02, float(np.max(finite)) + 0.02, 24)
    ax_hist.hist(finite, bins=bins, orientation="horizontal", color=PALETTE["main"], alpha=0.58)
    ax_hist.axhline(float(summary["mean"]), color=PALETTE["smooth"], linewidth=0.9)
    ax_hist.set_ylim(ax_ts.get_ylim())
    ax_hist.set_xlabel("Samples")
    ax_hist.set_title("Distribution", fontsize=7.5, pad=4)
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.42)

    for ax in (ax_ts, ax_hist):
        ax.spines["left"].set_color(PALETTE["neutral_mid"])
        ax.spines["bottom"].set_color(PALETTE["neutral_mid"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with np.load(DATA_PATH) as data:
        sample_index = data["sample_index"]
        r2 = data["r2"]

    order = np.argsort(sample_index)
    sample_index = sample_index[order]
    r2 = r2[order]
    summary = finite_summary(r2)
    smooth = rolling_median(r2, ROLLING_WINDOW)
    draw_figure(sample_index, r2, smooth, summary)

    print(f"Saved {OUTPUT_PATH}", flush=True)
    print(f"Mean snapshot R2: {summary['mean']:.6f}", flush=True)
    print(f"SD R2: {summary['std']:.6f}", flush=True)
    print(f"Min R2: {summary['min']:.6f}", flush=True)
    print(f"Max R2: {summary['max']:.6f}", flush=True)


if __name__ == "__main__":
    main()
