#!/usr/bin/env python3
"""Plot the MLE parameter-tuning results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TUNING_DIR = PROJECT_ROOT / "analysis/fk_mle/mle_tuning"
CE_CSV = TUNING_DIR / "Ce_base_case_calculation/ce_random_search_history.csv"
LFMIN_CSV = TUNING_DIR / "LfMin/lfmin_random_search_history.csv"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/mle_parameterization/mle_tuning_history.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#777777",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_xy(path: Path, x_key: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x.append(float(row[x_key]))
            y.append(float(row["mean_per_snapshot_r2"]))
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def main() -> None:
    ce, ce_r2 = read_xy(CE_CSV, "C_e")
    lfmin, lfmin_r2 = read_xy(LFMIN_CSV, "LfMin")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False, constrained_layout=True)

    axes[0].scatter(ce, ce_r2, s=26, color="#2C6688", alpha=0.72, linewidths=0, zorder=5)
    axes[0].set_title(r"$C_e$ sensitivity", fontsize=10, pad=6)
    axes[0].set_xlabel(r"$C_e$", fontsize=9)
    axes[0].set_ylabel(r"Mean per-snapshot $R^2$", fontsize=9)

    axes[1].plot(lfmin, lfmin_r2, color="#B84D3E", linewidth=1.2, alpha=0.75, zorder=3)
    axes[1].scatter(lfmin, lfmin_r2, s=32, color="#B84D3E", edgecolor="white", linewidth=0.5, zorder=5)
    axes[1].set_xscale("log")
    axes[1].set_xlim(100, max(2500.0, float(np.nanmax(lfmin)) * 1.15))
    y_pad = max((float(np.nanmax(lfmin_r2)) - float(np.nanmin(lfmin_r2))) * 0.18, 1.0e-5)
    axes[1].set_ylim(float(np.nanmin(lfmin_r2)) - y_pad, float(np.nanmax(lfmin_r2)) + y_pad)
    axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    axes[1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    axes[1].set_title(r"$L_{f,\min}$ sensitivity", fontsize=10, pad=6)
    axes[1].set_xlabel(r"$L_{f,\min}$ (m)", fontsize=9)
    axes[1].set_ylabel(r"Mean per-snapshot $R^2$", fontsize=9)

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.7, zorder=0)
        ax.tick_params(labelsize=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
