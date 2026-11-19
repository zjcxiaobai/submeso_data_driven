#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLUXNET_R2 = PROJECT_ROOT / "results/model/test_r2_series.csv"
MLE_R2 = PROJECT_ROOT / "analysis/fk_mle/tuned_mle_parameterization/mle_r2_series.csv"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/model_skill/mle_submesofluxnet_comparison/snapshot_skill.png"
ROLLING_WINDOW = 81

PALETTE = {
    "fluxnet": "#1F5A85",
    "mle": "#B64A3B",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "axes.labelcolor": PALETTE["neutral_dark"],
        "xtick.color": PALETTE["neutral_dark"],
        "ytick.color": PALETTE["neutral_dark"],
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_r2_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing R2 CSV: {path}")

    sample_index: list[int] = []
    r2: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "r2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            sample_index.append(int(row["sample_index"]))
            r2.append(float(row["r2"]))

    sample = np.asarray(sample_index, dtype=np.int64)
    values = np.asarray(r2, dtype=np.float64)
    order = np.argsort(sample)
    return sample[order], values[order]


def align_series(
    fluxnet_sample: np.ndarray,
    fluxnet_r2: np.ndarray,
    mle_sample: np.ndarray,
    mle_r2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fluxnet_sample.shape != mle_sample.shape or not np.array_equal(fluxnet_sample, mle_sample):
        flux_set = set(map(int, fluxnet_sample))
        mle_set = set(map(int, mle_sample))
        raise ValueError(
            "SubmesoFluxNet and MLE sample_index values do not match exactly: "
            f"fluxnet_rows={fluxnet_sample.size}, mle_rows={mle_sample.size}, "
            f"common={len(flux_set & mle_set)}"
        )
    return fluxnet_sample, fluxnet_r2, mle_r2


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


def draw_figure(
    sample: np.ndarray,
    fluxnet_r2: np.ndarray,
    mle_r2: np.ndarray,
) -> None:
    flux_smooth = rolling_median(fluxnet_r2, ROLLING_WINDOW)
    mle_smooth = rolling_median(mle_r2, ROLLING_WINDOW)
    flux_mean = float(np.nanmean(fluxnet_r2))
    mle_mean = float(np.nanmean(mle_r2))
    sample_number = np.arange(sample.size, dtype=np.int64)

    fig, (ax, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(5.6, 2.55),
        width_ratios=(3.35, 1.05),
        constrained_layout=True,
    )

    ax.plot(sample_number, fluxnet_r2, color=PALETTE["fluxnet"], linewidth=0.5, alpha=0.25)
    ax.plot(sample_number, mle_r2, color=PALETTE["mle"], linewidth=0.5, alpha=0.25)
    ax.plot(
        sample_number,
        flux_smooth,
        color=PALETTE["fluxnet"],
        linewidth=1.35,
        label=fr"SubmesoFluxNet, mean $R^2$ = {flux_mean:.3f}",
    )
    ax.plot(
        sample_number,
        mle_smooth,
        color=PALETTE["mle"],
        linewidth=1.35,
        label=fr"MLE parameterization, mean $R^2$ = {mle_mean:.3f}",
    )
    ax.set_xlim(0, int(sample_number.max()))
    ymin = min(float(np.nanmin(fluxnet_r2)), float(np.nanmin(mle_r2))) - 0.04
    ymax = min(1.0, max(float(np.nanmax(fluxnet_r2)), float(np.nanmax(mle_r2))) + 0.04)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Held-out test sample")
    ax.set_ylabel(r"Snapshot-level $R^2$")
    ax.set_title(r"MLE parameterization and SubmesoFluxNet test $R^2$", fontsize=7.5, pad=4)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower left", fontsize=6.4, handlelength=1.5)
    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])

    finite_flux = fluxnet_r2[np.isfinite(fluxnet_r2)]
    finite_mle = mle_r2[np.isfinite(mle_r2)]
    low = min(float(np.min(finite_flux)), float(np.min(finite_mle))) - 0.02
    high = max(float(np.max(finite_flux)), float(np.max(finite_mle))) + 0.02
    bins = np.linspace(low, high, 24)

    ax_hist.hist(
        finite_flux,
        bins=bins,
        orientation="horizontal",
        color=PALETTE["fluxnet"],
        alpha=0.52,
    )
    ax_hist.hist(
        finite_mle,
        bins=bins,
        orientation="horizontal",
        color=PALETTE["mle"],
        alpha=0.52,
    )
    ax_hist.axhline(flux_mean, color=PALETTE["fluxnet"], linewidth=0.8)
    ax_hist.axhline(mle_mean, color=PALETTE["mle"], linewidth=0.8)
    ax_hist.set_ylim(ax.get_ylim())
    ax_hist.set_xlabel("Samples")
    ax_hist.set_title("Distribution", fontsize=7.5, pad=4)
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.5)
    ax_hist.spines["left"].set_color(PALETTE["neutral_mid"])
    ax_hist.spines["bottom"].set_color(PALETTE["neutral_mid"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    flux_sample, flux_r2 = read_r2_csv(FLUXNET_R2)
    mle_sample, mle_r2 = read_r2_csv(MLE_R2)
    sample, flux_r2, mle_r2 = align_series(flux_sample, flux_r2, mle_sample, mle_r2)
    draw_figure(sample, flux_r2, mle_r2)
    print(f"Saved {OUTPUT}")
    print(f"SubmesoFluxNet mean R2: {np.nanmean(flux_r2):.6f}")
    print(f"MLE parameterization mean R2: {np.nanmean(mle_r2):.6f}")


if __name__ == "__main__":
    main()
