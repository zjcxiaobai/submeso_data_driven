#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FLUXNET_CSV = Path(__file__).with_name("submesofluxnet_spatial_corr.csv")
MLE_CSV = Path(__file__).with_name("mle_spatial_corr.csv")
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/model_skill/spatial_corr/submesofluxnet_mle_spatial_corr.png"

PALETTE = {
    "fluxnet": "#245E8A",
    "mle": "#B55345",
    "neutral_dark": "#2F2F2F",
    "neutral_mid": "#7A7A7A",
    "neutral_light": "#E3E3E3",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["neutral_dark"],
        "xtick.color": PALETTE["neutral_dark"],
        "ytick.color": PALETTE["neutral_dark"],
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def read_corr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing spatial-correlation CSV: {path}")
    sample: list[int] = []
    corr: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "spatial_corr"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            sample.append(int(row["sample_index"]))
            corr.append(float(row["spatial_corr"]))
    sample_arr = np.asarray(sample, dtype=np.int64)
    corr_arr = np.asarray(corr, dtype=np.float64)
    order = np.argsort(sample_arr)
    return sample_arr[order], corr_arr[order]


def align_series(
    flux_sample: np.ndarray,
    flux_corr: np.ndarray,
    mle_sample: np.ndarray,
    mle_corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if flux_sample.shape != mle_sample.shape or not np.array_equal(flux_sample, mle_sample):
        flux_set = set(map(int, flux_sample))
        mle_set = set(map(int, mle_sample))
        raise ValueError(
            "SubmesoFluxNet and MLE spatial-correlation sample_index values do not match exactly: "
            f"fluxnet_rows={flux_sample.size}, mle_rows={mle_sample.size}, "
            f"common={len(flux_set & mle_set)}"
        )
    return flux_sample, flux_corr, mle_corr


def rolling_median(values: np.ndarray, window: int = 81) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    smooth = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        lo = max(0, idx - half)
        hi = min(values.size, idx + half + 1)
        smooth[idx] = np.nanmedian(values[lo:hi])
    return smooth


def draw_figure(
    sample: np.ndarray,
    flux_corr: np.ndarray,
    mle_corr: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 2.35), constrained_layout=True)
    x = np.arange(sample.size, dtype=np.int64)

    flux_smooth = rolling_median(flux_corr)
    mle_smooth = rolling_median(mle_corr)
    flux_mean = float(np.nanmean(flux_corr))
    mle_mean = float(np.nanmean(mle_corr))

    ax.plot(x, flux_corr, color=PALETTE["fluxnet"], linewidth=0.45, alpha=0.18, zorder=1)
    ax.plot(x, mle_corr, color=PALETTE["mle"], linewidth=0.45, alpha=0.18, zorder=1)
    ax.plot(
        x,
        flux_smooth,
        color=PALETTE["fluxnet"],
        linewidth=1.35,
        label=fr"SubmesoFluxNet, mean $\rho$={flux_mean:.3f}",
        zorder=3,
    )
    ax.plot(
        x,
        mle_smooth,
        color=PALETTE["mle"],
        linewidth=1.35,
        label=fr"MLE parameterization, mean $\rho$={mle_mean:.3f}",
        zorder=3,
    )

    ax.axhline(flux_mean, color=PALETTE["fluxnet"], linewidth=0.75, linestyle=(0, (3, 2)), alpha=0.75)
    ax.axhline(mle_mean, color=PALETTE["mle"], linewidth=0.75, linestyle=(0, (3, 2)), alpha=0.75)

    all_values = np.concatenate([flux_corr[np.isfinite(flux_corr)], mle_corr[np.isfinite(mle_corr)]])
    ymin = max(-1.0, float(np.nanpercentile(all_values, 1)) - 0.04)
    ymax = min(1.0, float(np.nanpercentile(all_values, 99)) + 0.035)
    ax.set_xlim(0, int(x.max()))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Held-out test sample number")
    ax.set_ylabel(r"Spatial correlation, $\rho_n$")
    ax.set_title("Spatial pattern correlation", fontsize=8.2, pad=4)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="center",
        bbox_to_anchor=(0.7, 0.65),
        fontsize=6.7,
        handlelength=1.8,
        borderaxespad=0.0,
        frameon=False,
    )

    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])
    ax.tick_params(labelsize=6.8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    flux_sample, flux_corr = read_corr(FLUXNET_CSV)
    mle_sample, mle_corr = read_corr(MLE_CSV)
    sample, flux_corr, mle_corr = align_series(flux_sample, flux_corr, mle_sample, mle_corr)
    draw_figure(sample, flux_corr, mle_corr)

    print(f"Saved {OUTPUT}")
    print(f"SubmesoFluxNet mean spatial corr: {np.nanmean(flux_corr):.6f}")
    print(f"MLE parameterization mean spatial corr: {np.nanmean(mle_corr):.6f}")


if __name__ == "__main__":
    main()
