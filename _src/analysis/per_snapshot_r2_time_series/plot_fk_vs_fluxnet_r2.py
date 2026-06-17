#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "analysis/per_snapshot_r2_time_series"
FK_MLE_DIR = PROJECT_DIR / "analysis/fk_mle"

PALETTE = {
    "fluxnet": "#1F5A85",
    "fk": "#B64A3B",
    "fluxnet_fill": "#DCE8F2",
    "fk_fill": "#F1D8D2",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FK and SubmesoFluxNet per-snapshot R2 together.")
    parser.add_argument(
        "--fluxnet-r2",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_r2_series.csv",
        help="SubmesoFluxNet R2 CSV from evaluate.py.",
    )
    parser.add_argument(
        "--fk-r2",
        type=Path,
        default=FK_MLE_DIR / "fk_test_Ce2p13062237_r2_series.csv",
        help="Tuned FK R2 CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ANALYSIS_DIR / "fk_vs_submesofluxnet_per_snapshot_r2",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ANALYSIS_DIR / "fk_vs_submesofluxnet_per_snapshot_r2_summary.txt",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=ANALYSIS_DIR / "fk_vs_submesofluxnet_per_snapshot_r2_stats.csv",
    )
    parser.add_argument("--rolling-window", type=int, default=81)
    return parser.parse_args()


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
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
    fk_sample: np.ndarray,
    fk_r2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux_lookup = {int(s): n for n, s in enumerate(fluxnet_sample)}
    fk_lookup = {int(s): n for n, s in enumerate(fk_sample)}
    common = np.asarray(sorted(set(flux_lookup).intersection(fk_lookup)), dtype=np.int64)
    if common.size == 0:
        raise ValueError("No overlapping sample_index values between FK and SubmesoFluxNet R2 CSV files.")
    flux = np.asarray([fluxnet_r2[flux_lookup[int(s)]] for s in common], dtype=np.float64)
    fk = np.asarray([fk_r2[fk_lookup[int(s)]] for s in common], dtype=np.float64)
    return common, flux, fk


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


def summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def write_stats(path: Path, flux_summary: dict[str, float | int], fk_summary: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "value"])
        writer.writeheader()
        for method, vals in (("SubmesoFluxNet", flux_summary), ("FK", fk_summary)):
            for key, value in vals.items():
                writer.writerow({"method": method, "metric": key, "value": value})


def write_summary(
    path: Path,
    sample: np.ndarray,
    flux_summary: dict[str, float | int],
    fk_summary: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("FK versus SubmesoFluxNet per-snapshot R2 comparison\n")
        handle.write(f"sample_index_range: {int(sample.min())}-{int(sample.max())}\n")
        handle.write(f"n_common_snapshots: {sample.size}\n")
        for label, vals in (("SubmesoFluxNet", flux_summary), ("FK", fk_summary)):
            handle.write(f"\n{label}\n")
            for key, value in vals.items():
                if isinstance(value, int):
                    handle.write(f"{key}: {value}\n")
                else:
                    handle.write(f"{key}: {value:.10g}\n")


def draw_figure(
    sample: np.ndarray,
    fluxnet_r2: np.ndarray,
    fk_r2: np.ndarray,
    flux_summary: dict[str, float | int],
    fk_summary: dict[str, float | int],
    rolling_window: int,
    output_prefix: Path,
) -> None:
    apply_publication_style()
    fig, (ax_ts, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(5.6, 2.55),
        width_ratios=(3.35, 1.05),
        constrained_layout=True,
    )

    flux_smooth = rolling_median(fluxnet_r2, rolling_window)
    fk_smooth = rolling_median(fk_r2, rolling_window)

    ax_ts.plot(sample, fluxnet_r2, color=PALETTE["fluxnet"], linewidth=0.55, alpha=0.30, zorder=1)
    ax_ts.plot(sample, fk_r2, color=PALETTE["fk"], linewidth=0.55, alpha=0.30, zorder=1)
    ax_ts.plot(
        sample,
        flux_smooth,
        color=PALETTE["fluxnet"],
        linewidth=1.35,
        label=fr"SubmesoFluxNet, mean $R^2$={float(flux_summary['mean']):.3f}",
        zorder=3,
    )
    ax_ts.plot(
        sample,
        fk_smooth,
        color=PALETTE["fk"],
        linewidth=1.35,
        label=fr"FK, mean $R^2$={float(fk_summary['mean']):.3f}",
        zorder=4,
    )
    ax_ts.set_xlim(int(sample.min()), int(sample.max()))
    ymin = min(float(np.nanmin(fluxnet_r2)), float(np.nanmin(fk_r2))) - 0.04
    ymax = min(1.0, max(float(np.nanmax(fluxnet_r2)), float(np.nanmax(fk_r2))) + 0.04)
    ax_ts.set_ylim(ymin, ymax)
    ax_ts.set_xlabel("Held-out test data index")
    ax_ts.set_ylabel(r"Per-snapshot $R^2$")
    ax_ts.set_title(r"FK versus SubmesoFluxNet test $R^2$", fontsize=7.5, pad=4)
    ax_ts.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.5)
    ax_ts.legend(loc="lower left", fontsize=6.4, handlelength=1.5)

    finite_flux = fluxnet_r2[np.isfinite(fluxnet_r2)]
    finite_fk = fk_r2[np.isfinite(fk_r2)]
    low = min(float(np.min(finite_flux)), float(np.min(finite_fk))) - 0.02
    high = max(float(np.max(finite_flux)), float(np.max(finite_fk))) + 0.02
    bins = np.linspace(low, high, 24)
    ax_hist.hist(finite_flux, bins=bins, orientation="horizontal", color=PALETTE["fluxnet"], alpha=0.52, label="SubmesoFluxNet")
    ax_hist.hist(finite_fk, bins=bins, orientation="horizontal", color=PALETTE["fk"], alpha=0.52, label="FK")
    ax_hist.axhline(float(flux_summary["mean"]), color=PALETTE["fluxnet"], linewidth=0.8)
    ax_hist.axhline(float(fk_summary["mean"]), color=PALETTE["fk"], linewidth=0.8)
    ax_hist.set_ylim(ax_ts.get_ylim())
    ax_hist.set_xlabel("Snapshots")
    ax_hist.set_title("Distribution", fontsize=7.5, pad=4)
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.42)

    for ax in (ax_ts, ax_hist):
        ax.spines["left"].set_color(PALETTE["neutral_mid"])
        ax.spines["bottom"].set_color(PALETTE["neutral_mid"])

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png", ".tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(Path(f"{output_prefix}{ext}"), **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    flux_sample, flux_r2 = read_r2_csv(args.fluxnet_r2)
    fk_sample, fk_r2 = read_r2_csv(args.fk_r2)
    sample, flux_aligned, fk_aligned = align_series(flux_sample, flux_r2, fk_sample, fk_r2)
    flux_summary = summary(flux_aligned)
    fk_summary = summary(fk_aligned)

    draw_figure(sample, flux_aligned, fk_aligned, flux_summary, fk_summary, args.rolling_window, args.output_prefix)
    write_stats(args.stats_csv, flux_summary, fk_summary)
    write_summary(args.summary_output, sample, flux_summary, fk_summary)

    for ext in (".svg", ".pdf", ".png", ".tiff"):
        print(f"Saved {args.output_prefix}{ext}", flush=True)
    print(f"Saved {args.stats_csv}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"SubmesoFluxNet mean R2: {float(flux_summary['mean']):.6f}", flush=True)
    print(f"FK mean R2: {float(fk_summary['mean']):.6f}", flush=True)


if __name__ == "__main__":
    main()
