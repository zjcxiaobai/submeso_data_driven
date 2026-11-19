#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "analysis/time_series_r2_fk_submesofluxnet"
DEFAULT_R2 = PROJECT_DIR / "results/final_model/test_r2_series.csv"

PALETTE = {
    "main": "#1F5A85",
    "smooth": "#163B57",
    "fill": "#DCE8F2",
    "low": "#B64A3B",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-snapshot test R2 for the final model_analysis temporal bottleneck model."
    )
    parser.add_argument("--r2-series", type=Path, default=DEFAULT_R2)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ANALYSIS_DIR / "test_r2_time_series",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ANALYSIS_DIR / "test_r2_time_series_summary.txt",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=ANALYSIS_DIR / "test_r2_time_series_stats.csv",
    )
    parser.add_argument("--rolling-window", type=int, default=81)
    parser.add_argument("--label", default="SubmesoFluxNet")
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


def read_r2_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing R2 CSV: {path}")

    test_position: list[int] = []
    sample_index: list[int] = []
    r2: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"test_position", "sample_index", "r2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            test_position.append(int(row["test_position"]))
            sample_index.append(int(row["sample_index"]))
            r2.append(float(row["r2"]))

    return (
        np.asarray(test_position, dtype=np.int64),
        np.asarray(sample_index, dtype=np.int64),
        np.asarray(r2, dtype=np.float64),
    )


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


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("R2 series contains no finite values")
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


def lowest_window(sample_index: np.ndarray, r2: np.ndarray, window: int) -> tuple[int, int, float]:
    if window < 1:
        raise ValueError("rolling window must be positive")
    if window > r2.size:
        window = r2.size
    means = np.array([np.nanmean(r2[start : start + window]) for start in range(r2.size - window + 1)])
    start = int(np.nanargmin(means))
    end = start + window - 1
    return int(sample_index[start]), int(sample_index[end]), float(means[start])


def write_stats_csv(
    path: Path,
    summary: dict[str, float | int],
    low_start: int,
    low_end: int,
    low_mean: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})
        writer.writerow({"metric": "lowest_window_sample_start", "value": low_start})
        writer.writerow({"metric": "lowest_window_sample_end", "value": low_end})
        writer.writerow({"metric": "lowest_window_mean_r2", "value": low_mean})


def write_summary(
    path: Path,
    summary: dict[str, float | int],
    sample_min: int,
    sample_max: int,
    low_start: int,
    low_end: int,
    low_mean: float,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"{label} per-snapshot test R2\n")
        handle.write("source_model: final model_analysis temporal bottleneck, upper_40x20, rho replaces speed\n")
        handle.write(f"sample_index_range: {sample_min}-{sample_max}\n")
        for key, value in summary.items():
            if isinstance(value, int):
                handle.write(f"{key}: {value}\n")
            else:
                handle.write(f"{key}: {value:.10g}\n")
        handle.write(f"lowest_window_sample_start: {low_start}\n")
        handle.write(f"lowest_window_sample_end: {low_end}\n")
        handle.write(f"lowest_window_mean_r2: {low_mean:.10g}\n")


def draw_figure(
    sample_index: np.ndarray,
    r2: np.ndarray,
    smooth: np.ndarray,
    summary: dict[str, float | int],
    low_start: int,
    low_end: int,
    output_prefix: Path,
    label: str,
) -> None:
    apply_publication_style()
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
        label=label,
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

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _, sample_index, r2 = read_r2_series(args.r2_series)
    order = np.argsort(sample_index)
    sample_index = sample_index[order]
    r2 = r2[order]
    summary = finite_summary(r2)
    smooth = rolling_median(r2, args.rolling_window)
    low_start, low_end, low_mean = lowest_window(sample_index, r2, args.rolling_window)

    draw_figure(sample_index, r2, smooth, summary, low_start, low_end, args.output_prefix, args.label)
    write_stats_csv(args.stats_csv, summary, low_start, low_end, low_mean)
    write_summary(
        args.summary_output,
        summary,
        int(sample_index.min()),
        int(sample_index.max()),
        low_start,
        low_end,
        low_mean,
        args.label,
    )

    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Mean snapshot R2: {summary['mean']:.6f}", flush=True)
    print(f"SD R2: {summary['std']:.6f}", flush=True)
    print(f"Min R2: {summary['min']:.6f}", flush=True)
    print(f"Max R2: {summary['max']:.6f}", flush=True)


if __name__ == "__main__":
    main()
