#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]

PALETTE = {
    "line": "#1F5A85",
    "smooth": "#B64A3B",
    "fill": "#DCE8F2",
    "hist": "#87A9C4",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
    "neutral_lighter": "#F2F2F2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Nature-style per-snapshot R2 time series for the no-shuffle overfitting-test model."
    )
    parser.add_argument(
        "--r2-series",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_r2_series.csv",
        help="CSV produced by evaluate.py with columns test_position, sample_index, r2, mse, mae.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_DIR
        / "analysis/per_snapshot_r2_time_series/no_shuffle_overfitting_test_per_snapshot_r2",
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR
        / "analysis/per_snapshot_r2_time_series/no_shuffle_overfitting_test_per_snapshot_r2_summary.txt",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=PROJECT_DIR
        / "analysis/per_snapshot_r2_time_series/no_shuffle_overfitting_test_per_snapshot_r2_stats.csv",
    )
    parser.add_argument("--rolling-window", type=int, default=81, help="Odd rolling median window in snapshots.")
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
        raise FileNotFoundError(f"Missing {path}. Run evaluate.py for the overfitting-test model first.")

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

    if not r2:
        raise ValueError(f"No R2 rows found in {path}")

    position_arr = np.asarray(test_position, dtype=np.int64)
    sample_arr = np.asarray(sample_index, dtype=np.int64)
    r2_arr = np.asarray(r2, dtype=np.float64)
    order = np.argsort(sample_arr)
    return position_arr[order], sample_arr[order], r2_arr[order]


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
        return {
            "count": int(values.size),
            "finite_count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
        }
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
    if window % 2 == 0:
        window += 1
    window = min(window, r2.size)
    if window < 1:
        return int(sample_index[0]), int(sample_index[0]), float("nan")

    means = np.asarray([np.nanmean(r2[start : start + window]) for start in range(r2.size - window + 1)])
    start = int(np.nanargmin(means))
    stop = start + window - 1
    return int(sample_index[start]), int(sample_index[stop]), float(means[start])


def write_stats_csv(path: Path, summary: dict[str, float | int], low_start: int, low_end: int, low_mean: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})
        writer.writerow({"metric": "lowest_rolling_window_start_sample", "value": low_start})
        writer.writerow({"metric": "lowest_rolling_window_end_sample", "value": low_end})
        writer.writerow({"metric": "lowest_rolling_window_mean_r2", "value": low_mean})


def write_summary(
    path: Path,
    summary: dict[str, float | int],
    sample_start: int,
    sample_end: int,
    low_start: int,
    low_end: int,
    low_mean: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("count", "finite_count", "mean", "median", "std", "min", "max", "p05", "p25", "p75", "p95")
    with path.open("w") as handle:
        handle.write("No-shuffle overfitting-test per-snapshot R2 summary\n")
        handle.write(f"sample_index_range: {sample_start}-{sample_end}\n")
        for key in keys:
            value = summary[key]
            if isinstance(value, int):
                handle.write(f"{key}: {value}\n")
            else:
                handle.write(f"{key}: {value:.10g}\n")
        handle.write(
            f"lowest_rolling_window: sample {low_start}-{low_end}, "
            f"mean_r2={low_mean:.10g}\n"
        )


def draw_figure(
    sample_index: np.ndarray,
    r2: np.ndarray,
    smooth: np.ndarray,
    summary: dict[str, float | int],
    low_start: int,
    low_end: int,
    output_prefix: Path,
) -> None:
    apply_publication_style()
    fig, (ax_ts, ax_hist) = plt.subplots(
        1,
        2,
        figsize=(5.25, 2.45),
        width_ratios=(3.4, 1.0),
        constrained_layout=True,
    )

    mean_r2 = float(summary["mean"])
    median_r2 = float(summary["median"])
    p05 = float(summary["p05"])
    p95 = float(summary["p95"])

    x_axis = np.arange(r2.size, dtype=np.int64)
    ax_ts.fill_between(x_axis, p05, p95, color=PALETTE["fill"], alpha=0.36, linewidth=0, zorder=1)
    ax_ts.plot(x_axis, r2, color=PALETTE["line"], linewidth=0.75, alpha=0.62, zorder=2)
    ax_ts.plot(x_axis, smooth, color=PALETTE["smooth"], linewidth=1.35, zorder=4)
    ax_ts.axhline(mean_r2, color=PALETTE["neutral_dark"], linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    ax_ts.set_xlim(0, int(x_axis.max()))
    ax_ts.set_ylim(max(-0.05, float(np.nanmin(r2)) - 0.04), min(1.0, float(np.nanmax(r2)) + 0.04))
    ax_ts.set_xlabel("Held-out test sample number")
    ax_ts.set_ylabel(r"Per-snapshot $R^2$")
    ax_ts.set_title(r"test $R^2$ VS time ", fontsize=7.5, pad=4)
    ax_ts.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.5)
    y_text = ax_ts.get_ylim()[0] + 0.08 * (ax_ts.get_ylim()[1] - ax_ts.get_ylim()[0])
    x_text = 400 if x_axis.min() <= 400 <= x_axis.max() else int(x_axis.min())
    ax_ts.text(
        x_text,
        y_text,
        fr"mean $R^2$={mean_r2:.3f}",
        ha="left",
        va="bottom",
        color=PALETTE["neutral_dark"],
        fontsize=6.5,
    )

    bins = np.linspace(max(0.0, float(np.nanmin(r2)) - 0.02), min(1.0, float(np.nanmax(r2)) + 0.02), 18)
    ax_hist.hist(r2[np.isfinite(r2)], bins=bins, orientation="horizontal", color=PALETTE["hist"], alpha=0.82)
    ax_hist.axhline(mean_r2, color=PALETTE["neutral_dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax_hist.set_ylim(ax_ts.get_ylim())
    ax_hist.set_xlabel("Snapshots")
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
    summary = finite_summary(r2)
    smooth = rolling_median(r2, args.rolling_window)
    low_start, low_end, low_mean = lowest_window(sample_index, r2, args.rolling_window)

    write_summary(
        args.summary_output,
        summary,
        int(np.min(sample_index)),
        int(np.max(sample_index)),
        low_start,
        low_end,
        low_mean,
    )
    write_stats_csv(args.stats_csv, summary, low_start, low_end, low_mean)
    draw_figure(sample_index, r2, smooth, summary, low_start, low_end, args.output_prefix)

    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.stats_csv}", flush=True)
    print(f"mean R2: {summary['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
