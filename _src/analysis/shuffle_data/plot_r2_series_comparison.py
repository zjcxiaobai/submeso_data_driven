#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PROJECT_DIR = Path(__file__).resolve().parents[2]

PALETTE = {
    "no_shuffle": "#0F4D92",
    "shuffle_train_val": "#B64342",
    "shuffle_all": "#42949E",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-test-sample R2 time series for no-shuffle, shuffle-train-val, and shuffle-all models."
    )
    parser.add_argument(
        "--no-shuffle-r2",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle/test_r2_series.csv",
        help="CSV produced by evaluate.py for the no-shuffle model.",
    )
    parser.add_argument(
        "--shuffle-train-val-r2",
        type=Path,
        default=PROJECT_DIR / "outputs_shuffle_train_val/test_r2_series.csv",
        help="CSV produced by evaluate.py for the shuffle-train-val model.",
    )
    parser.add_argument(
        "--shuffle-all-r2",
        type=Path,
        default=PROJECT_DIR / "outputs_shuffle_all/test_r2_series.csv",
        help="CSV produced by evaluate.py for the shuffle-all model.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/shuffle_data/r2_series_shuffle_comparison",
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--zoom-count",
        type=int,
        default=25,
        help="Number of initial chronological test samples to show in the inset zoom.",
    )
    return parser.parse_args()


def read_r2_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluate.py first so it writes test_r2_series.csv."
        )

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

    if not r2:
        raise ValueError(f"No R2 rows found in {path}")
    sample_index_arr = np.asarray(sample_index, dtype=np.int64)
    r2_arr = np.asarray(r2, dtype=np.float64)
    order = np.argsort(sample_index_arr)
    return sample_index_arr[order], r2_arr[order]


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def finite_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(values[finite]))


def crop_to_window(x: np.ndarray, y: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    keep = (x >= start) & (x <= end)
    if not np.any(keep):
        raise ValueError(f"No samples found inside chronological test window [{start}, {end}]")
    return x[keep], y[keep]


def set_zoom_limits(ax, series: list[tuple[np.ndarray, np.ndarray]], start: int, end: int) -> None:
    values: list[np.ndarray] = []
    for x, y in series:
        keep = (x >= start) & (x <= end) & np.isfinite(y)
        if np.any(keep):
            values.append(y[keep])
    if not values:
        ax.set_xlim(start, end)
        return

    zoom_values = np.concatenate(values)
    y_min = float(np.min(zoom_values))
    y_max = float(np.max(zoom_values))
    pad = max((y_max - y_min) * 0.18, 0.02)
    ax.set_xlim(start, end)
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_comparison(
    no_shuffle_x: np.ndarray,
    no_shuffle_r2: np.ndarray,
    shuffle_train_val_x: np.ndarray,
    shuffle_train_val_r2: np.ndarray,
    shuffle_all_x: np.ndarray,
    shuffle_all_r2: np.ndarray,
    output_prefix: Path,
    zoom_count: int,
) -> None:
    apply_publication_style()
    fig, (ax, zoom_ax) = plt.subplots(
        1,
        2,
        figsize=(5.2, 2.25),
        width_ratios=(2.5, 1.0),
        constrained_layout=True,
    )

    test_start = int(np.min(no_shuffle_x))
    test_end = int(np.max(no_shuffle_x))
    shuffle_train_val_x, shuffle_train_val_r2 = crop_to_window(
        shuffle_train_val_x, shuffle_train_val_r2, test_start, test_end
    )
    shuffle_all_x, shuffle_all_r2 = crop_to_window(shuffle_all_x, shuffle_all_r2, test_start, test_end)
    zoom_end = min(test_end, test_start + max(zoom_count, 2) - 1)

    ax.axvspan(
        test_start,
        zoom_end,
        color=PALETTE["neutral_light"],
        alpha=0.24,
        linewidth=0,
        zorder=0,
    )

    ax.plot(
        no_shuffle_x,
        no_shuffle_r2,
        color=PALETTE["no_shuffle"],
        linewidth=1.15,
        zorder=3,
        label=fr"No shuffle, mean $R^2$={finite_mean(no_shuffle_r2):.3f}",
    )
    ax.plot(
        shuffle_train_val_x,
        shuffle_train_val_r2,
        color=PALETTE["shuffle_train_val"],
        linewidth=1.15,
        zorder=3,
        label=fr"Shuffle train/val, mean $R^2$={finite_mean(shuffle_train_val_r2):.3f}",
    )
    ax.plot(
        shuffle_all_x,
        shuffle_all_r2,
        color=PALETTE["shuffle_all"],
        linewidth=1.15,
        zorder=3,
        label=fr"Shuffle all, mean $R^2$={finite_mean(shuffle_all_r2):.3f}",
    )

    ax.axhline(0.0, color=PALETTE["neutral_light"], linewidth=0.8, zorder=1)
    ax.set_xlim(test_start, test_end)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"$R^2$ of held-out test data")
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
    ax.tick_params(colors=PALETTE["neutral_dark"])
    ax.legend(loc="best", handlelength=2.2)

    for x, y, color in (
        (no_shuffle_x, no_shuffle_r2, PALETTE["no_shuffle"]),
        (shuffle_train_val_x, shuffle_train_val_r2, PALETTE["shuffle_train_val"]),
        (shuffle_all_x, shuffle_all_r2, PALETTE["shuffle_all"]),
    ):
        zoom_ax.plot(x, y, color=color, linewidth=0.95, zorder=3)
    zoom_ax.axhline(0.0, color=PALETTE["neutral_light"], linewidth=0.6, zorder=1)
    set_zoom_limits(
        zoom_ax,
        [
            (no_shuffle_x, no_shuffle_r2),
            (shuffle_train_val_x, shuffle_train_val_r2),
            (shuffle_all_x, shuffle_all_r2),
        ],
        test_start,
        zoom_end,
    )
    zoom_ax.set_xlabel("Test sample index")
    zoom_ax.set_title(r"$R^2$ of first few test samples", fontsize=7, pad=3)
    zoom_ax.tick_params(labelsize=6, length=2, width=0.6, colors=PALETTE["neutral_dark"])
    zoom_ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.5)
    for spine in zoom_ax.spines.values():
        spine.set_color(PALETTE["neutral_mid"])
        spine.set_linewidth(0.6)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    no_shuffle_x, no_shuffle_r2 = read_r2_series(args.no_shuffle_r2)
    shuffle_train_val_x, shuffle_train_val_r2 = read_r2_series(args.shuffle_train_val_r2)
    shuffle_all_x, shuffle_all_r2 = read_r2_series(args.shuffle_all_r2)
    plot_comparison(
        no_shuffle_x,
        no_shuffle_r2,
        shuffle_train_val_x,
        shuffle_train_val_r2,
        shuffle_all_x,
        shuffle_all_r2,
        args.output_prefix,
        args.zoom_count,
    )
    print(f"Saved {args.output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
