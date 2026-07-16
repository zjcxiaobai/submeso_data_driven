#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


DEFAULT_SUMMARY = Path(__file__).resolve().with_name("learning_rate_random_search_summary.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("test_r2_vs_learning_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot test R2 against learning rate for random-search results.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "trial": float(row["trial"]),
                    "learning_rate": float(row["learning_rate"]),
                    "test_r2": float(row["test_r2"]),
                    "val_r2": float(row["val_r2"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )


def main() -> None:
    args = parse_args()
    rows = read_results(args.summary_csv)
    valid = [row for row in rows if math.isfinite(row["learning_rate"]) and math.isfinite(row["test_r2"])]
    if not valid:
        raise ValueError("No finite test_r2 values found.")

    valid = sorted(valid, key=lambda row: row["learning_rate"])
    best = max(valid, key=lambda row: row["test_r2"])

    setup_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.35), constrained_layout=True)

    lr = [row["learning_rate"] for row in valid]
    test_r2 = [row["test_r2"] for row in valid]
    ax.plot(lr, test_r2, color="#486A8E", linewidth=1.1, alpha=0.75, zorder=1)
    ax.scatter(lr, test_r2, s=22, color="#486A8E", edgecolor="white", linewidth=0.45, zorder=2)
    ax.scatter(
        [best["learning_rate"]],
        [best["test_r2"]],
        s=48,
        color="#B45F4D",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label=f"best test R2 = {best['test_r2']:.3f}",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test R2")
    ax.set_title("Learning-rate random search")
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.6)
    ax.grid(axis="x", color="#EEF1F4", linewidth=0.4, which="both")
    ax.legend(loc="lower right")

    ax.annotate(
        f"lr={best['learning_rate']:.2g}",
        xy=(best["learning_rate"], best["test_r2"]),
        xytext=(8, 8),
        textcoords="offset points",
        color="#7A3D34",
        fontsize=7,
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output}.svg", bbox_inches="tight")
    fig.savefig(f"{output}.pdf", bbox_inches="tight")
    print(f"Best finite test R2: {best['test_r2']:.6f} at learning_rate={best['learning_rate']:.8g}")
    print(f"Saved {output}.png")
    print(f"Saved {output}.svg")
    print(f"Saved {output}.pdf")


if __name__ == "__main__":
    main()
