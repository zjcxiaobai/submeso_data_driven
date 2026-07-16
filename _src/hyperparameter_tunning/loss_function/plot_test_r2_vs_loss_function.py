#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


DEFAULT_SUMMARY = Path(__file__).resolve().with_name("loss_function_tunning_summary.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("test_r2_vs_loss_function")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot test R2 against tuned loss functions.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path without extension.")
    return parser.parse_args()


def short_label(row: dict[str, str]) -> str:
    name = row["loss_name"]
    if name == "weighted_mse":
        return "Weighted\nMSE"
    if name == "gradient_loss":
        return "Gradient\nloss"
    if name == "spectral_loss":
        return "Spectral\nloss"
    if name == "mse_plus_gradient":
        return "MSE +\ngradient"
    return name.upper() if name in {"mse", "mae"} else name.title()


def finite_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_results(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="") as handle:
        for raw in csv.DictReader(handle):
            test_r2 = finite_float(raw.get("test_r2", ""))
            if test_r2 is None:
                continue
            rows.append(
                {
                    "trial": int(float(raw["trial"])),
                    "loss_name": raw["loss_name"],
                    "label": short_label(raw),
                    "status": raw.get("status", ""),
                    "test_r2": test_r2,
                    "val_r2": finite_float(raw.get("val_r2", "")),
                    "train_r2": finite_float(raw.get("train_r2", "")),
                }
            )
    if not rows:
        raise ValueError(f"No finite test_r2 values found in {path}")
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
    rows = sorted(rows, key=lambda row: int(row["trial"]))
    best = max(rows, key=lambda row: float(row["test_r2"]))

    setup_style()
    fig, ax = plt.subplots(figsize=(4.25, 2.65), constrained_layout=True)

    x = list(range(len(rows)))
    y = [float(row["test_r2"]) for row in rows]
    colors = ["#486A8E" if row is not best else "#B45F4D" for row in rows]
    bars = ax.bar(x, y, color=colors, width=0.72, edgecolor="white", linewidth=0.6)

    for bar, row in zip(bars, rows):
        value = float(row["test_r2"])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#27313B",
        )
        status = str(row["status"])
        if status not in {"", "ok"}:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(0.02, value - 0.06),
                status.replace("_", "\n"),
                ha="center",
                va="top",
                fontsize=5.8,
                color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(row["label"]) for row in rows])
    ax.set_ylabel("Test R2")
    ax.set_xlabel("Loss function")
    ax.set_title("Loss-function tuning")
    ymin = min(-0.05, min(y) - 0.08)
    ymax = min(1.0, max(y) + 0.12)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0.0, color="#333333", linewidth=0.7)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.6)

    ax.annotate(
        f"best: {best['label']} ({float(best['test_r2']):.3f})",
        xy=(rows.index(best), float(best["test_r2"])),
        xytext=(0, 22),
        textcoords="offset points",
        ha="center",
        color="#7A3D34",
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": "#7A3D34", "linewidth": 0.7},
    )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{output}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output}.svg", bbox_inches="tight")
    fig.savefig(f"{output}.pdf", bbox_inches="tight")
    print(f"Best finite test R2: {float(best['test_r2']):.6f} for {best['loss_name']}")
    print(f"Saved {output}.png")
    print(f"Saved {output}.svg")
    print(f"Saved {output}.pdf")


if __name__ == "__main__":
    main()
