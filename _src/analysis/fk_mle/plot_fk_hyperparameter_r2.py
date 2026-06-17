#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

PALETTE = {
    "ce": "#1F5A85",
    "lf": "#B64A3B",
    "ce_light": "#DCE8F2",
    "lf_light": "#F1D8D2",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FK R2 sensitivity to C_e and LfMin.")
    parser.add_argument("--ce-csv", type=Path, default=HERE / "ce_random_search_history.csv")
    parser.add_argument("--lfmin-csv", type=Path, default=HERE / "grid_search_LfMin.csv")
    parser.add_argument("--ce-y", default="mean_per_snapshot_r2")
    parser.add_argument("--lfmin-y", default="val_r2_scaled_to_Ce_max")
    parser.add_argument("--output-prefix", type=Path, default=HERE / "fk_hyperparameter_r2")
    parser.add_argument("--summary-output", type=Path, default=HERE / "fk_hyperparameter_r2_summary.txt")
    return parser.parse_args()


def apply_style() -> None:
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


def read_xy(path: Path, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    x_vals: list[float] = []
    y_vals: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {x_col, y_col}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except ValueError:
                continue
            if np.isfinite(x) and np.isfinite(y):
                x_vals.append(x)
                y_vals.append(y)
    if not x_vals:
        raise ValueError(f"No finite {x_col}/{y_col} rows found in {path}")
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    order = np.argsort(x)
    return x[order], y[order]


def best_point(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    idx = int(np.nanargmax(y))
    return float(x[idx]), float(y[idx])


def draw_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    fill: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xscale: str = "linear",
) -> tuple[float, float]:
    bx, by = best_point(x, y)
    if x.size > 25:
        ax.scatter(x, y, s=13, color=color, alpha=0.62, linewidths=0, zorder=2)
    else:
        ax.plot(x, y, color=color, linewidth=1.25, marker="o", markersize=3.0, zorder=2)
        ax.fill_between(x, y, np.nanmin(y), color=fill, alpha=0.38, linewidth=0, zorder=1)
    ax.scatter([bx], [by], s=24, color=color, edgecolor="white", linewidth=0.55, zorder=4)
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.5, pad=4)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.55)
    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])
    return bx, by


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png", ".tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(Path(f"{output_prefix}{ext}"), **kwargs)


def write_summary(
    path: Path,
    ce_csv: Path,
    lfmin_csv: Path,
    ce_y: str,
    lfmin_y: str,
    ce_best: tuple[float, float],
    lf_best: tuple[float, float],
) -> None:
    with path.open("w") as handle:
        handle.write("FK hyperparameter R2 sensitivity\n")
        handle.write(f"ce_csv: {ce_csv}\n")
        handle.write(f"lfmin_csv: {lfmin_csv}\n")
        handle.write(f"ce_metric: {ce_y}\n")
        handle.write(f"lfmin_metric: {lfmin_y}\n")
        handle.write(f"best_C_e: {ce_best[0]:.10g}\n")
        handle.write(f"best_C_e_R2: {ce_best[1]:.10g}\n")
        handle.write(f"best_LfMin: {lf_best[0]:.10g}\n")
        handle.write(f"best_LfMin_R2: {lf_best[1]:.10g}\n")


def main() -> None:
    args = parse_args()
    apply_style()

    ce_x, ce_y = read_xy(args.ce_csv, "C_e", args.ce_y)
    lf_x, lf_y = read_xy(args.lfmin_csv, "LfMin", args.lfmin_y)

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.45), constrained_layout=True)
    ce_best = draw_panel(
        axes[0],
        ce_x,
        ce_y,
        color=PALETTE["ce"],
        fill=PALETTE["ce_light"],
        xlabel=r"$C_e$",
        ylabel=r"Mean per-snapshot $R^2$",
        title=r"$C_e$ sensitivity",
    )
    lf_best = draw_panel(
        axes[1],
        lf_x,
        lf_y,
        color=PALETTE["lf"],
        fill=PALETTE["lf_light"],
        xlabel=r"$L_{f,\min}$ (m)",
        ylabel="",
        title=r"$L_{f,\min}$ sensitivity",
        xscale="log",
    )

    save_figure(fig, args.output_prefix)
    plt.close(fig)
    write_summary(args.summary_output, args.ce_csv, args.lfmin_csv, args.ce_y, args.lfmin_y, ce_best, lf_best)

    for ext in (".svg", ".pdf", ".png", ".tiff"):
        print(f"Saved {args.output_prefix}{ext}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Best C_e: {ce_best[0]:.8g}, R2={ce_best[1]:.6f}", flush=True)
    print(f"Best LfMin: {lf_best[0]:.8g}, R2={lf_best[1]:.6f}", flush=True)


if __name__ == "__main__":
    main()
