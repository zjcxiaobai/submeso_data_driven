#!/usr/bin/env python3
"""Plot first-eight last-decoder PCA maps with physics diagnostics for model_analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_PCA = HERE / "last_decoder_pca/last_decoder_pca_results.npz"
DEFAULT_PHYSICS = HERE / "physics_variables_top90/physics_variables.npz"
DEFAULT_OUTPUT = HERE / "final_assessment/pca_first8_with_physics_diagnostics.png"
DEFAULT_RANK_CSV = HERE / "final_assessment/pca_physics_diagnostic_order.csv"

FIELD_SPECS = [
    ("T", r"$T$", "magma", False),
    ("rho", "Density", "magma", False),
    ("dTdx", r"$\partial T/\partial x$", "RdBu_r", True),
    ("dTdy", r"$\partial T/\partial y$", "RdBu_r", True),
    ("frontogenesis_FT", r"Frontogenesis $F_T$", "RdBu_r", True),
    ("gradT", r"$|\nabla T|$", "magma", False),
    ("Rossby_number", r"$Ro$", "RdBu_r", True),
    ("shear_strain", "Shear strain", "RdBu_r", True),
    ("strain", "Strain", "magma", False),
    ("strain_times_gradT", r"Strain $\times |\nabla T|$", "magma", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rank-csv", type=Path, default=DEFAULT_RANK_CSV)
    parser.add_argument("--rank-by", choices=("max_abs", "pc1", "positive_pc1"), default="max_abs")
    parser.add_argument("--n-physics", type=int, default=8)
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def limits(image: np.ndarray, signed: bool) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if signed:
        vmax = float(np.percentile(np.abs(finite), 99))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        return -vmax, vmax
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    return vmin, vmax


def order_fields(field_specs, pca_variables, pca_correlations, rank_by: str):
    rows = []
    for name, label, cmap, signed in field_specs:
        if name not in pca_variables:
            continue
        corr = np.asarray(pca_correlations[pca_variables.index(name)], dtype=float)
        if rank_by == "max_abs":
            score = float(np.nanmax(np.abs(corr)))
            display_corr = float(corr[int(np.nanargmax(np.abs(corr)))])
            display_pc = int(np.nanargmax(np.abs(corr))) + 1
        elif rank_by == "positive_pc1":
            score = float(corr[0])
            display_corr = float(corr[0])
            display_pc = 1
        else:
            score = float(abs(corr[0]))
            display_corr = float(corr[0])
            display_pc = 1
        rows.append({
            "name": name,
            "label": label,
            "cmap": cmap,
            "signed": signed,
            "score": score,
            "display_corr": display_corr,
            "display_pc": display_pc,
            "all_corr": corr,
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def write_rank_csv(path: Path, ordered) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "physics_variable", "display_label", "score", "display_pc", "display_r", *[f"pc{i}_r" for i in range(1, 9)]])
        writer.writeheader()
        for rank, row in enumerate(ordered, start=1):
            record = {
                "rank": rank,
                "physics_variable": row["name"],
                "display_label": row["label"],
                "score": float(row["score"]),
                "display_pc": int(row["display_pc"]),
                "display_r": float(row["display_corr"]),
            }
            corr = row["all_corr"]
            for pc in range(8):
                record[f"pc{pc + 1}_r"] = float(corr[pc]) if pc < corr.size else float("nan")
            writer.writerow(record)


def main() -> None:
    args = parse_args()
    with np.load(args.pca, allow_pickle=True) as data:
        if "mean_pc_maps" in data.files:
            pc_maps = np.asarray(data["mean_pc_maps"][:8], dtype=float)
        else:
            pc_maps = np.nanmean(np.asarray(data["pc_maps"], dtype=float), axis=0)[:8]
        explained = np.asarray(data["explained_variance_ratio"][:8], dtype=float)
        pca_variables = [str(value) for value in data["physics_variables"]]
        pca_correlations = np.asarray(data["physics_correlations"], dtype=float)
    with np.load(args.physics, allow_pickle=True) as data:
        fields = {}
        for name, *_ in FIELD_SPECS:
            if name not in data.files:
                continue
            field = np.asarray(data[name], dtype=float)
            fields[name] = np.nanmean(field, axis=0) if field.ndim == 3 else field

    ordered = [row for row in order_fields(FIELD_SPECS, pca_variables, pca_correlations, args.rank_by) if row["name"] in fields]
    ordered = ordered[: args.n_physics]
    if len(ordered) < 1:
        raise ValueError("No requested physics diagnostics were found in both PCA results and physics file.")
    write_rank_csv(args.rank_csv, ordered)

    style()
    fig = plt.figure(figsize=(8.5, 5.0), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        8,
        width_ratios=[1.0, 0.04, 1.0, 0.04, 1.0, 0.04, 1.0, 0.04],
        height_ratios=[1, 1, 1, 1],
        wspace=0.2,
        hspace=0.2,
        left=0.095,
        right=0.990,
        top=0.965,
        bottom=0.045,
    )

    pc_vmax = float(np.percentile(np.abs(pc_maps[np.isfinite(pc_maps)]), 99))
    if not np.isfinite(pc_vmax) or pc_vmax == 0.0:
        pc_vmax = 1.0
    for index in range(8):
        row = index // 4
        col = (index % 4) * 2
        ax = fig.add_subplot(grid[row, col])
        cax = fig.add_subplot(grid[row, col + 1])
        mesh = ax.imshow(pc_maps[index], origin="lower", aspect="equal", cmap="RdBu_r", vmin=-pc_vmax, vmax=pc_vmax, interpolation="nearest")
        ax.set_title(f"PC{index + 1} ({100.0 * explained[index]:.1f}%)", fontsize=7.4, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.tick_params(labelsize=5.3, length=1.5, width=0.5, pad=1.0)
        if index == 3:
            cbar.set_label("PC score", fontsize=6.2, labelpad=2)

    for index, row_data in enumerate(ordered[:8]):
        row = 2 + index // 4
        col = (index % 4) * 2
        ax = fig.add_subplot(grid[row, col])
        cax = fig.add_subplot(grid[row, col + 1])
        image = fields[row_data["name"]]
        vmin, vmax = limits(image, bool(row_data["signed"]))
        mesh = ax.imshow(image, origin="lower", aspect="equal", cmap=row_data["cmap"], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{row_data['label']}\nPC{row_data['display_pc']} $r={row_data['display_corr']:.2f}$", fontsize=7.4, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.tick_params(labelsize=5.3, length=1.5, width=0.5, pad=1.0)

    fig.text(0.060, 0.755, "PCA modes", rotation=90, va="center", ha="center", fontsize=8)
    fig.text(0.060, 0.255, "Physics diagnostics", rotation=90, va="center", ha="center", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Physics order:", ", ".join(row["name"] for row in ordered))
    print(f"Saved {args.output}")
    print(f"Saved {args.rank_csv}")


if __name__ == "__main__":
    main()
