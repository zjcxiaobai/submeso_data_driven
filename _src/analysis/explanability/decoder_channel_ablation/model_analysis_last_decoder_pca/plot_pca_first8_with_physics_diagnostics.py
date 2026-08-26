#!/usr/bin/env python3
"""Plot first-eight last-decoder PCA maps with input fields and top physics diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
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

INPUT_FIELD_SPECS = [
    ("U", r"$U$", "RdBu_r", True),
    ("V", r"$V$", "RdBu_r", True),
    ("W", r"$W$", "RdBu_r", True),
    ("zeta", r"$\zeta$", "RdBu_r", True),
    ("strain", r"$S$", "magma", False),
    ("T", r"$T$", "magma", False),
    ("rho", r"$\rho$", "magma", False),
    ("KPPhbl", r"$h_{\mathrm{bl}}$", "magma", False),
]

DIAGNOSTIC_FIELD_SPECS = [
    ("dTdx", r"$\partial T/\partial x$", "RdBu_r", True),
    ("dTdy", r"$\partial T/\partial y$", "RdBu_r", True),
    ("gradT", r"$|\nabla_h T|$", "magma", False),
    ("drhodx", r"$\partial \rho/\partial x$", "RdBu_r", True),
    ("drhody", r"$\partial \rho/\partial y$", "RdBu_r", True),
    ("grad_rho", r"$|\nabla_h \rho|$", "magma", False),
    ("frontogenesis_FT", r"$F_T$", "RdBu_r", True),
    ("gradT_dot_S_gradT", r"$\nabla_h T\cdot\mathbf{S}\cdot\nabla_h T$", "RdBu_r", True),
    ("Rossby_number", r"$Ro$", "RdBu_r", True),
    ("abs_Ro_times_gradT", r"$|Ro||\nabla_h T|$", "magma", False),
    ("shear_strain", "Shear strain", "RdBu_r", True),
    ("normal_strain", "Normal strain", "RdBu_r", True),
    ("strain_derived", r"$S_{\mathrm{derived}}$", "magma", False),
    ("strain_times_gradT", r"$S|\nabla_h T|$", "magma", False),
    ("strain_times_grad_rho", r"$S|\nabla_h \rho|$", "magma", False),
    ("okubo_weiss_input_times_gradT", r"$(S^2-\zeta^2)|\nabla_h T|$", "RdBu_r", True),
    ("hbl_strain_times_gradT", r"$h_{\mathrm{bl}}S|\nabla_h T|$", "magma", False),
    ("vorticity_times_gradT", r"$\zeta|\nabla_h T|$", "RdBu_r", True),
    ("horizontal_T_advection", r"$-\mathbf{u}\cdot\nabla_h T$", "RdBu_r", True),
    ("grad_speed", r"$|\nabla_h |\mathbf{u}||$", "magma", False),
    ("Okubo_Weiss", r"$S_d^2-\zeta_d^2$", "RdBu_r", True),
    ("Q_convergence", r"$-\nabla_h\cdot\mathbf{Q}$", "RdBu_r", True),
    ("laplacian_T", r"$\nabla_h^2T$", "RdBu_r", True),
    ("laplacian_vorticity", r"$\nabla_h^2\zeta$", "RdBu_r", True),
]

FIELD_SPECS = INPUT_FIELD_SPECS + DIAGNOSTIC_FIELD_SPECS
INPUT_FIELD_NAMES = {name for name, *_ in INPUT_FIELD_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rank-csv", type=Path, default=DEFAULT_RANK_CSV)
    parser.add_argument("--rank-by", choices=("max_abs", "pc1", "positive_pc1"), default="pc1")
    parser.add_argument("--n-physics", type=int, default=10, help="Number of non-input diagnostics to keep after the input features.")
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
        writer = csv.DictWriter(handle, fieldnames=["rank", "group", "physics_variable", "display_label", "score", "display_pc", "display_r", *[f"pc{i}_r" for i in range(1, 9)]])
        writer.writeheader()
        for rank, row in enumerate(ordered, start=1):
            corr = row["all_corr"]
            record = {
                "rank": rank,
                "group": "input" if row["name"] in INPUT_FIELD_NAMES else "top_diagnostic",
                "physics_variable": row["name"],
                "display_label": row["label"],
                "score": float(row["score"]),
                "display_pc": int(row["display_pc"]),
                "display_r": float(row["display_corr"]),
            }
            for pc in range(8):
                record[f"pc{pc + 1}_r"] = float(corr[pc]) if pc < corr.size else float("nan")
            writer.writerow(record)


def load_field_mean(data, name: str) -> np.ndarray | None:
    if name not in data.files:
        return None
    field = np.asarray(data[name], dtype=float)
    return np.nanmean(field, axis=0) if field.ndim == 3 else field


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
            field = load_field_mean(data, name)
            if field is not None:
                fields[name] = field

    ranked = [row for row in order_fields(FIELD_SPECS, pca_variables, pca_correlations, args.rank_by) if row["name"] in fields]
    row_by_name = {row["name"]: row for row in ranked}
    input_rows = [row_by_name[name] for name, *_ in INPUT_FIELD_SPECS if name in row_by_name]
    diagnostic_rows = [row for row in ranked if row["name"] not in INPUT_FIELD_NAMES][: args.n_physics]
    ordered = input_rows + diagnostic_rows
    if not ordered:
        raise ValueError("No requested fields were found in both PCA results and physics file.")
    write_rank_csv(args.rank_csv, ordered)

    style()
    n_cols = 4
    pc_rows = int(math.ceil(8 / n_cols))
    field_rows = int(math.ceil(len(ordered) / n_cols))
    total_rows = pc_rows + field_rows
    fig = plt.figure(figsize=(8.5, max(5.0, 1.18 * total_rows + 0.4)), constrained_layout=False)
    grid = fig.add_gridspec(
        total_rows,
        8,
        width_ratios=[1.0, 0.04, 1.0, 0.04, 1.0, 0.04, 1.0, 0.04],
        height_ratios=[1] * total_rows,
        wspace=0.2,
        hspace=0.32,
        left=0.095,
        right=0.990,
        top=0.970,
        bottom=0.035,
    )

    pc_vmax = float(np.percentile(np.abs(pc_maps[np.isfinite(pc_maps)]), 99))
    if not np.isfinite(pc_vmax) or pc_vmax == 0.0:
        pc_vmax = 1.0
    for index in range(8):
        row = index // n_cols
        col = (index % n_cols) * 2
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

    for index, row_data in enumerate(ordered):
        row = pc_rows + index // n_cols
        col = (index % n_cols) * 2
        ax = fig.add_subplot(grid[row, col])
        cax = fig.add_subplot(grid[row, col + 1])
        image = fields[row_data["name"]]
        vmin, vmax = limits(image, bool(row_data["signed"]))
        mesh = ax.imshow(image, origin="lower", aspect="equal", cmap=row_data["cmap"], vmin=vmin, vmax=vmax, interpolation="nearest")
        if row_data["name"] in INPUT_FIELD_NAMES:
            title = row_data["label"]
        else:
            title = f"{row_data['label']}\nPC{row_data['display_pc']} $r={row_data['display_corr']:.2f}$"
        ax.set_title(title, fontsize=7.2, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.tick_params(labelsize=5.3, length=1.5, width=0.5, pad=1.0)

    fig.text(0.060, 1.0 - (pc_rows / total_rows) / 2, "PCA modes", rotation=90, va="center", ha="center", fontsize=8)
    fig.text(0.060, (field_rows / total_rows) / 2, "Inputs + top diagnostics", rotation=90, va="center", ha="center", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Input fields:", ", ".join(row["name"] for row in input_rows))
    print("Top diagnostics:", ", ".join(row["name"] for row in diagnostic_rows))
    print(f"Saved {args.output}")
    print(f"Saved {args.rank_csv}")


if __name__ == "__main__":
    main()
