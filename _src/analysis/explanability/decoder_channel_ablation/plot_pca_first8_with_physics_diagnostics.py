#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
DECODER_DIR = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation"
DEFAULT_PCA = DECODER_DIR / "last_decoder_pca/last_decoder_pca_results.npz"
DEFAULT_PHYSICS = DECODER_DIR / "physics_variables_sample_9523/physics_variables.npz"
DEFAULT_CORRELATIONS = DECODER_DIR / "layerwise_feature_map_ablation_sample_9523/physics_variable_spatial_correlations/all_layer_physics_spatial_correlations.npz"
DEFAULT_OUTPUT = DECODER_DIR / "final_assessment/pca_first8_with_physics_diagnostics.png"

FIELD_SPECS = [
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
    parser = argparse.ArgumentParser(description="Combine first eight last-decoder PCA maps with selected physics diagnostics.")
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--correlations", type=Path, default=DEFAULT_CORRELATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def main() -> None:
    args = parse_args()
    with np.load(args.pca, allow_pickle=True) as data:
        pc_maps = np.asarray(data["pc_maps"][0, :8], dtype=float)
        explained = np.asarray(data["explained_variance_ratio"][:8], dtype=float)

    with np.load(args.physics, allow_pickle=True) as data:
        fields = {name: np.asarray(data[name], dtype=float) for name, *_ in FIELD_SPECS}

    with np.load(args.correlations, allow_pickle=True) as data:
        variables = [str(value) for value in data["physics_variables"]]
        layers = [str(value) for value in data["layers"]]
        layer_index = layers.index("decoder_up1_20x40")
        correlations = {
            name: float(data["correlations"][variables.index(name), layer_index, 0])
            for name, *_ in FIELD_SPECS
        }
    ordered = sorted(FIELD_SPECS, key=lambda item: correlations[item[0]], reverse=True)

    style()
    fig = plt.figure(figsize=(8.5, 5), constrained_layout=False)
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

    pc_vmin, pc_vmax = limits(pc_maps, True)
    for index in range(8):
        row = index // 4
        col = (index % 4) * 2
        ax = fig.add_subplot(grid[row, col])
        cax = fig.add_subplot(grid[row, col + 1])
        mesh = ax.imshow(
            pc_maps[index],
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            vmin=pc_vmin,
            vmax=pc_vmax,
            interpolation="nearest",
        )
        ax.set_title(f"PC{index + 1} ({100.0 * explained[index]:.1f}%)", fontsize=7.4, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.tick_params(labelsize=5.3, length=1.5, width=0.5, pad=1.0)
        if index == 3:
            cbar.set_label("PC score", fontsize=6.2, labelpad=2)

    for index, (name, label, cmap, signed) in enumerate(ordered):
        row = 2 + index // 4
        col = (index % 4) * 2
        ax = fig.add_subplot(grid[row, col])
        cax = fig.add_subplot(grid[row, col + 1])
        image = fields[name]
        vmin, vmax = limits(image, signed)
        mesh = ax.imshow(
            image,
            origin="lower",
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{label}\n$r={correlations[name]:.2f}$", fontsize=7.4, pad=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.tick_params(labelsize=5.3, length=1.5, width=0.5, pad=1.0)

    fig.text(0.060, 0.755, "PCA modes", rotation=90, va="center", ha="center", fontsize=8)
    fig.text(0.060, 0.255, "Physics diagnostics", rotation=90, va="center", ha="center", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Physics order:", ", ".join(name for name, *_ in ordered))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
