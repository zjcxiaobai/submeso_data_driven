#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DECODER_DIR = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation_encoder_convlstm"
DEFAULT_PCA = DECODER_DIR / "last_decoder_pca/last_decoder_pca_results.npz"
DEFAULT_PHYSICS = DECODER_DIR / "physics_variables_sample_9523/physics_variables.npz"
DEFAULT_OUTPUT = DECODER_DIR / "final_assessment/pc1_physics_snapshot_comparison.png"

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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PC1 decoder mode and physics fields ordered by PC1 correlation.")
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--flip-pc1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot -PC1 and use correlations with -PC1. This only fixes the arbitrary PCA sign.",
    )
    return parser.parse_args()


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def limits(image, signed):
    finite = image[np.isfinite(image)]
    if not finite.size:
        return 0.0, 1.0
    if signed:
        vmax = float(np.percentile(np.abs(finite), 99))
        return -vmax, vmax
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def main():
    args = parse_args()
    with np.load(args.pca, allow_pickle=True) as data:
        pc1 = np.asarray(data["pc_maps"][0, 0], dtype=float)
        explained = float(data["explained_variance_ratio"][0])
        variables = [str(value) for value in data["physics_variables"]]
        correlations = np.asarray(data["physics_correlations"], dtype=float)[:, 0]
    if args.flip_pc1:
        pc1 = -pc1
        correlations = -correlations
        pc_label = r"$-PC1$"
    else:
        pc_label = "PC1"

    with np.load(args.physics, allow_pickle=True) as data:
        fields = {name: np.asarray(data[name], dtype=float) for name, *_ in FIELD_SPECS}

    corr_by_name = {name: float(correlations[variables.index(name)]) for name, *_ in FIELD_SPECS}
    ordered = sorted(FIELD_SPECS, key=lambda item: corr_by_name[item[0]], reverse=True)
    physics_panels = [
        (f"{label}\n$r={corr_by_name[name]:.2f}$", fields[name], cmap, signed)
        for name, label, cmap, signed in ordered
    ]

    style()
    fig = plt.figure(figsize=(9.2, 2.20), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, width_ratios=(1.25, 0.035, 1, 1, 1, 1))

    ax_pc = fig.add_subplot(grid[:, 0])
    vmin, vmax = limits(pc1, True)
    mesh = ax_pc.imshow(
        pc1,
        origin="lower",
        aspect="equal",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_pc.set_title(f"{pc_label} ({100.0 * explained:.1f}%)", fontsize=8, pad=3)
    ax_pc.set_xticks([])
    ax_pc.set_yticks([])
    fig.colorbar(mesh, ax=ax_pc, fraction=0.046, pad=0.02)

    ax_separator = fig.add_subplot(grid[:, 1])
    ax_separator.axvline(0.5, color="black", linewidth=1.0)
    ax_separator.axis("off")

    physics_axes = [
        fig.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(2, 6)
    ]
    for ax, (title, image, cmap, signed) in zip(physics_axes, physics_panels):
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
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("Physics order:", ", ".join(name for name, *_ in ordered))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
