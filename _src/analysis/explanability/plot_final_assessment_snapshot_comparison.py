#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
DECODER_DIR = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation_encoder_convlstm"
DEFAULT_FEATURES = DECODER_DIR / "layerwise_feature_map_ablation_sample_9523/decoder_up1_20x40/top_feature_maps_and_ablation.npz"
DEFAULT_PHYSICS = DECODER_DIR / "physics_variables_sample_9523/physics_variables.npz"
DEFAULT_CORRELATIONS = DECODER_DIR / "layerwise_feature_map_ablation_sample_9523/physics_variable_spatial_correlations/all_layer_physics_spatial_correlations.npz"
DEFAULT_OUTPUT = DECODER_DIR / "final_assessment/prediction_rank1_physics_snapshot_comparison.png"

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
    parser = argparse.ArgumentParser(description="Plot rank-1 decoder map and physics fields ordered by correlation.")
    parser.add_argument("--feature-maps", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--correlations", type=Path, default=DEFAULT_CORRELATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    with np.load(args.feature_maps, allow_pickle=True) as data:
        channel = int(data["top_channels"][0])
        feature_map = np.asarray(data["feature_maps"][0], dtype=float)
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
    physics_panels = [
        (f"{label}\n$r={correlations[name]:.2f}$", fields[name], cmap, signed)
        for name, label, cmap, signed in ordered
    ]

    style()
    fig = plt.figure(figsize=(9.2, 2.20), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, width_ratios=(1.25, 0.035, 1, 1, 1, 1))

    ax_rank = fig.add_subplot(grid[:, 0])
    vmin, vmax = limits(feature_map, True)
    mesh = ax_rank.imshow(
        feature_map, origin="lower", aspect="equal", cmap="RdBu_r",
        vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_rank.set_title(f"Rank 1: ch {channel}", fontsize=8, pad=3)
    ax_rank.set_xticks([])
    ax_rank.set_yticks([])
    fig.colorbar(mesh, ax=ax_rank, fraction=0.046, pad=0.02)

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
            image, origin="lower", aspect="equal", cmap=cmap,
            vmin=vmin, vmax=vmax, interpolation="nearest",
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
