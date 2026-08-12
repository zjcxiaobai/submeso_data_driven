#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / "layerwise_feature_map_ablation_sample_9523"
DEFAULT_INPUT = (
    ROOT
    / "physics_variable_spatial_correlations"
    / "all_layer_physics_spatial_correlations.npz"
)
DEFAULT_OUTPUT_DIR = HERE / "final_assessment"

VARIABLES = [
    ("divergence", "Divergence"),
    ("dTdx", r"$\partial T/\partial x$"),
    ("dTdy", r"$\partial T/\partial y$"),
    ("frontogenesis_FT", r"Frontogenesis $F_T$"),
    ("gradT", r"$|\nabla T|$"),
    ("KPPhbl", "KPPHBL"),
    ("Rossby_number", r"$Ro$"),
    ("shear_strain", "Shear strain"),
    ("speed", "Speed"),
    ("strain", "Strain"),
    ("T", r"$T$"),
    ("U", r"$U$"),
    ("V", r"$V$"),
    ("W", r"$W$"),
    ("strain_times_gradT", r"Strain $\times |\nabla T|$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot final-decoder spatial correlations for the top ten "
            "ablation-ranked channels and selected physical diagnostics."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        correlations = np.asarray(data["correlations"], dtype=np.float64)
        physics_variables = [str(value) for value in data["physics_variables"]]
        layers = [str(value) for value in data["layers"]]
        channels = np.asarray(data["channels"], dtype=np.int64)
        r2_drop = np.asarray(data["r2_drop"], dtype=np.float64)

    layer_index = layers.index("decoder_up1_20x40")
    variable_indices = [physics_variables.index(name) for name, _ in VARIABLES]
    values = correlations[variable_indices, layer_index, :10]
    selected_channels = channels[layer_index, :10]
    selected_r2_drop = r2_drop[layer_index, :10]

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    mesh = ax.imshow(
        values,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(
        [f"#{rank}\nch {channel}" for rank, channel in enumerate(selected_channels, start=1)]
    )
    ax.set_yticks(np.arange(len(VARIABLES)))
    ax.set_yticklabels([label for _, label in VARIABLES])
    ax.set_xlabel("Final-decoder channel ranked by prediction degradation")
    ax.set_title(
        "Spatial correlation between final-decoder feature maps and physical diagnostics",
        fontsize=9,
        pad=5,
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if abs(value) >= 0.55 else "#222222"
            ax.text(
                column,
                row,
                f"{value:.2f}" if np.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=5.5,
                color=text_color,
            )
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.018)
    colorbar.set_label("Spatial Pearson r")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / "final_decoder_top10_physics_correlations.png"
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    with (
        args.output_dir / "final_decoder_top10_physics_correlations.csv"
    ).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "physics_variable",
                "display_label",
                "ablation_rank",
                "channel",
                "r2_drop",
                "spatial_pearson_r",
            ],
        )
        writer.writeheader()
        for variable_row, (variable, label) in enumerate(VARIABLES):
            for rank, channel in enumerate(selected_channels, start=1):
                writer.writerow(
                    {
                        "physics_variable": variable,
                        "display_label": label,
                        "ablation_rank": rank,
                        "channel": int(channel),
                        "r2_drop": float(selected_r2_drop[rank - 1]),
                        "spatial_pearson_r": float(values[variable_row, rank - 1]),
                    }
                )
    print(f"Saved {output_png}")


if __name__ == "__main__":
    main()
