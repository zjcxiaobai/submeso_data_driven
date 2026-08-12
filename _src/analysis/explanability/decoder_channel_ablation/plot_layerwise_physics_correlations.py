#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DECODER_DIR = HERE
DEFAULT_LAYER_ROOT = DECODER_DIR / "layerwise_feature_map_ablation_sample_9523"
DEFAULT_PHYSICS_DIR = DECODER_DIR / "physics_variables_sample_9523"
DEFAULT_OUTPUT_DIR = DEFAULT_LAYER_ROOT / "physics_variable_spatial_correlations"

LAYER_ORDER = [
    "encoder_lstm_20x40",
    "encoder_lstm_10x20",
    "encoder_lstm_5x10",
    "bottleneck_lstm_2x5",
    "decoder_up3_5x10",
    "decoder_up2_10x20",
    "decoder_up1_20x40",
]

LAYER_LABELS = {
    "encoder_lstm_20x40": "Encoder LSTM 1 (20x40)",
    "encoder_lstm_10x20": "Encoder LSTM 2 (10x20)",
    "encoder_lstm_5x10": "Encoder LSTM 3 (5x10)",
    "bottleneck_lstm_2x5": "Bottleneck LSTM (2x5)",
    "decoder_up3_5x10": "Decoder 3 (5x10)",
    "decoder_up2_10x20": "Decoder 2 (10x20)",
    "decoder_up1_20x40": "Decoder 1 (20x40)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate each layer's top-ranked feature maps with saved physical fields. "
            "Physics fields are block-averaged to each layer's native grid."
        )
    )
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--physics-dir", type=Path, default=DEFAULT_PHYSICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rank", type=int, default=100)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            row["variable"]: row
            for row in csv.DictReader(handle)
        }


def block_average(field: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    input_y, input_x = field.shape
    output_y, output_x = output_shape
    if input_y % output_y or input_x % output_x:
        raise ValueError(
            f"Cannot block-average shape {field.shape} to {output_shape}"
        )
    factor_y = input_y // output_y
    factor_x = input_x // output_x
    return field.reshape(
        output_y, factor_y, output_x, factor_x
    ).mean(axis=(1, 3))


def pearson_r(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    x = left[valid].astype(np.float64)
    y = right[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float("nan") if denominator <= 0.0 else float(np.sum(x * y) / denominator)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def plot_variable(
    correlations: np.ndarray,
    layer_labels: list[str],
    variable: str,
    units: str,
    output: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.35), constrained_layout=True)
    mesh = ax.imshow(
        correlations,
        origin="upper",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        interpolation="nearest",
    )
    rank_count = correlations.shape[1]
    ticks = np.unique(
        np.clip(
            np.asarray([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]),
            0,
            rank_count - 1,
        )
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticks(np.arange(len(layer_labels)))
    ax.set_yticklabels(layer_labels)
    ax.set_xlabel("Channel ablation rank")
    ax.set_title(
        f"Spatial correlation of feature maps with {variable}"
        + (f" ({units})" if units else ""),
        fontsize=8,
        pad=4,
    )
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.018)
    colorbar.set_label("Pearson r")
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_rank < 1:
        raise ValueError("--max-rank must be positive")
    layer_root = args.layer_root.expanduser().resolve()
    physics_dir = args.physics_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(physics_dir / "physics_variable_manifest.csv")
    with np.load(physics_dir / "physics_variables.npz", allow_pickle=True) as source:
        variable_names = [
            name for name in source.files if name != "sample_index"
        ]
        physics = {
            name: np.asarray(source[name], dtype=np.float64)
            for name in variable_names
        }

    layer_data = []
    for layer_name in LAYER_ORDER:
        path = layer_root / layer_name / "top_feature_maps_and_ablation.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing layer result: {path}")
        with np.load(path, allow_pickle=True) as data:
            maps = np.asarray(data["feature_maps"], dtype=np.float64)
            channels = np.asarray(data["top_channels"], dtype=np.int64)
            all_r2_drop = np.asarray(data["r2_drop"], dtype=np.float64)
        rank_count = min(args.max_rank, len(channels))
        channels = channels[:rank_count]
        layer_data.append(
            {
                "name": layer_name,
                "maps": maps[:rank_count],
                "channels": channels,
                "r2_drop": all_r2_drop[channels],
                "shape": tuple(maps.shape[-2:]),
            }
        )

    common_rank_count = min(len(layer["channels"]) for layer in layer_data)
    correlations = np.full(
        (len(variable_names), len(layer_data), common_rank_count),
        np.nan,
        dtype=np.float64,
    )
    output_csv = output_dir / "all_layer_physics_spatial_correlations.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "physics_variable",
                "units",
                "layer",
                "layer_grid_y",
                "layer_grid_x",
                "spatial_cell_count",
                "ablation_rank",
                "channel",
                "r2_drop",
                "spatial_pearson_r",
            ],
        )
        writer.writeheader()
        for variable_index, variable in enumerate(variable_names):
            units = manifest.get(variable, {}).get("units", "")
            field = physics[variable]
            for layer_index, layer in enumerate(layer_data):
                native_field = block_average(field, layer["shape"])
                for rank in range(common_rank_count):
                    correlation = pearson_r(layer["maps"][rank], native_field)
                    correlations[variable_index, layer_index, rank] = correlation
                    writer.writerow(
                        {
                            "physics_variable": variable,
                            "units": units,
                            "layer": layer["name"],
                            "layer_grid_y": layer["shape"][0],
                            "layer_grid_x": layer["shape"][1],
                            "spatial_cell_count": int(np.prod(layer["shape"])),
                            "ablation_rank": rank + 1,
                            "channel": int(layer["channels"][rank]),
                            "r2_drop": float(layer["r2_drop"][rank]),
                            "spatial_pearson_r": correlation,
                        }
                    )
            plot_variable(
                correlations[variable_index],
                [LAYER_LABELS[layer["name"]] for layer in layer_data],
                variable,
                units,
                output_dir / f"spatial_corr_all_layers_vs_{safe_filename(variable)}.png",
            )
            print(
                f"  plotted {variable_index + 1}/{len(variable_names)}: {variable}",
                flush=True,
            )

    np.savez_compressed(
        output_dir / "all_layer_physics_spatial_correlations.npz",
        correlations=correlations,
        physics_variables=np.asarray(variable_names),
        layers=np.asarray([layer["name"] for layer in layer_data]),
        layer_labels=np.asarray([LAYER_LABELS[layer["name"]] for layer in layer_data]),
        channels=np.stack(
            [layer["channels"][:common_rank_count] for layer in layer_data]
        ),
        r2_drop=np.stack(
            [layer["r2_drop"][:common_rank_count] for layer in layer_data]
        ),
        ablation_rank=np.arange(1, common_rank_count + 1, dtype=np.int64),
    )
    metadata = {
        "sample_index": 9523,
        "physics_variable_count": len(variable_names),
        "layer_count": len(layer_data),
        "channels_per_layer": common_rank_count,
        "correlation": "Pearson spatial correlation",
        "resolution_method": (
            "Each 20x40 physics field is area-averaged by exact non-overlapping "
            "blocks to the native feature-map grid before correlation."
        ),
        "inference_warning": (
            "Correlations at coarse layers use few spatial cells, especially the "
            "2x5 bottleneck, and should be interpreted descriptively."
        ),
        "png_only": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved layer-wise physics correlations to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
