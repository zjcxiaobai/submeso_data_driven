#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz"
DEFAULT_OUTPUT_DIR = HERE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the 4 learned bottleneck feature maps from model_analysis outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-mode",
        choices=["best", "best_middle_worst", "mean", "all"],
        default="best",
        help="Which bottleneck maps to plot from the test set.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Plot one physical sample_index from the prediction file.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--cmap", default="RdBu_r")
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def robust_symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(finite, percentile))
    return limit if limit > 0 else 1.0


def plot_feature_set(features: np.ndarray, output: Path, *, title: str, cmap: str, dpi: int) -> None:
    if features.ndim != 3:
        raise ValueError(f"Expected one feature set with shape (4,H,W), got {features.shape}")
    n_features = features.shape[0]
    style()
    fig, axes = plt.subplots(1, n_features, figsize=(7.2, 1.95), constrained_layout=True)
    if n_features == 1:
        axes = np.asarray([axes])
    limit = robust_symmetric_limit(features)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    mesh = None
    for feature_id, ax in enumerate(axes):
        mesh = ax.imshow(
            features[feature_id],
            origin="lower",
            aspect="equal",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        ax.set_title(f"Feature {feature_id + 1}", fontsize=8, pad=3)
        ax.set_xticks([0, features.shape[2] // 2, features.shape[2] - 1])
        ax.set_yticks([0, features.shape[1] // 2, features.shape[1] - 1])
        ax.tick_params(labelsize=6.5, length=2)
        if feature_id == 0:
            ax.set_ylabel("y index")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("x index")
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.030, pad=0.018)
        cbar.set_label("Latent activation", fontsize=7)
        cbar.ax.tick_params(labelsize=6.5, length=2)
    fig.suptitle(title, fontsize=9, y=1.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def select_best_middle_worst(r2: np.ndarray) -> list[tuple[str, int]]:
    valid = np.where(np.isfinite(r2))[0]
    if valid.size == 0:
        raise ValueError("No finite sample R2 values are available for selection.")
    ordered = valid[np.argsort(r2[valid])]
    return [
        ("worst", int(ordered[0])),
        ("middle", int(ordered[len(ordered) // 2])),
        ("best", int(ordered[-1])),
    ]


def write_selection_csv(path: Path, records: list[dict[str, int | float | str]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    with np.load(args.input, allow_pickle=True) as data:
        if "latent_features" not in data.files:
            raise KeyError(f"{args.input} does not contain latent_features")
        latent = np.asarray(data["latent_features"], dtype=np.float64)
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64) if "r2" in data.files else np.full(latent.shape[0], np.nan)
        mse = np.asarray(data["mse"], dtype=np.float64) if "mse" in data.files else np.full(latent.shape[0], np.nan)
        mae = np.asarray(data["mae"], dtype=np.float64) if "mae" in data.files else np.full(latent.shape[0], np.nan)

    if latent.ndim != 4:
        raise ValueError(f"Expected latent_features with shape (N,4,H,W), got {latent.shape}")
    if latent.shape[1] != 4:
        raise ValueError(f"Expected 4 bottleneck feature maps, got {latent.shape[1]}")

    records: list[dict[str, int | float | str]] = []
    selections: list[tuple[str, int]] = []
    if args.sample_index is not None:
        matches = np.where(sample_index == int(args.sample_index))[0]
        if matches.size == 0:
            raise ValueError(f"sample_index {args.sample_index} was not found in {args.input}")
        selections = [(f"sample_{args.sample_index}", int(matches[0]))]
    elif args.sample_mode == "best":
        selections = [select_best_middle_worst(r2)[-1]]
    elif args.sample_mode == "best_middle_worst":
        selections = select_best_middle_worst(r2)
    elif args.sample_mode == "all":
        selections = [(f"sample_{int(idx)}", pos) for pos, idx in enumerate(sample_index)]

    for label, pos in selections:
        idx = int(sample_index[pos])
        title = f"Bottleneck feature maps: {label}, sample {idx}"
        if np.isfinite(r2[pos]):
            title += f", R2={r2[pos]:.3f}"
        output = output_dir / f"bottleneck_4_features_{label}.png"
        plot_feature_set(latent[pos], output, title=title, cmap=args.cmap, dpi=args.dpi)
        records.append({
            "label": label,
            "test_position": int(pos),
            "sample_index": idx,
            "r2": float(r2[pos]),
            "mse": float(mse[pos]),
            "mae": float(mae[pos]),
        })

    if args.sample_mode in {"mean", "best_middle_worst"} and args.sample_index is None:
        mean_features = np.nanmean(latent, axis=0)
        plot_feature_set(
            mean_features,
            output_dir / "bottleneck_4_features_test_mean.png",
            title="Mean bottleneck feature maps across test samples",
            cmap=args.cmap,
            dpi=args.dpi,
        )

    write_selection_csv(output_dir / "bottleneck_4_features_selected_samples.csv", records)
    print(f"Saved bottleneck feature figures under {output_dir}")


if __name__ == "__main__":
    main()
