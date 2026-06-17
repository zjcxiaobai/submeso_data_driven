#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PROJECT_DIR = Path(__file__).resolve().parents[2]

PALETTE = {
    "truth": "#272727",
    "no_shuffle": "#0F4D92",
    "shuffle_train_val": "#B64342",
    "shuffle_all": "#42949E",
    "neutral_dark": "#4D4D4D",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match shuffle-all samples inside the chronological held-out window, "
            "average fields over the matched times, and plot heat-flux PDFs."
        )
    )
    parser.add_argument(
        "--no-shuffle-predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle/test_predictions_denorm.npz",
        help="Denormalized predictions produced by evaluate.py for no-shuffle.",
    )
    parser.add_argument(
        "--shuffle-train-val-predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_shuffle_train_val/test_predictions_denorm.npz",
        help="Denormalized predictions produced by evaluate.py for shuffle-train-val.",
    )
    parser.add_argument(
        "--shuffle-all-predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_shuffle_all/test_predictions_denorm.npz",
        help="Denormalized predictions produced by evaluate.py for shuffle-all.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/shuffle_data/heatflux_pdf_matched_heldout_time_mean",
        help="Output prefix without extension.",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of PDF bins.")
    return parser.parse_args()


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def load_prediction_file(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run evaluate.py first.")
    data = np.load(path)
    required = {"sample_index", "prediction", "target"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
    sample_index = np.asarray(data["sample_index"], dtype=np.int64)
    order = np.argsort(sample_index)
    return {
        "sample_index": sample_index[order],
        "prediction": np.asarray(data["prediction"])[order],
        "target": np.asarray(data["target"])[order],
    }


def select_by_indices(data: dict[str, np.ndarray], selected_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(data["sample_index"], selected_indices)
    if np.any(positions >= data["sample_index"].size):
        raise ValueError("Some selected sample indices are outside the available prediction file.")
    if not np.array_equal(data["sample_index"][positions], selected_indices):
        missing = selected_indices[data["sample_index"][positions] != selected_indices]
        raise ValueError(f"Missing sample indices in prediction file: {missing[:10]}")
    return data["prediction"][positions], data["target"][positions]


def time_mean(values: np.ndarray) -> np.ndarray:
    return np.mean(values, axis=0)


def pdf_line(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values[np.isfinite(values)], bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def label_with_mean_std(name: str, values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    return f"{name}, mean={np.mean(finite):.3g}, std={np.std(finite):.3g}"


def plot_pdf(fields: dict[str, np.ndarray], bins_count: int, output_prefix: Path, n_times: int) -> None:
    apply_publication_style()
    flattened = {name: field.ravel() for name, field in fields.items()}
    all_values = np.concatenate([values[np.isfinite(values)] for values in flattened.values()])
    bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), bins_count + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    for name, color in (
        ("Truth", PALETTE["truth"]),
        ("No shuffle", PALETTE["no_shuffle"]),
        ("Shuffle train/val", PALETTE["shuffle_train_val"]),
        ("Shuffle all", PALETTE["shuffle_all"]),
    ):
        centers, density = pdf_line(flattened[name], bins)
        ax.plot(
            centers,
            density,
            color=color,
            linewidth=1.35,
            label=label_with_mean_std(name, flattened[name]),
        )

    ax.set_xlabel("Submesoscale heat flux")
    ax.set_ylabel("Probability density")
    ax.set_title(f"PDF of time-mean heat flux, matched n={n_times}")
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
    ax.tick_params(colors=PALETTE["neutral_dark"])
    ax.legend(loc="best", fontsize=6, handlelength=2.2)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    no_shuffle = load_prediction_file(args.no_shuffle_predictions)
    shuffle_train_val = load_prediction_file(args.shuffle_train_val_predictions)
    shuffle_all = load_prediction_file(args.shuffle_all_predictions)

    heldout_start = int(np.min(no_shuffle["sample_index"]))
    heldout_end = int(np.max(no_shuffle["sample_index"]))
    shuffle_all_heldout = shuffle_all["sample_index"][
        (shuffle_all["sample_index"] >= heldout_start) & (shuffle_all["sample_index"] <= heldout_end)
    ]
    selected_indices = np.intersect1d(shuffle_all_heldout, no_shuffle["sample_index"])
    selected_indices = np.intersect1d(selected_indices, shuffle_train_val["sample_index"])
    if selected_indices.size == 0:
        raise ValueError("No shuffle-all test samples fall inside the chronological held-out window.")

    no_shuffle_prediction, truth = select_by_indices(no_shuffle, selected_indices)
    shuffle_train_val_prediction, shuffle_train_val_truth = select_by_indices(shuffle_train_val, selected_indices)
    shuffle_all_prediction, shuffle_all_truth = select_by_indices(shuffle_all, selected_indices)

    if not np.allclose(truth, shuffle_train_val_truth, rtol=1.0e-5, atol=1.0e-4):
        raise ValueError("Truth fields differ between no-shuffle and shuffle-train-val files.")
    if not np.allclose(truth, shuffle_all_truth, rtol=1.0e-5, atol=1.0e-4):
        raise ValueError("Truth fields differ between no-shuffle and shuffle-all files.")

    fields = {
        "Truth": time_mean(truth),
        "No shuffle": time_mean(no_shuffle_prediction),
        "Shuffle train/val": time_mean(shuffle_train_val_prediction),
        "Shuffle all": time_mean(shuffle_all_prediction),
    }

    np.savez_compressed(
        args.output_prefix.with_name(args.output_prefix.name + "_fields.npz"),
        sample_index=selected_indices,
        truth=fields["Truth"],
        no_shuffle=fields["No shuffle"],
        shuffle_train_val=fields["Shuffle train/val"],
        shuffle_all=fields["Shuffle all"],
        heldout_start=np.asarray(heldout_start),
        heldout_end=np.asarray(heldout_end),
    )

    plot_pdf(fields, args.bins, args.output_prefix, int(selected_indices.size))
    print(f"Matched {selected_indices.size} shuffle-all samples inside [{heldout_start}, {heldout_end}].", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.output_prefix.with_name(args.output_prefix.name + '_fields.npz')}", flush=True)


if __name__ == "__main__":
    main()
