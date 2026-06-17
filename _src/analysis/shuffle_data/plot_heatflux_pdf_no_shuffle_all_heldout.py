#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    "neutral_dark": "#4D4D4D",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot heat-flux PDFs for truth and no-shuffle prediction over all held-out test data."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle/test_predictions_denorm.npz",
        help="Denormalized no-shuffle predictions produced by evaluate.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/shuffle_data/heatflux_pdf_no_shuffle_all_heldout",
        help="Output prefix without extension.",
    )
    parser.add_argument("--bins", type=int, default=120, help="Number of PDF bins.")
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


def load_truth_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run evaluate.py first.")
    data = np.load(path)
    required = {"sample_index", "prediction", "target"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing required arrays: {sorted(missing)}")
    sample_index = np.asarray(data["sample_index"], dtype=np.int64)
    order = np.argsort(sample_index)
    return sample_index[order], np.asarray(data["target"])[order], np.asarray(data["prediction"])[order]


def finite_flatten(values: np.ndarray) -> np.ndarray:
    flat = values.ravel()
    return flat[np.isfinite(flat)]


def pdf_line(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def summary_stats(name: str, values: np.ndarray) -> dict[str, float | str]:
    return {
        "field": name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def label_with_mean_std(name: str, values: np.ndarray) -> str:
    return f"{name}, mean={np.mean(values):.3g}, std={np.std(values):.3g}"


def save_stats(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pdf(
    truth_values: np.ndarray,
    prediction_values: np.ndarray,
    bins_count: int,
    output_prefix: Path,
    n_times: int,
) -> None:
    apply_publication_style()
    all_values = np.concatenate([truth_values, prediction_values])
    bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), bins_count + 1)

    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    for name, values, color in (
        ("Truth", truth_values, PALETTE["truth"]),
        ("No shuffle prediction", prediction_values, PALETTE["no_shuffle"]),
    ):
        centers, density = pdf_line(values, bins)
        ax.plot(
            centers,
            density,
            color=color,
            linewidth=1.35,
            label=label_with_mean_std(name, values),
        )

    ax.set_xlabel("Submesoscale heat flux")
    ax.set_ylabel("Probability density")
    ax.set_title(f"PDF over all held-out test fields, n={n_times}")
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
    sample_index, truth, prediction = load_truth_prediction(args.predictions)
    truth_values = finite_flatten(truth)
    prediction_values = finite_flatten(prediction)

    plot_pdf(truth_values, prediction_values, args.bins, args.output_prefix, int(sample_index.size))
    save_stats(
        args.output_prefix.with_name(args.output_prefix.name + "_stats.csv"),
        [
            summary_stats("truth", truth_values),
            summary_stats("no_shuffle_prediction", prediction_values),
        ],
    )

    print(
        f"Used {sample_index.size} held-out samples inside [{int(sample_index.min())}, {int(sample_index.max())}].",
        flush=True,
    )
    print(f"Saved {args.output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.output_prefix.with_name(args.output_prefix.name + '_stats.csv')}", flush=True)


if __name__ == "__main__":
    main()
