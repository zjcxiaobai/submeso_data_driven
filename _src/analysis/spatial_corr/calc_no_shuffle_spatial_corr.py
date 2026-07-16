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
    "line": "#0F4D92",
    "mean": "#B64342",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate per-snapshot spatial correlation for the no-shuffle test predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm/test_predictions_denorm.npz",
        help="NPZ produced by evaluate.py containing prediction, target, and sample_index.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spatial_corr/no_shuffle_overfitting_test_spatial_corr.csv",
        help="Output CSV for per-snapshot spatial correlations.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spatial_corr/no_shuffle_overfitting_test_spatial_corr_summary.txt",
        help="Output text file with mean and distribution summary.",
    )
    parser.add_argument(
        "--plot-output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/spatial_corr/no_shuffle_overfitting_test_spatial_corr",
        help="Output plot prefix without extension.",
    )
    return parser.parse_args()


def spatial_corr(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    if prediction.ndim < 2:
        raise ValueError("prediction and target must include a sample axis plus spatial axes")

    reduce_axes = tuple(range(1, target.ndim))
    pred_anom = prediction - np.mean(prediction, axis=reduce_axes, keepdims=True)
    target_anom = target - np.mean(target, axis=reduce_axes, keepdims=True)
    numerator = np.sum(pred_anom * target_anom, axis=reduce_axes, dtype=np.float64)
    pred_sum_sq = np.sum(pred_anom**2, axis=reduce_axes, dtype=np.float64)
    target_sum_sq = np.sum(target_anom**2, axis=reduce_axes, dtype=np.float64)
    denominator = np.sqrt(pred_sum_sq * target_sum_sq)

    rho = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=rho, where=denominator > 0.0)
    return rho


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run _src/evaluate.py for the no-shuffle model first.")

    with np.load(path) as data:
        required = {"prediction", "target", "sample_index"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)

    if sample_index.shape[0] != prediction.shape[0]:
        raise ValueError("sample_index length does not match prediction sample count")
    return sample_index, prediction, target


def write_csv(path: Path, sample_index: np.ndarray, rho: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(sample_index)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "spatial_corr"])
        writer.writeheader()
        for test_position, idx in enumerate(order):
            writer.writerow(
                {
                    "test_position": test_position,
                    "sample_index": int(sample_index[idx]),
                    "spatial_corr": float(rho[idx]),
                }
            )


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def write_summary(path: Path, summary: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("No-shuffle spatial correlation summary\n")
        for key in ("count", "finite_count", "mean", "median", "std", "min", "max", "p05", "p95"):
            value = summary[key]
            if isinstance(value, int):
                handle.write(f"{key}: {value}\n")
            else:
                handle.write(f"{key}: {value:.10g}\n")


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


def plot_series(path_prefix: Path, sample_index: np.ndarray, rho: np.ndarray, mean_rho: float) -> None:
    apply_publication_style()
    order = np.argsort(sample_index)
    x = sample_index[order]
    y = rho[order]

    fig, ax = plt.subplots(figsize=(4.4, 2.2), constrained_layout=True)
    ax.plot(x, y, color=PALETTE["line"], linewidth=1.05, label=fr"$\rho_n$")
    ax.axhline(mean_rho, color=PALETTE["mean"], linewidth=1.0, label=fr"Mean $\rho$={mean_rho:.3f}")
    ax.set_xlim(int(np.min(x)), int(np.max(x)))
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Test sample index")
    ax.set_ylabel(r"Spatial correlation $\rho_n$")
    ax.set_title("No-shuffle held-out test data")
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
    ax.tick_params(colors=PALETTE["neutral_dark"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["neutral_mid"])
    ax.legend(loc="best", handlelength=2.2)

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_index, prediction, target = load_predictions(args.predictions)
    rho = spatial_corr(prediction, target)
    summary = finite_summary(rho)

    write_csv(args.csv_output, sample_index, rho)
    write_summary(args.summary_output, summary)
    plot_series(args.plot_output_prefix, sample_index, rho, float(summary["mean"]))

    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.png')}", flush=True)
    print(f"mean spatial correlation: {summary['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
