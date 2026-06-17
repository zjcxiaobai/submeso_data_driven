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
    "rmse": "#0F4D92",
    "nrmse": "#B64342",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate per-snapshot RMSE and NRMSE for no-shuffle predictions.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle/test_predictions_denorm.npz",
        help="NPZ produced by evaluate.py containing prediction, target, and sample_index.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_DIR / "analysis/RMSE/no_shuffle_rmse.csv",
        help="Output CSV for per-snapshot RMSE and NRMSE.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR / "analysis/RMSE/no_shuffle_rmse_summary.txt",
        help="Output text file with RMSE and NRMSE distribution summaries.",
    )
    parser.add_argument(
        "--plot-output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/RMSE/no_shuffle_rmse",
        help="Output plot prefix without extension.",
    )
    return parser.parse_args()


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

    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    if sample_index.shape[0] != prediction.shape[0]:
        raise ValueError("sample_index length does not match prediction sample count")
    if prediction.ndim < 2:
        raise ValueError("prediction and target must include a sample axis plus spatial axes")
    return sample_index, prediction, target


def rmse_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduce_axes = tuple(range(1, target.ndim))
    mse = np.mean((prediction - target) ** 2, axis=reduce_axes, dtype=np.float64)
    rmse = np.sqrt(mse)
    target_std = np.std(target, axis=reduce_axes, dtype=np.float64)

    nrmse = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    np.divide(rmse, target_std, out=nrmse, where=target_std > 0.0)
    return rmse, target_std, nrmse


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


def write_csv(path: Path, sample_index: np.ndarray, rmse: np.ndarray, target_std: np.ndarray, nrmse: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(sample_index)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["test_position", "sample_index", "rmse", "target_std", "nrmse"],
        )
        writer.writeheader()
        for test_position, idx in enumerate(order):
            writer.writerow(
                {
                    "test_position": test_position,
                    "sample_index": int(sample_index[idx]),
                    "rmse": float(rmse[idx]),
                    "target_std": float(target_std[idx]),
                    "nrmse": float(nrmse[idx]),
                }
            )


def write_summary(path: Path, rmse_summary: dict[str, float | int], nrmse_summary: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ("count", "finite_count", "mean", "median", "std", "min", "max", "p05", "p95")
    with path.open("w") as handle:
        handle.write("No-shuffle RMSE summary\n")
        for name, summary in (("rmse", rmse_summary), ("nrmse", nrmse_summary)):
            handle.write(f"\n{name}\n")
            for key in keys:
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


def plot_series(
    path_prefix: Path,
    sample_index: np.ndarray,
    rmse: np.ndarray,
    nrmse: np.ndarray,
    mean_rmse: float,
    mean_nrmse: float,
) -> None:
    apply_publication_style()
    order = np.argsort(sample_index)
    x = sample_index[order]

    fig, (ax_rmse, ax_nrmse) = plt.subplots(2, 1, figsize=(4.4, 3.0), sharex=True, constrained_layout=True)
    ax_rmse.plot(x, rmse[order], color=PALETTE["rmse"], linewidth=1.0, label="RMSE")
    ax_rmse.axhline(mean_rmse, color=PALETTE["neutral_mid"], linewidth=0.9, label=f"Mean={mean_rmse:.3g}")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("No-shuffle held-out test data")

    ax_nrmse.plot(x, nrmse[order], color=PALETTE["nrmse"], linewidth=1.0, label="NRMSE")
    ax_nrmse.axhline(mean_nrmse, color=PALETTE["neutral_mid"], linewidth=0.9, label=f"Mean={mean_nrmse:.3f}")
    ax_nrmse.set_xlabel("Test sample index")
    ax_nrmse.set_ylabel(r"NRMSE / $\sigma(y_n)$")

    for ax in (ax_rmse, ax_nrmse):
        ax.set_xlim(int(np.min(x)), int(np.max(x)))
        ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
        ax.tick_params(colors=PALETTE["neutral_dark"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["neutral_mid"])
        ax.legend(loc="best", handlelength=2.2)

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_index, prediction, target = load_predictions(args.predictions)
    rmse, target_std, nrmse = rmse_metrics(prediction, target)
    rmse_summary = finite_summary(rmse)
    nrmse_summary = finite_summary(nrmse)

    write_csv(args.csv_output, sample_index, rmse, target_std, nrmse)
    write_summary(args.summary_output, rmse_summary, nrmse_summary)
    plot_series(
        args.plot_output_prefix,
        sample_index,
        rmse,
        nrmse,
        float(rmse_summary["mean"]),
        float(nrmse_summary["mean"]),
    )

    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.png')}", flush=True)
    print(f"mean RMSE: {rmse_summary['mean']:.6g}", flush=True)
    print(f"mean NRMSE: {nrmse_summary['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
