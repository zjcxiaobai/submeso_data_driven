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
FK_MLE_DIR = PROJECT_DIR / "analysis/fk_mle"

PALETTE = {
    "line": "#B64A3B",
    "mean": "#1F5A85",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate per-snapshot spatial correlation for tuned FK test predictions.")
    parser.add_argument(
        "--fk-predictions",
        type=Path,
        default=FK_MLE_DIR / "fk_test_Ce2p13062237_prediction_physical.npz",
        help="FK NPZ containing prediction and sample_index arrays.",
    )
    parser.add_argument(
        "--truth-predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_predictions_denorm.npz",
        help="SubmesoFluxNet test NPZ containing the physical-unit target array.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spatial_corr/fk_Ce2p13062237_spatial_corr.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spatial_corr/fk_Ce2p13062237_spatial_corr_summary.txt",
    )
    parser.add_argument(
        "--plot-output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/spatial_corr/fk_Ce2p13062237_spatial_corr",
    )
    return parser.parse_args()


def spatial_corr(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
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


def load_aligned_fk_and_truth(fk_path: Path, truth_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not fk_path.exists():
        raise FileNotFoundError(f"Missing FK prediction file: {fk_path}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth file: {truth_path}")

    with np.load(fk_path) as fk_data:
        required = {"sample_index", "prediction"}
        missing = required.difference(fk_data.files)
        if missing:
            raise ValueError(f"{fk_path} is missing arrays: {sorted(missing)}")
        fk_sample = np.asarray(fk_data["sample_index"], dtype=np.int64)
        fk_prediction = np.asarray(fk_data["prediction"], dtype=np.float64)

    with np.load(truth_path) as truth_data:
        required = {"sample_index", "target"}
        missing = required.difference(truth_data.files)
        if missing:
            raise ValueError(f"{truth_path} is missing arrays: {sorted(missing)}")
        truth_sample = np.asarray(truth_data["sample_index"], dtype=np.int64)
        target = np.asarray(truth_data["target"], dtype=np.float64)

    truth_lookup = {int(sample): pos for pos, sample in enumerate(truth_sample)}
    keep_fk: list[int] = []
    keep_truth: list[int] = []
    for fk_pos, sample in enumerate(fk_sample):
        truth_pos = truth_lookup.get(int(sample))
        if truth_pos is not None:
            keep_fk.append(fk_pos)
            keep_truth.append(truth_pos)
    if not keep_fk:
        raise ValueError("No overlapping sample_index values between FK predictions and truth file.")

    aligned_sample = fk_sample[np.asarray(keep_fk, dtype=np.int64)]
    aligned_prediction = fk_prediction[np.asarray(keep_fk, dtype=np.int64)]
    aligned_target = target[np.asarray(keep_truth, dtype=np.int64)]
    return aligned_sample, aligned_prediction, aligned_target


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
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


def write_summary(path: Path, summary: dict[str, float | int], fk_path: Path, truth_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("FK spatial correlation summary\n")
        handle.write(f"fk_predictions: {fk_path}\n")
        handle.write(f"truth_predictions: {truth_path}\n")
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
    ax.set_title("FK held-out test data")
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
    sample_index, prediction, target = load_aligned_fk_and_truth(args.fk_predictions, args.truth_predictions)
    rho = spatial_corr(prediction, target)
    summary = finite_summary(rho)

    write_csv(args.csv_output, sample_index, rho)
    write_summary(args.summary_output, summary, args.fk_predictions, args.truth_predictions)
    plot_series(args.plot_output_prefix, sample_index, rho, float(summary["mean"]))

    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.png')}", flush=True)
    print(f"mean spatial correlation: {summary['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
