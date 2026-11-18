#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate per-snapshot spatial correlation for SubmesoFluxNet test predictions.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz",
        help="NPZ containing sample_index, prediction, and target arrays.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=HERE / "submesofluxnet_spatial_corr.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=HERE / "submesofluxnet_spatial_corr_summary.txt",
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


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    with np.load(path) as data:
        required = {"sample_index", "prediction", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
    if sample_index.shape[0] != prediction.shape[0]:
        raise ValueError("sample_index length does not match prediction sample count")
    return sample_index, prediction, target


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite spatial-correlation values were calculated.")
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "spatial_corr"])
        writer.writeheader()
        for test_position, (sample, corr) in enumerate(zip(sample_index, rho)):
            writer.writerow(
                {
                    "test_position": test_position,
                    "sample_index": int(sample),
                    "spatial_corr": float(corr),
                }
            )


def write_summary(path: Path, summary: dict[str, float | int], predictions_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("SubmesoFluxNet spatial correlation summary\n")
        handle.write(f"predictions: {predictions_path}\n")
        for key in ("count", "finite_count", "mean", "median", "std", "min", "max", "p05", "p95"):
            value = summary[key]
            if isinstance(value, int):
                handle.write(f"{key}: {value}\n")
            else:
                handle.write(f"{key}: {value:.10g}\n")


def main() -> None:
    args = parse_args()
    sample_index, prediction, target = load_predictions(args.predictions)
    rho = spatial_corr(prediction, target)
    summary = finite_summary(rho)

    write_csv(args.csv_output, sample_index, rho)
    write_summary(args.summary_output, summary, args.predictions)

    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"mean spatial correlation: {summary['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
