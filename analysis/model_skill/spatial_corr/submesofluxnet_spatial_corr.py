#!/usr/bin/env python3
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
OUTPUT = Path(__file__).with_name("submesofluxnet_spatial_corr.csv")


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(PREDICTIONS) as data:
        sample_index = data["sample_index"].astype(np.int64)
        prediction = data["prediction"].astype(np.float64)
        target = data["target"].astype(np.float64)

    if prediction.shape != target.shape:
        raise ValueError(
            f"The prediction shape {prediction.shape} does not match "
            f"the target shape {target.shape}."
        )
    return sample_index, prediction, target


def spatial_correlation(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    prediction = prediction - prediction.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)

    numerator = np.sum(prediction * target, axis=1)
    denominator = np.sqrt(np.sum(prediction**2, axis=1) * np.sum(target**2, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.full(prediction.shape[0], np.nan),
        where=denominator > 0,
    )


def save_results(sample_index: np.ndarray, correlation: np.ndarray) -> None:
    with OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["test_position", "sample_index", "spatial_corr"])
        for position, (sample, corr) in enumerate(zip(sample_index, correlation)):
            writer.writerow([position, int(sample), float(corr)])


def main() -> None:
    sample_index, prediction, target = load_data()
    correlation = spatial_correlation(prediction, target)
    save_results(sample_index, correlation)

    print(f"Saved {OUTPUT}")
    print(f"Mean spatial correlation: {np.nanmean(correlation):.6f}")


if __name__ == "__main__":
    main()
