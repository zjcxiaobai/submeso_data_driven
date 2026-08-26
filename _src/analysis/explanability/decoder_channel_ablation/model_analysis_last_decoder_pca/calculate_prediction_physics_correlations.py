#!/usr/bin/env python3
"""Correlate physics diagnostics with the final denormalized VHF prediction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from last_decoder_pca_analysis import PHYSICS_VARIABLES, pearsonr_flat


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_PHYSICS = HERE / "physics_variables_sample_1379/physics_variables.npz"
DEFAULT_PREDICTIONS = PROJECT_DIR / "analysis/model_analysis/outputs/test_predictions_denorm.npz"
DEFAULT_PCA = HERE / "last_decoder_pca_sample_1379/last_decoder_pca_results.npz"
DEFAULT_OUTPUT = HERE / "final_assessment_sample_1379/prediction_physics_correlations.csv"
DEFAULT_NPZ = HERE / "final_assessment_sample_1379/prediction_physics_correlations.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--prediction-key", choices=("prediction", "target"), default="prediction")
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA, help="Optional PCA result file used to add PC1 correlations for comparison.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npz-output", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--rank-by", choices=("abs_prediction", "prediction", "abs_pc1", "delta_abs"), default="abs_prediction")
    return parser.parse_args()


def load_prediction_maps(path: Path, sample_indices: np.ndarray, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "sample_index" not in data.files or key not in data.files:
            raise ValueError(f"{path} must contain sample_index and {key}")
        all_samples = np.asarray(data["sample_index"], dtype=np.int64).reshape(-1)
        all_maps = np.asarray(data[key], dtype=np.float64)
    if all_maps.ndim == 4 and all_maps.shape[1] == 1:
        all_maps = all_maps[:, 0]
    elif all_maps.ndim != 3:
        raise ValueError(f"Expected {key} shape (N,1,H,W) or (N,H,W), got {all_maps.shape}")
    lookup = {int(sample): pos for pos, sample in enumerate(all_samples)}
    positions = []
    missing = []
    for sample in sample_indices:
        pos = lookup.get(int(sample))
        if pos is None:
            missing.append(int(sample))
        else:
            positions.append(pos)
    if missing:
        raise ValueError(f"{len(missing)} physics sample(s) are missing from {path}. First missing: {missing[:8]}")
    return all_maps[np.asarray(positions, dtype=np.int64)]


def load_pc1_lookup(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=True) as data:
        if "physics_variables" not in data.files or "physics_correlations" not in data.files:
            return {}
        variables = [str(value) for value in data["physics_variables"]]
        correlations = np.asarray(data["physics_correlations"], dtype=np.float64)
    if correlations.ndim != 2 or correlations.shape[1] < 1:
        return {}
    return {name: float(correlations[index, 0]) for index, name in enumerate(variables)}


def rank_key(row: dict[str, float | str], mode: str) -> float:
    pred = float(row["prediction_r"])
    pc1 = float(row["pc1_r"])
    if mode == "prediction":
        return pred
    if mode == "abs_pc1":
        return abs(pc1) if np.isfinite(pc1) else -np.inf
    if mode == "delta_abs":
        return abs(abs(pred) - abs(pc1)) if np.isfinite(pc1) else -np.inf
    return abs(pred)


def main() -> None:
    args = parse_args()
    with np.load(args.physics, allow_pickle=True) as physics:
        if "sample_index" not in physics.files:
            raise ValueError(f"{args.physics} is missing sample_index")
        sample_indices = np.asarray(physics["sample_index"], dtype=np.int64).reshape(-1)
        prediction_maps = load_prediction_maps(args.predictions, sample_indices, args.prediction_key)
        pc1_lookup = load_pc1_lookup(args.pca)

        rows = []
        corr_by_sample = []
        variable_names = []
        labels = []
        for variable, label in PHYSICS_VARIABLES:
            if variable not in physics.files:
                continue
            field = np.asarray(physics[variable], dtype=np.float64)
            if field.ndim == 2:
                field = np.broadcast_to(field[None, ...], prediction_maps.shape)
            if field.ndim != 3:
                raise ValueError(f"Physics variable {variable} has shape {field.shape}, expected (N,H,W) or (H,W)")
            if field.shape[0] != prediction_maps.shape[0]:
                raise ValueError(f"Physics variable {variable} has {field.shape[0]} samples but prediction has {prediction_maps.shape[0]}")
            sample_r = np.asarray([pearsonr_flat(prediction_maps[i], field[i]) for i in range(prediction_maps.shape[0])], dtype=np.float64)
            pred_r = float(np.nanmean(sample_r))
            pred_r_sd = float(np.nanstd(sample_r, ddof=1)) if sample_r.size > 1 else float("nan")
            pc1_r = pc1_lookup.get(variable, float("nan"))
            rows.append({
                "physics_variable": variable,
                "display_label": label,
                "group": "input" if variable in {"U", "V", "W", "zeta", "strain", "T", "rho", "KPPhbl"} else "diagnostic",
                "n_samples": int(sample_r.size),
                "prediction_r": pred_r,
                "prediction_r_sd": pred_r_sd,
                "abs_prediction_r": abs(pred_r),
                "pc1_r": pc1_r,
                "abs_pc1_r": abs(pc1_r) if np.isfinite(pc1_r) else float("nan"),
                "abs_prediction_minus_abs_pc1": abs(pred_r) - abs(pc1_r) if np.isfinite(pc1_r) else float("nan"),
            })
            corr_by_sample.append(sample_r)
            variable_names.append(variable)
            labels.append(label)

    rows = sorted(rows, key=lambda row: rank_key(row, args.rank_by), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        fieldnames = [
            "rank",
            "physics_variable",
            "display_label",
            "group",
            "n_samples",
            "prediction_r",
            "prediction_r_sd",
            "abs_prediction_r",
            "pc1_r",
            "abs_pc1_r",
            "abs_prediction_minus_abs_pc1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        sample_index=sample_indices.astype(np.int64),
        prediction_key=args.prediction_key,
        physics_variables=np.asarray(variable_names),
        physics_labels=np.asarray(labels),
        prediction_correlations_by_sample=np.stack(corr_by_sample, axis=0).astype(np.float64) if corr_by_sample else np.empty((0, sample_indices.size)),
        mean_prediction_correlations=np.asarray([next(row["prediction_r"] for row in rows if row["physics_variable"] == name) for name in variable_names], dtype=np.float64),
        pc1_correlations=np.asarray([pc1_lookup.get(name, np.nan) for name in variable_names], dtype=np.float64),
    )
    print(f"Samples: {sample_indices.size}")
    print(f"Prediction source: {args.predictions} [{args.prediction_key}]")
    print(f"Saved {args.output}")
    print(f"Saved {args.npz_output}")
    print("Top variables:", ", ".join(row["physics_variable"] for row in rows[:10]))


if __name__ == "__main__":
    main()
