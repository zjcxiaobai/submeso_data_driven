#!/usr/bin/env python3
"""Prepare physics diagnostics for the best-performing model_analysis test sample."""

from __future__ import annotations

import argparse
import csv
import json
import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
SOURCE_SCRIPT = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation/model_analysis_last_decoder_pca/prepare_physics_variables.py"
spec = importlib.util.spec_from_file_location("last_decoder_prepare_physics_variables", SOURCE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {SOURCE_SCRIPT}")
source_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_module)
FEATURE_NAMES = source_module.FEATURE_NAMES
calculate_fields = source_module.calculate_fields


DEFAULT_X = PROJECT_DIR / "analysis/model_analysis/data/X.npy"
DEFAULT_TIME_INDEX = PROJECT_DIR / "analysis/model_analysis/data/time_index.npy"
DEFAULT_PREDICTIONS = PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz"
DEFAULT_OUTPUT_DIR = HERE / "physics_variables_best_sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-path", type=Path, default=DEFAULT_X)
    parser.add_argument("--time-index", type=Path, default=DEFAULT_TIME_INDEX)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--sample-index", type=int, default=None, help="Optional explicit sample_index. Default is best test R2 sample.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dx-m", type=float, default=50_000.0)
    parser.add_argument("--dy-m", type=float, default=50_000.0)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--domain-height-m", type=float, default=1.0e6)
    parser.add_argument("--beta-reference", choices=("center", "south", "north"), default="center")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def best_sample_from_predictions(path: Path) -> tuple[int, int, float]:
    with np.load(path, allow_pickle=True) as data:
        if "sample_index" not in data.files or "r2" not in data.files:
            raise ValueError(f"{path} must contain sample_index and r2 arrays")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
    valid = np.where(np.isfinite(r2))[0]
    if valid.size == 0:
        raise ValueError("No finite test-sample R2 values found")
    pos = int(valid[np.argmax(r2[valid])])
    return int(sample_index[pos]), pos, float(r2[pos])


def main() -> None:
    args = parse_args()
    x_path = args.x_path.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    x = np.load(x_path, mmap_mode="r")
    if x.ndim != 4 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected X shape (N,{len(FEATURE_NAMES)},H,W), got {x.shape}")

    if args.sample_index is None:
        sample_index, test_position, selected_r2 = best_sample_from_predictions(predictions_path)
    else:
        sample_index = int(args.sample_index)
        test_position = -1
        selected_r2 = float("nan")

    if not 0 <= sample_index < x.shape[0]:
        raise ValueError(f"sample_index {sample_index} is outside X range [0, {x.shape[0]})")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / "physics_variables.npz", output_dir / "physics_variable_manifest.csv", output_dir / "selected_samples.csv", output_dir / "metadata.json"]
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"{output_dir} already contains outputs; pass --overwrite")

    source = {name: np.asarray(x[sample_index, i], dtype=np.float64) for i, name in enumerate(FEATURE_NAMES)}
    fields, definitions = calculate_fields(
        source,
        args.dx_m,
        args.dy_m,
        args.f0,
        args.beta,
        args.domain_height_m,
        args.beta_reference,
    )
    field_stacks = {name: field[None, ...].astype(np.float32) for name, field in fields.items()}
    for name, field in field_stacks.items():
        np.save(output_dir / f"{name}.npy", field)

    time_value = None
    if args.time_index.exists():
        time_all = np.load(args.time_index, mmap_mode="r")
        time_value = int(time_all[sample_index])

    np.savez_compressed(
        output_dir / "physics_variables.npz",
        sample_index=np.asarray([sample_index], dtype=np.int64),
        test_position=np.asarray([test_position], dtype=np.int64),
        time_index=np.asarray([] if time_value is None else [time_value], dtype=np.int64),
        selected_r2=np.asarray([selected_r2], dtype=np.float64),
        **field_stacks,
    )

    with (output_dir / "physics_variable_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "units", "definition", "filename", "shape", "dtype"])
        writer.writeheader()
        for name, field in field_stacks.items():
            units, definition = definitions[name]
            writer.writerow({
                "variable": name,
                "units": units,
                "definition": definition,
                "filename": f"{name}.npy",
                "shape": "x".join(str(value) for value in field.shape),
                "dtype": str(field.dtype),
            })

    with (output_dir / "selected_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "sample_index", "test_position", "time_index", "r2"])
        writer.writeheader()
        writer.writerow({
            "rank": 1,
            "sample_index": sample_index,
            "test_position": test_position,
            "time_index": "" if time_value is None else time_value,
            "r2": selected_r2,
        })

    metadata = {
        "selection": "best_test_sample_by_r2" if args.sample_index is None else "single_sample",
        "sample_index": sample_index,
        "test_position": test_position,
        "time_index": time_value,
        "selected_r2": selected_r2,
        "x_path": str(x_path),
        "predictions": str(predictions_path),
        "features": FEATURE_NAMES,
        "shape_nyx": list(next(iter(field_stacks.values())).shape),
        "variable_count": len(field_stacks),
        "dx_m": args.dx_m,
        "dy_m": args.dy_m,
        "periodic_x": True,
        "periodic_y": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(field_stacks)} physics variables for sample {sample_index} to {output_dir}")


if __name__ == "__main__":
    main()
