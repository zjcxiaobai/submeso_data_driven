#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml


TRIAL = "temporal_layers5"


def project_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def config_path() -> Path:
    return experiment_dir() / "configs" / f"{TRIAL}.yaml"


def summary_path() -> Path:
    return project_dir() / "analysis/hyperparameter_tunning/temporal_layers5/outputs/summary.csv"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def load_model_analysis_module():
    module_path = project_dir() / "analysis/model_analysis/run_model_analysis.py"
    if str(project_dir()) not in sys.path:
        sys.path.insert(0, str(project_dir()))
    spec = importlib.util.spec_from_file_location("model_analysis_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_metrics(path: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    if not path.exists():
        return metrics
    with path.open("r") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw = line.strip().split(":", 1)
            value = raw.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics


def maybe_float(value: Any) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value or ""


def best_history(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def score(row: dict[str, str]) -> float:
        try:
            return float(row.get("val_r2", "nan"))
        except ValueError:
            return float("nan")

    best = max(rows, key=score)
    return {f"best_{key}": maybe_float(value) for key, value in best.items()}


def complete(require_evaluation: bool) -> bool:
    cfg = load_yaml(config_path())
    checkpoint = project_dir() / cfg["train"]["checkpoint"]
    summary = project_dir() / cfg["output"]["summary"]
    if require_evaluation:
        return checkpoint.exists() and summary.exists()
    return checkpoint.exists()


def run(*, train: bool, evaluate: bool, skip_completed: bool) -> None:
    if skip_completed and complete(require_evaluation=evaluate):
        print(f"{TRIAL} already complete; skipping", flush=True)
        return
    cfg = load_yaml(config_path())
    module = load_model_analysis_module()
    print(f"trial={TRIAL} config={config_path()}", flush=True)
    if train:
        module.train(cfg)
    if evaluate:
        module.evaluate(cfg)


def summarize() -> None:
    cfg = load_yaml(config_path())
    row: dict[str, Any] = {
        "trial": TRIAL,
        "temporal_offsets": " ".join(str(v) for v in cfg["data"]["temporal_offsets"]),
        "spatial_channels": " ".join(str(v) for v in cfg["model"]["spatial_channels"]),
        "temporal_channels": cfg["model"]["temporal_channels"],
        "temporal_layers": cfg["model"]["temporal_layers"],
        "decoder_channels": " ".join(str(v) for v in cfg["model"]["decoder_channels"]),
        "latent_channels": cfg["model"]["latent_channels"],
        "dropout": cfg["model"]["dropout"],
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "loss": cfg["train"]["loss"],
    }
    row.update(best_history(project_dir() / cfg["train"]["history_csv"]))
    row.update(parse_metrics(project_dir() / cfg["output"]["summary"]))
    if "best_train_r2" in row and "test_r2" in row:
        row["train_test_gap"] = float(row["best_train_r2"]) - float(row["test_r2"])
    out = summary_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"saved summary: {out}", flush=True)
    print(
        f"{TRIAL} val_r2={row.get('best_val_r2', '')} "
        f"test_r2={row.get('test_r2', '')} gap={row.get('train_test_gap', '')}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate current model_analysis config with temporal_layers changed to 5.")
    parser.add_argument("command", choices=["train", "evaluate", "train-evaluate", "resume", "summarize"])
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize()
        return
    if args.command == "resume":
        run(train=True, evaluate=True, skip_completed=True)
        return
    run(
        train=args.command in {"train", "train-evaluate"},
        evaluate=args.command in {"evaluate", "train-evaluate"},
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
