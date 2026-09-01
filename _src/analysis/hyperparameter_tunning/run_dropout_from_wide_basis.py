#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
from pathlib import Path
from typing import Any

import yaml


DROPOUT_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def basis_config_path() -> Path:
    return project_dir() / "analysis/hyperparameter_tunning/configs_one_at_time/channel_width/channels_wide.yaml"


def configs_dir() -> Path:
    return experiment_dir() / "configs_dropout_from_wide_basis"


def output_dir(trial: str) -> Path:
    return Path("analysis/hyperparameter_tunning/outputs_dropout_from_wide_basis") / trial


def config_path(trial: str) -> Path:
    return configs_dir() / f"{trial}.yaml"


def summary_path() -> Path:
    return project_dir() / "analysis/hyperparameter_tunning/outputs_dropout_from_wide_basis/summary.csv"


def load_temporal_module():
    module_path = project_dir() / "analysis/temporal_bottleneck/run_temporal_bottleneck.py"
    spec = importlib.util.spec_from_file_location("temporal_bottleneck_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def trial_name(dropout: float) -> str:
    return f"dropout_{dropout:.2f}".replace(".", "p")


def materialize() -> None:
    basis = load_yaml(basis_config_path())
    for dropout in DROPOUT_VALUES:
        trial = trial_name(dropout)
        cfg = copy.deepcopy(basis)
        out = output_dir(trial)
        cfg["model"]["dropout"] = dropout
        cfg["train"]["checkpoint"] = str(out / "temporal_bottleneck.pt")
        cfg["train"]["history_csv"] = str(out / "history.csv")
        cfg["output"]["dir"] = str(out)
        cfg["output"]["predictions"] = str(out / "test_predictions_denorm.npz")
        cfg["output"]["r2_series"] = str(out / "test_r2_series.csv")
        cfg["output"]["summary"] = str(out / "evaluation_summary.txt")
        cfg["output"]["stats"] = str(out / "norm_stats.npz")
        save_yaml(config_path(trial), cfg)
    print(f"created {len(DROPOUT_VALUES)} configs in {configs_dir()}", flush=True)


def available_trials() -> list[str]:
    return [path.stem for path in sorted(configs_dir().glob("*.yaml"))]


def parse_trials(raw: str) -> list[str]:
    if raw == "all":
        trials = available_trials()
        if not trials:
            raise RuntimeError("No configs found. Run materialize first.")
        return trials
    return [part.strip() for part in raw.split(",") if part.strip()]


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


def summarize() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in available_trials():
        cfg = load_yaml(config_path(trial))
        row: dict[str, Any] = {
            "trial": trial,
            "dropout": cfg["model"]["dropout"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "history_length": -min(int(v) for v in cfg["data"]["temporal_offsets"]),
            "temporal_offsets": " ".join(str(v) for v in cfg["data"]["temporal_offsets"]),
            "batch_size": cfg["data"]["batch_size"],
            "latent_channels": cfg["model"]["latent_channels"],
            "spatial_channels": " ".join(str(v) for v in cfg["model"]["spatial_channels"]),
            "temporal_channels": cfg["model"]["temporal_channels"],
            "temporal_layers": cfg["model"]["temporal_layers"],
            "decoder_channels": " ".join(str(v) for v in cfg["model"]["decoder_channels"]),
            "loss": cfg["train"]["loss"],
        }
        row.update(best_history(project_dir() / cfg["train"]["history_csv"]))
        row.update(parse_metrics(project_dir() / cfg["output"]["summary"]))
        if "best_train_r2" in row and "test_r2" in row:
            row["train_test_gap"] = float(row["best_train_r2"]) - float(row["test_r2"])
        rows.append(row)
    rows.sort(key=lambda row: float(row.get("best_val_r2", float("-inf"))), reverse=True)
    out = summary_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved summary: {out}", flush=True)
    for row in rows:
        print(
            f"{row['trial']} dropout={row['dropout']} val_r2={row.get('best_val_r2', '')} "
            f"test_r2={row.get('test_r2', '')} gap={row.get('train_test_gap', '')}",
            flush=True,
        )
    return rows


def trial_complete(trial: str, require_evaluation: bool) -> bool:
    cfg = load_yaml(config_path(trial))
    checkpoint = project_dir() / cfg["train"]["checkpoint"]
    summary = project_dir() / cfg["output"]["summary"]
    if require_evaluation:
        return checkpoint.exists() and summary.exists()
    return checkpoint.exists()


def run_trials(trials: list[str], *, train: bool, evaluate: bool, skip_completed: bool) -> None:
    module = load_temporal_module()
    for trial in trials:
        if skip_completed and trial_complete(trial, require_evaluation=evaluate):
            print(f"{trial} already complete; skipping", flush=True)
            continue
        cfg = load_yaml(config_path(trial))
        print(f"trial={trial} config={config_path(trial)}", flush=True)
        if train:
            module.train(cfg)
        if evaluate:
            module.evaluate(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune dropout using the channels_wide config as the fixed basis.")
    parser.add_argument("command", choices=["materialize", "train", "evaluate", "train-evaluate", "resume", "summarize"])
    parser.add_argument("--trials", default="all", help="all, or comma-separated trial names")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    if args.command == "materialize":
        materialize()
        return
    if args.command == "summarize":
        summarize()
        return
    trials = parse_trials(args.trials)
    if args.command == "resume":
        run_trials(trials, train=True, evaluate=True, skip_completed=True)
        return
    run_trials(
        trials,
        train=args.command in {"train", "train-evaluate"},
        evaluate=args.command in {"evaluate", "train-evaluate"},
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
