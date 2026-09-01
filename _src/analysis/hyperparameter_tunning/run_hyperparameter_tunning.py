#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import math
import random
from pathlib import Path
from typing import Any

import yaml


LATENT_CHANNELS = 4
DROPOUT_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SEQUENCE_HISTORY_LENGTHS = list(range(10))
CHANNEL_PRESETS = {
    "small": {
        "spatial_channels": [24, 48],
        "temporal_channels": 48,
        "temporal_layers": 1,
        "decoder_channels": [24, 12],
    },
    "base": {
        "spatial_channels": [32, 64, 96],
        "temporal_channels": 96,
        "temporal_layers": 2,
        "decoder_channels": [32, 16],
    },
    "wide": {
        "spatial_channels": [64, 128, 192],
        "temporal_channels": 192,
        "temporal_layers": 3,
        "decoder_channels": [64, 32],
    },
    "very_wide": {
        "spatial_channels": [96, 192, 256],
        "temporal_channels": 256,
        "temporal_layers": 3,
        "decoder_channels": [96, 48],
    },
    "overfit": {
        "spatial_channels": [128, 256, 384],
        "temporal_channels": 384,
        "temporal_layers": 4,
        "decoder_channels": [128, 64],
    },
}
BATCH_VALUES = [64, 128, 256]
WEIGHT_DECAY_VALUES = [0.0, 1.0e-6, 1.0e-5, 1.0e-4]
LOSS_VALUES = ["huber", "mse"]


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def experiment_dir() -> Path:
    return Path(__file__).resolve().parent


def base_config_path() -> Path:
    return project_dir() / "analysis/temporal_bottleneck/config.yaml"


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


def configs_dir() -> Path:
    return experiment_dir() / "configs"


def relative_output_dir(trial: str) -> Path:
    return Path("analysis/hyperparameter_tunning/outputs") / trial


def config_path(trial: str) -> Path:
    return configs_dir() / f"{trial}.yaml"


def summary_path() -> Path:
    return project_dir() / "analysis/hyperparameter_tunning/outputs/summary.csv"


def sequence_offsets(history_length: int) -> list[int]:
    if history_length < 0:
        raise ValueError("history_length must be >= 0")
    return list(range(-history_length, 1))


def log_uniform(rng: random.Random, low: float = 1.0e-5, high: float = 1.0e-2) -> float:
    return 10 ** rng.uniform(math.log10(low), math.log10(high))


def set_trial_paths(config: dict[str, Any], trial: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    out = relative_output_dir(trial)
    config["train"]["checkpoint"] = str(out / "temporal_bottleneck.pt")
    config["train"]["history_csv"] = str(out / "history.csv")
    config["output"]["dir"] = str(out)
    config["output"]["predictions"] = str(out / "test_predictions_denorm.npz")
    config["output"]["r2_series"] = str(out / "test_r2_series.csv")
    config["output"]["summary"] = str(out / "evaluation_summary.txt")
    config["output"]["stats"] = str(out / "norm_stats.npz")
    return config


def apply_trial(
    base: dict[str, Any],
    *,
    trial: str,
    lr: float,
    channel_preset: str,
    history_length: int,
    dropout: float,
    batch_size: int,
    weight_decay: float,
    loss: str,
) -> dict[str, Any]:
    cfg = set_trial_paths(base, trial)
    channels = CHANNEL_PRESETS[channel_preset]
    cfg["data"]["temporal_offsets"] = sequence_offsets(history_length)
    cfg["data"]["batch_size"] = batch_size
    cfg["model"]["latent_channels"] = LATENT_CHANNELS
    cfg["model"]["spatial_channels"] = channels["spatial_channels"]
    cfg["model"]["temporal_channels"] = channels["temporal_channels"]
    cfg["model"]["temporal_layers"] = channels["temporal_layers"]
    cfg["model"]["decoder_channels"] = channels["decoder_channels"]
    cfg["model"]["dropout"] = dropout
    cfg["train"]["lr"] = lr
    cfg["train"]["weight_decay"] = weight_decay
    cfg["train"]["loss"] = loss
    cfg["train"]["patience"] = 40
    cfg["train"]["latent_smooth_weight"] = 0
    cfg["train"]["latent_corr_weight"] = 0
    return cfg


def materialize_random(num_trials: int, seed: int) -> None:
    base = load_yaml(base_config_path())
    rng = random.Random(seed)
    required = [
        # Current best-like setting.
        {"lr": 1.0e-3, "channel_preset": "base", "history_length": 4, "dropout": 0.15, "batch_size": 64, "weight_decay": 1.0e-4, "loss": "huber"},
        # Explicit overfit probes.
        {"lr": 3.0e-4, "channel_preset": "overfit", "history_length": 4, "dropout": 0.0, "batch_size": 64, "weight_decay": 0.0, "loss": "mse"},
        {"lr": 1.0e-3, "channel_preset": "overfit", "history_length": 9, "dropout": 0.0, "batch_size": 64, "weight_decay": 0.0, "loss": "mse"},
        {"lr": 3.0e-4, "channel_preset": "very_wide", "history_length": 4, "dropout": 0.05, "batch_size": 64, "weight_decay": 1.0e-6, "loss": "huber"},
        # Strong regularization anchors.
        {"lr": 3.0e-4, "channel_preset": "base", "history_length": 4, "dropout": 0.30, "batch_size": 128, "weight_decay": 1.0e-4, "loss": "huber"},
        {"lr": 1.0e-4, "channel_preset": "wide", "history_length": 4, "dropout": 0.40, "batch_size": 128, "weight_decay": 1.0e-4, "loss": "huber"},
    ]
    trials = list(required)
    seen = {tuple(sorted(trial.items())) for trial in trials}
    while len(trials) < num_trials:
        trial = {
            "lr": log_uniform(rng),
            "channel_preset": rng.choice(list(CHANNEL_PRESETS)),
            "history_length": rng.choice(SEQUENCE_HISTORY_LENGTHS),
            "dropout": rng.choice(DROPOUT_VALUES),
            "batch_size": rng.choice(BATCH_VALUES),
            "weight_decay": rng.choice(WEIGHT_DECAY_VALUES),
            "loss": rng.choice(LOSS_VALUES),
        }
        signature = tuple(sorted((key, round(value, 12) if isinstance(value, float) else value) for key, value in trial.items()))
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)

    for idx, trial_params in enumerate(trials, start=1):
        trial = f"trial_{idx:03d}"
        cfg = apply_trial(base, trial=trial, **trial_params)
        save_yaml(config_path(trial), cfg)
    print(f"created {len(trials)} configs in {configs_dir()}", flush=True)


def materialize_grid() -> None:
    base = load_yaml(base_config_path())
    idx = 0
    for channel_preset in CHANNEL_PRESETS:
        for history_length in SEQUENCE_HISTORY_LENGTHS:
            for dropout in DROPOUT_VALUES:
                idx += 1
                trial = f"grid_{idx:04d}"
                cfg = apply_trial(
                    base,
                    trial=trial,
                    lr=1.0e-3,
                    channel_preset=channel_preset,
                    history_length=history_length,
                    dropout=dropout,
                    batch_size=64,
                    weight_decay=0.0 if channel_preset in {"very_wide", "overfit"} else 1.0e-5,
                    loss="huber",
                )
                save_yaml(config_path(trial), cfg)
    print(f"created {idx} grid configs in {configs_dir()}", flush=True)


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


def best_history(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def as_float(row: dict[str, str], key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except ValueError:
            return float("nan")

    best = max(rows, key=lambda row: as_float(row, "val_r2"))
    return {f"best_{key}": maybe_float(value) for key, value in best.items()}


def maybe_float(value: Any) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value or ""


def available_trials() -> list[str]:
    return [path.stem for path in sorted(configs_dir().glob("*.yaml"))]


def summarize() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in available_trials():
        cfg = load_yaml(config_path(trial))
        history_length = -min(int(v) for v in cfg["data"]["temporal_offsets"])
        row: dict[str, Any] = {
            "trial": trial,
            "seed": cfg["seed"],
            "history_length": history_length,
            "temporal_offsets": " ".join(str(v) for v in cfg["data"]["temporal_offsets"]),
            "batch_size": cfg["data"]["batch_size"],
            "latent_channels": cfg["model"]["latent_channels"],
            "spatial_channels": " ".join(str(v) for v in cfg["model"]["spatial_channels"]),
            "temporal_channels": cfg["model"]["temporal_channels"],
            "temporal_layers": cfg["model"]["temporal_layers"],
            "decoder_channels": " ".join(str(v) for v in cfg["model"]["decoder_channels"]),
            "dropout": cfg["model"]["dropout"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "loss": cfg["train"]["loss"],
            "latent_smooth_weight": cfg["train"]["latent_smooth_weight"],
            "latent_corr_weight": cfg["train"]["latent_corr_weight"],
        }
        train_r2 = row.get("best_train_r2")
        row.update(best_history(project_dir() / cfg["train"]["history_csv"]))
        row.update(parse_metrics(project_dir() / cfg["output"]["summary"]))
        if "best_train_r2" in row and "test_r2" in row:
            row["train_test_gap"] = float(row["best_train_r2"]) - float(row["test_r2"])
        rows.append(row)
    rows.sort(key=sort_key, reverse=True)
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
    for row in rows[:15]:
        print(
            f"{row['trial']} val_r2={row.get('best_val_r2', '')} test_r2={row.get('test_r2', '')} "
            f"gap={row.get('train_test_gap', '')} hist={row.get('history_length', '')} "
            f"dropout={row.get('dropout', '')} lr={row.get('lr', '')} channels={row.get('spatial_channels', '')}",
            flush=True,
        )
    return rows


def sort_key(row: dict[str, Any]) -> float:
    try:
        return float(row.get("best_val_r2", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def parse_trials(raw: str) -> list[str]:
    if raw == "all":
        trials = available_trials()
        if not trials:
            raise RuntimeError("No configs found. Run materialize-random or materialize-grid first.")
        return trials
    return [part.strip() for part in raw.split(",") if part.strip()]


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
    parser = argparse.ArgumentParser(description="Broad hyperparameter tuning for temporal bottleneck overfitting/generalization.")
    parser.add_argument("command", choices=["materialize-random", "materialize-grid", "train", "evaluate", "train-evaluate", "resume", "summarize"])
    parser.add_argument("--num-trials", type=int, default=60, help="number of random trials to create")
    parser.add_argument("--seed", type=int, default=42, help="random seed for log-random learning rates")
    parser.add_argument("--trials", default="all", help="all, or comma-separated trial names")
    parser.add_argument("--skip-completed", action="store_true", help="skip trials that already have expected outputs")
    args = parser.parse_args()

    if args.command == "materialize-random":
        materialize_random(args.num_trials, args.seed)
        return
    if args.command == "materialize-grid":
        materialize_grid()
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
