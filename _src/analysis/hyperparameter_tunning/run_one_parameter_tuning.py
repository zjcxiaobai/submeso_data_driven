#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
from pathlib import Path
from typing import Any

import yaml


BASELINE = {
    "lr": 1.0e-3,
    "channel_preset": "base_deep",
    "history_length": 4,
    "dropout": 0.15,
    "batch_size": 64,
    "weight_decay": 1.0e-4,
    "loss": "huber",
    "patience": 40,
    "latent_channels": 4,
}

LEARNING_RATES = [
    1.0e-5,
    1.8e-5,
    3.2e-5,
    5.6e-5,
    1.0e-4,
    1.8e-4,
    3.2e-4,
    5.6e-4,
    1.0e-3,
    1.8e-3,
    3.2e-3,
    5.6e-3,
    1.0e-2,
]
DROPOUT_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
HISTORY_LENGTHS = list(range(10))
CHANNEL_PRESETS = {
    "tiny": {
        "spatial_channels": [16, 32],
        "temporal_channels": 32,
        "temporal_layers": 1,
        "decoder_channels": [16, 8],
    },
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
    "base_deep": {
        "spatial_channels": [32, 64, 96],
        "temporal_channels": 96,
        "temporal_layers": 3,
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
SWEEPS = {"learning_rate", "channel_width", "history_length", "dropout"}


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


def configs_dir(sweep: str) -> Path:
    return experiment_dir() / "configs_one_at_time" / sweep


def config_path(sweep: str, trial: str) -> Path:
    return configs_dir(sweep) / f"{trial}.yaml"


def output_dir(sweep: str, trial: str) -> Path:
    return Path("analysis/hyperparameter_tunning/outputs_one_at_time") / sweep / trial


def summary_path(sweep: str) -> Path:
    return project_dir() / "analysis/hyperparameter_tunning/outputs_one_at_time" / sweep / "summary.csv"


def sequence_offsets(history_length: int) -> list[int]:
    return list(range(-history_length, 1))


def trial_name(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1e}".replace(".", "p").replace("-", "m")
    return str(value).replace(".", "p").replace("-", "m")


def apply_settings(
    base: dict[str, Any],
    *,
    sweep: str,
    trial: str,
    lr: float,
    channel_preset: str,
    history_length: int,
    dropout: float,
    batch_size: int,
    weight_decay: float,
    loss: str,
    patience: int,
    latent_channels: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    channels = CHANNEL_PRESETS[channel_preset]
    out = output_dir(sweep, trial)
    cfg["data"]["temporal_offsets"] = sequence_offsets(history_length)
    cfg["data"]["batch_size"] = batch_size
    cfg["model"]["latent_channels"] = latent_channels
    cfg["model"]["spatial_channels"] = channels["spatial_channels"]
    cfg["model"]["temporal_channels"] = channels["temporal_channels"]
    cfg["model"]["temporal_layers"] = channels["temporal_layers"]
    cfg["model"]["decoder_channels"] = channels["decoder_channels"]
    cfg["model"]["dropout"] = dropout
    cfg["train"]["lr"] = lr
    cfg["train"]["weight_decay"] = weight_decay
    cfg["train"]["loss"] = loss
    cfg["train"]["patience"] = patience
    cfg["train"]["latent_smooth_weight"] = 0
    cfg["train"]["latent_corr_weight"] = 0
    cfg["train"]["checkpoint"] = str(out / "temporal_bottleneck.pt")
    cfg["train"]["history_csv"] = str(out / "history.csv")
    cfg["output"]["dir"] = str(out)
    cfg["output"]["predictions"] = str(out / "test_predictions_denorm.npz")
    cfg["output"]["r2_series"] = str(out / "test_r2_series.csv")
    cfg["output"]["summary"] = str(out / "evaluation_summary.txt")
    cfg["output"]["stats"] = str(out / "norm_stats.npz")
    return cfg


def sweep_values(sweep: str) -> list[tuple[str, Any]]:
    if sweep == "learning_rate":
        return [(f"lr_{trial_name(v)}", v) for v in LEARNING_RATES]
    if sweep == "channel_width":
        return [(f"channels_{name}", name) for name in CHANNEL_PRESETS]
    if sweep == "history_length":
        return [(f"history_{v}", v) for v in HISTORY_LENGTHS]
    if sweep == "dropout":
        return [(f"dropout_{trial_name(v)}", v) for v in DROPOUT_VALUES]
    raise ValueError(f"Unknown sweep: {sweep}")


def materialize(sweep: str) -> None:
    base = load_yaml(base_config_path())
    sweeps = sorted(SWEEPS) if sweep == "all" else [sweep]
    total = 0
    for one_sweep in sweeps:
        for trial, value in sweep_values(one_sweep):
            settings = dict(BASELINE)
            if one_sweep == "learning_rate":
                settings["lr"] = value
            elif one_sweep == "channel_width":
                settings["channel_preset"] = value
            elif one_sweep == "history_length":
                settings["history_length"] = value
            elif one_sweep == "dropout":
                settings["dropout"] = value
            cfg = apply_settings(base, sweep=one_sweep, trial=trial, **settings)
            save_yaml(config_path(one_sweep, trial), cfg)
            total += 1
        print(f"created {len(sweep_values(one_sweep))} configs in {configs_dir(one_sweep)}", flush=True)
    print(f"created {total} total one-parameter configs", flush=True)


def available_trials(sweep: str) -> list[str]:
    return [path.stem for path in sorted(configs_dir(sweep).glob("*.yaml"))]


def parse_trials(sweep: str, raw: str) -> list[str]:
    if raw == "all":
        trials = available_trials(sweep)
        if not trials:
            raise RuntimeError(f"No configs found for {sweep}. Run materialize first.")
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


def summarize_one(sweep: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in available_trials(sweep):
        cfg = load_yaml(config_path(sweep, trial))
        history_length = -min(int(v) for v in cfg["data"]["temporal_offsets"])
        row: dict[str, Any] = {
            "sweep": sweep,
            "trial": trial,
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
        }
        row.update(best_history(project_dir() / cfg["train"]["history_csv"]))
        row.update(parse_metrics(project_dir() / cfg["output"]["summary"]))
        if "best_train_r2" in row and "test_r2" in row:
            row["train_test_gap"] = float(row["best_train_r2"]) - float(row["test_r2"])
        rows.append(row)
    rows.sort(key=lambda row: float(row.get("best_val_r2", float("-inf"))), reverse=True)
    out = summary_path(sweep)
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
    for row in rows[:10]:
        print(
            f"{row['sweep']}/{row['trial']} val_r2={row.get('best_val_r2', '')} "
            f"test_r2={row.get('test_r2', '')} gap={row.get('train_test_gap', '')}",
            flush=True,
        )
    return rows


def summarize(sweep: str) -> None:
    sweeps = sorted(SWEEPS) if sweep == "all" else [sweep]
    for one_sweep in sweeps:
        summarize_one(one_sweep)


def trial_complete(sweep: str, trial: str, require_evaluation: bool) -> bool:
    cfg = load_yaml(config_path(sweep, trial))
    checkpoint = project_dir() / cfg["train"]["checkpoint"]
    summary = project_dir() / cfg["output"]["summary"]
    if require_evaluation:
        return checkpoint.exists() and summary.exists()
    return checkpoint.exists()


def run_one_sweep(sweep: str, trials: list[str], *, train: bool, evaluate: bool, skip_completed: bool) -> None:
    module = load_temporal_module()
    for trial in trials:
        if skip_completed and trial_complete(sweep, trial, require_evaluation=evaluate):
            print(f"{sweep}/{trial} already complete; skipping", flush=True)
            continue
        cfg = load_yaml(config_path(sweep, trial))
        print(f"sweep={sweep} trial={trial} config={config_path(sweep, trial)}", flush=True)
        if train:
            module.train(cfg)
        if evaluate:
            module.evaluate(cfg)


def run(sweep: str, trials_raw: str, *, train: bool, evaluate: bool, skip_completed: bool) -> None:
    sweeps = sorted(SWEEPS) if sweep == "all" else [sweep]
    for one_sweep in sweeps:
        trials = parse_trials(one_sweep, trials_raw)
        run_one_sweep(one_sweep, trials, train=train, evaluate=evaluate, skip_completed=skip_completed)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-parameter-at-a-time hyperparameter tuning.")
    parser.add_argument("command", choices=["materialize", "train", "evaluate", "train-evaluate", "resume", "summarize"])
    parser.add_argument("--sweep", choices=sorted(SWEEPS) + ["all"], default="all")
    parser.add_argument("--trials", default="all", help="all, or comma-separated trial names for a single sweep")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    if args.command == "materialize":
        materialize(args.sweep)
        return
    if args.command == "summarize":
        summarize(args.sweep)
        return
    if args.command == "resume":
        run(args.sweep, args.trials, train=True, evaluate=True, skip_completed=True)
        return
    run(
        args.sweep,
        args.trials,
        train=args.command in {"train", "train-evaluate"},
        evaluate=args.command in {"evaluate", "train-evaluate"},
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
