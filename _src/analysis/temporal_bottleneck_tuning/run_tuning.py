#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import random
from pathlib import Path
from typing import Any

import yaml


LATENT_CHANNELS = 4
ZERO_LATENT_REG = {
    "latent_smooth_weight": 0,
    "latent_corr_weight": 0,
}

STAGE1_TRIALS = 16
ARCH_VARIANTS = [
    {"spatial_channels": [24, 48], "temporal_channels": 48, "temporal_layers": 1, "decoder_channels": [24, 12], "dropout_delta": 0.00},
    {"spatial_channels": [32, 64], "temporal_channels": 64, "temporal_layers": 1, "decoder_channels": [32, 16], "dropout_delta": 0.00},
    {"spatial_channels": [32, 64], "temporal_channels": 64, "temporal_layers": 2, "decoder_channels": [32, 16], "dropout_delta": 0.00},
    {"spatial_channels": [32, 64, 96], "temporal_channels": 64, "temporal_layers": 2, "decoder_channels": [32, 16], "dropout_delta": 0.00},
    {"spatial_channels": [32, 64, 96], "temporal_channels": 96, "temporal_layers": 2, "decoder_channels": [32, 16], "dropout_delta": 0.00},
    {"spatial_channels": [48, 96], "temporal_channels": 96, "temporal_layers": 2, "decoder_channels": [48, 24], "dropout_delta": 0.05},
    {"spatial_channels": [32, 64, 96], "temporal_channels": 96, "temporal_layers": 3, "decoder_channels": [32, 16], "dropout_delta": 0.05},
    {"spatial_channels": [48, 96, 128], "temporal_channels": 96, "temporal_layers": 2, "decoder_channels": [48, 24], "dropout_delta": 0.05},
]

TIME_WINDOWS = [
    [0],
    [-1, 0],
    [-2, -1, 0],
    [-4, -3, -2, -1, 0],
    [-8, -7, -6, -5, -4, -3, -2, -1, 0],
]

FINAL_SEEDS = [11, 42, 73]


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def tuning_dir() -> Path:
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


def configs_dir(stage: str) -> Path:
    return tuning_dir() / "configs" / stage


def outputs_dir(stage: str, trial: str) -> Path:
    return Path("analysis/temporal_bottleneck_tuning/outputs") / stage / trial


def config_path(stage: str, trial: str) -> Path:
    return configs_dir(stage) / f"{trial}.yaml"


def stage_summary_path(stage: str) -> Path:
    return project_dir() / "analysis/temporal_bottleneck_tuning/outputs" / stage / "summary.csv"


def apply_common_settings(config: dict[str, Any], stage: str, trial: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    out = outputs_dir(stage, trial)
    config["model"]["latent_channels"] = LATENT_CHANNELS
    config["train"].update(ZERO_LATENT_REG)
    config["train"]["checkpoint"] = str(out / "temporal_bottleneck.pt")
    config["train"]["history_csv"] = str(out / "history.csv")
    config["output"]["dir"] = str(out)
    config["output"]["predictions"] = str(out / "test_predictions_denorm.npz")
    config["output"]["r2_series"] = str(out / "test_r2_series.csv")
    config["output"]["summary"] = str(out / "evaluation_summary.txt")
    config["output"]["stats"] = str(out / "norm_stats.npz")
    return config


def make_stage1_configs(seed: int = 42) -> list[Path]:
    base = load_yaml(base_config_path())
    rng = random.Random(seed)
    space = {
        "dropout": [0.0, 0.05, 0.1, 0.2],
        "weight_decay": [1e-5, 1e-4, 3e-4, 1e-3],
        "lr": [1e-4, 3e-4, 1e-3],
        "batch_size": [64, 128, 256],
        "loss": ["huber", "mse"],
        "patience": [15, 25, 40],
    }
    required = [
        {"dropout": 0.05, "weight_decay": 1e-5, "lr": 3e-4, "batch_size": 128, "loss": "huber", "patience": 25},
        {"dropout": 0.10, "weight_decay": 1e-4, "lr": 3e-4, "batch_size": 128, "loss": "huber", "patience": 25},
        {"dropout": 0.20, "weight_decay": 3e-4, "lr": 3e-4, "batch_size": 128, "loss": "huber", "patience": 25},
        {"dropout": 0.05, "weight_decay": 1e-4, "lr": 1e-4, "batch_size": 128, "loss": "huber", "patience": 40},
    ]
    trials = list(required)
    seen = {tuple(sorted(trial.items())) for trial in trials}
    while len(trials) < STAGE1_TRIALS:
        trial = {key: rng.choice(values) for key, values in space.items()}
        signature = tuple(sorted(trial.items()))
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(trial)

    paths = []
    for idx, trial in enumerate(trials, start=1):
        name = f"trial_{idx:02d}"
        cfg = apply_common_settings(base, "stage1", name)
        cfg["model"]["dropout"] = trial["dropout"]
        cfg["train"]["weight_decay"] = trial["weight_decay"]
        cfg["train"]["lr"] = trial["lr"]
        cfg["train"]["loss"] = trial["loss"]
        cfg["train"]["patience"] = trial["patience"]
        cfg["data"]["batch_size"] = trial["batch_size"]
        path = config_path("stage1", name)
        save_yaml(path, cfg)
        paths.append(path)
    return paths


def parse_metrics(summary_path: Path) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    if not summary_path.exists():
        return metrics
    with summary_path.open("r") as handle:
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

    def value(row: dict[str, str], key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except ValueError:
            return float("nan")

    best = max(rows, key=lambda row: value(row, "val_r2"))
    return {f"best_{key}": _maybe_float(val) for key, val in best.items()}


def _maybe_float(value: Any) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value or ""


def available_trials(stage: str) -> list[str]:
    paths = sorted(configs_dir(stage).glob("*.yaml"))
    return [path.stem for path in paths]


def summarize(stage: str, output: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in available_trials(stage):
        cfg = load_yaml(config_path(stage, trial))
        summary_path = project_dir() / cfg["output"]["summary"]
        history_path = project_dir() / cfg["train"]["history_csv"]
        row: dict[str, Any] = {
            "stage": stage,
            "trial": trial,
            "seed": cfg["seed"],
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
            "patience": cfg["train"]["patience"],
            "latent_smooth_weight": cfg["train"]["latent_smooth_weight"],
            "latent_corr_weight": cfg["train"]["latent_corr_weight"],
        }
        row.update(best_history(history_path))
        row.update(parse_metrics(summary_path))
        rows.append(row)
    rows.sort(key=lambda row: _sort_metric(row), reverse=True)
    if output is None:
        output = stage_summary_path(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved summary: {output}", flush=True)
    for row in rows[:10]:
        print(
            f"{row['stage']}/{row['trial']} best_val_r2={row.get('best_val_r2', '')} "
            f"val_r2={row.get('val_r2', '')} test_r2={row.get('test_r2', '')}",
            flush=True,
        )
    return rows


def _sort_metric(row: dict[str, Any]) -> float:
    for key in ("best_val_r2", "val_r2"):
        try:
            return float(row.get(key, float("-inf")))
        except (TypeError, ValueError):
            continue
    return float("-inf")


def top_configs(stage: str, top_k: int) -> list[tuple[str, dict[str, Any]]]:
    rows = summarize(stage)
    selected = [row for row in rows if _sort_metric(row) != float("-inf")][:top_k]
    if len(selected) < top_k:
        raise RuntimeError(f"Need {top_k} completed trials in {stage}, found {len(selected)}")
    return [(row["trial"], load_yaml(config_path(stage, row["trial"]))) for row in selected]


def make_stage2_configs(top_k: int = 3, from_stage: str = "stage1") -> list[Path]:
    paths = []
    for source_trial, source_cfg in top_configs(from_stage, top_k):
        for idx, variant in enumerate(ARCH_VARIANTS, start=1):
            name = f"{source_trial}_arch_{idx:02d}"
            cfg = apply_common_settings(source_cfg, "stage2", name)
            cfg["model"]["spatial_channels"] = variant["spatial_channels"]
            cfg["model"]["temporal_channels"] = variant["temporal_channels"]
            cfg["model"]["temporal_layers"] = variant["temporal_layers"]
            cfg["model"]["decoder_channels"] = variant["decoder_channels"]
            cfg["model"]["dropout"] = min(0.3, max(0.0, float(source_cfg["model"]["dropout"]) + variant["dropout_delta"]))
            path = config_path("stage2", name)
            save_yaml(path, cfg)
            paths.append(path)
    return paths


def make_stage3_configs(top_k: int = 2, from_stage: str = "stage2") -> list[Path]:
    paths = []
    for source_trial, source_cfg in top_configs(from_stage, top_k):
        for offsets in TIME_WINDOWS:
            label = "t" + "_".join(str(v).replace("-", "m") for v in offsets)
            name = f"{source_trial}_{label}"
            cfg = apply_common_settings(source_cfg, "stage3", name)
            cfg["data"]["temporal_offsets"] = offsets
            path = config_path("stage3", name)
            save_yaml(path, cfg)
            paths.append(path)
    return paths


def make_stage4_configs(top_k: int = 1, from_stage: str = "stage3") -> list[Path]:
    paths = []
    for source_trial, source_cfg in top_configs(from_stage, top_k):
        for seed in FINAL_SEEDS:
            name = f"{source_trial}_seed_{seed}"
            cfg = apply_common_settings(source_cfg, "stage4", name)
            cfg["seed"] = seed
            path = config_path("stage4", name)
            save_yaml(path, cfg)
            paths.append(path)
    return paths


def materialize(stage: str, top_k: int, from_stage: str) -> None:
    if stage == "stage1":
        paths = make_stage1_configs()
    elif stage == "stage2":
        paths = make_stage2_configs(top_k=top_k, from_stage=from_stage)
    elif stage == "stage3":
        paths = make_stage3_configs(top_k=top_k, from_stage=from_stage)
    elif stage == "stage4":
        paths = make_stage4_configs(top_k=top_k, from_stage=from_stage)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    print(f"created {len(paths)} configs in {configs_dir(stage)}", flush=True)


def parse_trials(stage: str, raw: str) -> list[str]:
    if raw == "all":
        trials = available_trials(stage)
        if not trials:
            raise RuntimeError(f"No configs found for {stage}. Run materialize first.")
        return trials
    return [part.strip() for part in raw.split(",") if part.strip()]


def trial_is_complete(stage: str, trial: str, *, require_evaluation: bool) -> bool:
    cfg = load_yaml(config_path(stage, trial))
    root = project_dir()
    checkpoint = root / cfg["train"]["checkpoint"]
    summary = root / cfg["output"]["summary"]
    if require_evaluation:
        return checkpoint.exists() and summary.exists()
    return checkpoint.exists()


def run_trials(stage: str, trials: list[str], *, train: bool, evaluate: bool, skip_completed: bool = False) -> None:
    module = load_temporal_module()
    for trial in trials:
        path = config_path(stage, trial)
        cfg = load_yaml(path)
        if skip_completed and trial_is_complete(stage, trial, require_evaluation=evaluate):
            print(f"stage={stage} trial={trial} already complete; skipping", flush=True)
            continue
        print(f"stage={stage} trial={trial} config={path}", flush=True)
        if train:
            module.train(cfg)
        if evaluate:
            module.evaluate(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged tuning for the temporal bottleneck model with 4 feature maps.")
    parser.add_argument("command", choices=["materialize", "train", "evaluate", "train-evaluate", "resume", "summarize"])
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "stage4"], required=True)
    parser.add_argument("--trials", default="all", help="all, or comma-separated trial names")
    parser.add_argument("--skip-completed", action="store_true", help="skip trials that already have the expected checkpoint/evaluation output")
    parser.add_argument("--top-k", type=int, default=3, help="number of previous-stage winners to use when materializing")
    parser.add_argument("--from-stage", default="", help="source stage for materializing stage2, stage3, or stage4")
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "materialize":
        default_source = {"stage2": "stage1", "stage3": "stage2", "stage4": "stage3"}.get(args.stage, "")
        materialize(args.stage, args.top_k, args.from_stage or default_source)
        return

    if args.command == "summarize":
        output = project_dir() / args.summary_output if args.summary_output else None
        summarize(args.stage, output)
        return

    trials = parse_trials(args.stage, args.trials)
    if args.command == "resume":
        run_trials(args.stage, trials, train=True, evaluate=True, skip_completed=True)
        return

    run_trials(
        args.stage,
        trials,
        train=args.command in {"train", "train-evaluate"},
        evaluate=args.command in {"evaluate", "train-evaluate"},
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
