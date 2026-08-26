#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import yaml


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def sweep_dir() -> Path:
    return Path(__file__).resolve().parent


def load_temporal_module():
    module_path = project_dir() / "analysis/temporal_bottleneck/run_temporal_bottleneck.py"
    spec = importlib.util.spec_from_file_location("temporal_bottleneck_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_path(latent_channels: int) -> Path:
    return sweep_dir() / "configs" / f"latent_{latent_channels}.yaml"


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def parse_summary(path: Path) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    if not path.exists():
        return row
    with path.open("r") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw = line.strip().split(":", 1)
            value = raw.strip()
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
    return row


def best_history_row(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def to_float(row: dict, key: str) -> float:
        try:
            return float(row.get(key, "nan"))
        except ValueError:
            return float("nan")

    best = min(rows, key=lambda row: to_float(row, "val_loss"))
    out: dict[str, float | str] = {}
    for key, value in best.items():
        try:
            out[f"best_{key}"] = float(value)
        except (TypeError, ValueError):
            out[f"best_{key}"] = value or ""
    return out


def last_history_row(path: Path) -> dict[str, float | str]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    out: dict[str, float | str] = {}
    for key, value in rows[-1].items():
        try:
            out[f"last_{key}"] = float(value)
        except (TypeError, ValueError):
            out[f"last_{key}"] = value or ""
    return out


def run_one(latent_channels: int, *, train: bool, evaluate: bool) -> None:
    module = load_temporal_module()
    path = config_path(latent_channels)
    config = load_config(path)
    print(f"latent_channels={latent_channels} config={path}", flush=True)
    if train:
        module.train(config)
    if evaluate:
        module.evaluate(config)


def summarize(latents: list[int], output: Path) -> None:
    rows = []
    for k in latents:
        cfg = load_config(config_path(k))
        summary_path = project_dir() / cfg["output"]["summary"]
        history_path = project_dir() / cfg["train"]["history_csv"]
        row = {"latent_channels": k}
        row.update(best_history_row(history_path))
        row.update(last_history_row(history_path))
        row.update(parse_summary(summary_path))
        rows.append(row)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"saved summary: {output}", flush=True)
    for row in rows:
        print(
            "latent={latent_channels} best_val_r2={best_val_r2} test_r2={test_r2}".format(
                latent_channels=row.get("latent_channels", ""),
                best_val_r2=row.get("best_val_r2", ""),
                test_r2=row.get("test_r2", ""),
            ),
            flush=True,
        )


def parse_latents(raw: str) -> list[int]:
    if raw == "all":
        return list(range(1, 9))
    latents = [int(part.strip()) for part in raw.split(",") if part.strip()]
    bad = [k for k in latents if k < 1 or k > 8]
    if bad:
        raise ValueError(f"Latent channels must be between 1 and 8, got {bad}")
    return latents


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a temporal bottleneck sweep over latent feature counts 1..8.")
    parser.add_argument("command", choices=["train", "evaluate", "train-evaluate", "summarize"])
    parser.add_argument("--latents", default="all", help="all, or comma-separated values like 1,3,6,8")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("analysis/temporal_bottleneck_feature_sweep/outputs/sweep_summary.csv"),
    )
    args = parser.parse_args()
    latents = parse_latents(args.latents)

    if args.command == "summarize":
        summarize(latents, project_dir() / args.summary_output)
        return

    for k in latents:
        run_one(k, train=args.command in {"train", "train-evaluate"}, evaluate=args.command in {"evaluate", "train-evaluate"})


if __name__ == "__main__":
    main()
