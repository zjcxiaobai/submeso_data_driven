#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = PROJECT_DIR / "configs" / "unet_no_shuffle_upper_40x20.yaml"
DEFAULT_CONFIG_PREFIX = "unet_no_shuffle_upper_40x20_base_channels"
DEFAULT_OUTPUT_ROOT = Path("outputs_train_on_small_domain/upper_40x20_base_channel")
DEFAULT_SUMMARY = PROJECT_DIR / "hyperparameter_tunning" / "base_channel" / "base_channel_grid_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search UNet_3Layers model.base_channels for the no-shuffle upper 40x20 config."
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Base YAML config to copy and modify.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        nargs="+",
        default=[16, 32, 64, 96, 128],
        help="Candidate model.base_channels values.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for faster pilot searches.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Optional train.early_stopping_patience override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root, relative to _src unless absolute.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="CSV summary of the best validation epoch for each run.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch train.py.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to train.py for each generated config.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training when the run already has a best checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write configs and print training commands.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def as_project_relative(path: Path) -> str:
    path = path.expanduser()
    if path.is_absolute():
        return str(path)
    return str(path)


def run_dir_for(output_root: Path, base_channels: int) -> Path:
    return output_root / f"base_channels_{base_channels:03d}"


def make_config(
    template: dict[str, Any],
    *,
    base_channels: int,
    output_root: Path,
    epochs: int | None,
    early_stopping_patience: int | None,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    run_dir = run_dir_for(output_root, base_channels)
    checkpoint_dir = run_dir / "checkpoints"

    config.setdefault("model", {})["base_channels"] = int(base_channels)
    if epochs is not None:
        config.setdefault("train", {})["epochs"] = int(epochs)
    if early_stopping_patience is not None:
        config.setdefault("train", {})["early_stopping_patience"] = int(early_stopping_patience)

    config.setdefault("data", {})["stats_path"] = as_project_relative(run_dir / "norm_stats.npz")
    train_cfg = config.setdefault("train", {})
    train_cfg["checkpoint_dir"] = as_project_relative(checkpoint_dir)
    train_cfg["best_checkpoint"] = as_project_relative(checkpoint_dir / "best.pt")
    train_cfg["last_checkpoint"] = as_project_relative(checkpoint_dir / "last.pt")
    train_cfg["history_csv"] = as_project_relative(run_dir / "history.csv")
    train_cfg["config_snapshot"] = as_project_relative(run_dir / "config.yaml")
    return config


def command_for(config_path: Path, python_exe: str, resume: bool) -> list[str]:
    command = [python_exe, "train.py", "--config", str(config_path)]
    if resume:
        command.append("--resume")
    return command


def best_history_row(history_csv: Path) -> dict[str, str]:
    if not history_csv.exists():
        return {}
    with history_csv.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def val_loss(row: dict[str, str]) -> float:
        for key in ("val_loss", "val_mse", "loss"):
            value = row.get(key)
            if value not in (None, ""):
                return float(value)
        return float("inf")

    return min(rows, key=val_loss)


def write_summary(summary_csv: Path, rows: list[dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_channels",
        "status",
        "config",
        "run_dir",
        "best_checkpoint",
        "best_epoch",
        "best_val_loss",
        "best_val_mse",
        "best_val_mae",
        "best_val_r2",
        "history_csv",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    template_path = args.template_config.expanduser()
    if not template_path.is_absolute():
        template_path = PROJECT_DIR / template_path

    output_root = args.output_root.expanduser()
    if output_root.is_absolute():
        output_root_for_config = output_root
    else:
        output_root_for_config = output_root

    template = load_yaml(template_path)
    summary_rows: list[dict[str, Any]] = []

    for base_channels in args.base_channels:
        config = make_config(
            template,
            base_channels=base_channels,
            output_root=output_root_for_config,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
        )
        config_path = PROJECT_DIR / "configs" / f"{DEFAULT_CONFIG_PREFIX}_{base_channels:03d}.yaml"
        write_yaml(config_path, config)

        run_dir = run_dir_for(output_root_for_config, base_channels)
        best_checkpoint = PROJECT_DIR / config["train"]["best_checkpoint"]
        history_csv = PROJECT_DIR / config["train"]["history_csv"]
        command = command_for(config_path, args.python, args.resume)
        print(" ".join(command), flush=True)

        status = "dry_run"
        if args.skip_existing and best_checkpoint.exists():
            status = "skipped_existing"
        elif not args.dry_run:
            result = subprocess.run(command, cwd=PROJECT_DIR, check=False)
            status = "ok" if result.returncode == 0 else f"failed_{result.returncode}"
            if result.returncode != 0:
                write_summary(args.summary_csv, summary_rows)
                raise SystemExit(result.returncode)

        best_row = best_history_row(history_csv)
        summary_rows.append(
            {
                "base_channels": base_channels,
                "status": status,
                "config": config_path,
                "run_dir": PROJECT_DIR / run_dir if not run_dir.is_absolute() else run_dir,
                "best_checkpoint": best_checkpoint,
                "best_epoch": best_row.get("epoch", ""),
                "best_val_loss": best_row.get("val_loss", ""),
                "best_val_mse": best_row.get("val_mse", ""),
                "best_val_mae": best_row.get("val_mae", ""),
                "best_val_r2": best_row.get("val_r2", ""),
                "history_csv": history_csv,
            }
        )
        write_summary(args.summary_csv, summary_rows)

    print(f"Wrote summary: {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
