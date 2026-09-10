#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MODEL_ANALYSIS_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.model_analysis.run_model_analysis import (  # noqa: E402
    append_csv,
    build_loss,
    build_model,
    evaluate,
    evaluate_metrics,
    get_device,
    load_config,
    make_loaders,
    project_dir,
    resolve,
    seed_everything,
)


DEFAULT_LRS = [5.6e-4, 1.0e-3, 1.8e-3, 3.2e-3]
DEFAULT_CLIP_NORMS = [0.0, 1.0, 5.0]


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.1e}".replace(".", "p").replace("-", "m").replace("+", "")


def experiment_name(lr: float, clip_norm: float, scheduler: str) -> str:
    clip_label = "none" if clip_norm <= 0.0 else format_float(clip_norm)
    return f"lr_{format_float(lr)}__clip_{clip_label}__sched_{scheduler}"


def prepare_config(base_config: dict, lr: float, clip_norm: float, scheduler: str) -> dict:
    cfg = copy.deepcopy(base_config)
    name = experiment_name(lr, clip_norm, scheduler)
    out_dir = f"analysis/model_analysis/lr_clip_tuning/outputs/{name}"
    cfg["train"]["lr"] = float(lr)
    cfg["train"]["gradient_clip_norm"] = float(clip_norm)
    cfg["train"]["lr_scheduler"] = scheduler
    cfg["train"]["checkpoint"] = f"{out_dir}/temporal_bottleneck.pt"
    cfg["train"]["resume_checkpoint"] = f"{out_dir}/training_state.pt"
    cfg["train"]["history_csv"] = f"{out_dir}/history.csv"
    cfg["output"]["dir"] = out_dir
    cfg["output"]["predictions"] = f"{out_dir}/test_predictions_denorm.npz"
    cfg["output"]["r2_series"] = f"{out_dir}/test_r2_series.csv"
    cfg["output"]["summary"] = f"{out_dir}/evaluation_summary.txt"
    cfg["output"]["stats"] = f"{out_dir}/norm_stats.npz"
    return cfg


def train_epoch_with_clip(model, loader, optimizer, loss_fn, device, *, smooth_weight: float, corr_weight: float, clip_norm: float):
    from analysis.model_analysis.run_model_analysis import latent_regularization

    model.train()
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        features = model.encode_features(x)
        pred = model.decode_features(features)
        pred_loss = loss_fn(pred, y)
        reg_loss = latent_regularization(features, smooth_weight=smooth_weight, corr_weight=corr_weight)
        loss = pred_loss + reg_loss
        loss.backward()
        grad_norm = torch.tensor(float("nan"), dtype=torch.float64, device=device)
        if clip_norm > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm).detach().double()
        optimizer.step()
        batch = x.shape[0]
        totals[0] += loss.detach().double() * batch
        totals[1] += pred_loss.detach().double() * batch
        totals[2] += torch.mean((pred - y) ** 2).detach().double() * batch
        totals[3] += torch.mean(torch.abs(pred - y)).detach().double() * batch
        totals[4] += batch
        if clip_norm > 0.0 and torch.isfinite(grad_norm):
            totals[5] += grad_norm * batch
    denom = totals[4].clamp_min(1.0)
    avg_grad_norm = float((totals[5] / denom).item()) if clip_norm > 0.0 else float("nan")
    return {
        "loss": float((totals[0] / denom).item()),
        "pred_loss": float((totals[1] / denom).item()),
        "mse": float((totals[2] / denom).item()),
        "mae": float((totals[3] / denom).item()),
        "grad_norm": avg_grad_norm,
    }


def build_scheduler(name: str, optimizer):
    if name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1.0e-5,
        )
    raise ValueError(f"Unknown scheduler: {name}")


def load_resume_state(path: Path, model, optimizer, scheduler, device) -> tuple[int, float, int]:
    if not path.exists():
        return 1, float("inf"), 0
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = int(state["epoch"]) + 1
    best_val = float(state.get("best_val", float("inf")))
    stale = int(state.get("stale", 0))
    print(f"Resuming from epoch {start_epoch}; best_val={best_val:.6g}; stale={stale}", flush=True)
    return start_epoch, best_val, stale


def save_resume_state(path: Path, model, optimizer, scheduler, epoch: int, best_val: float, stale: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
            "stale": stale,
        },
        path,
    )


def train_with_clip(config: dict, *, resume: bool = False) -> None:
    root = project_dir()
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    train_loader, val_loader, _, _ = make_loaders(config, recompute_stats=not resume)
    model = build_model(config).to(device)
    train_cfg = config["train"]
    loss_fn = build_loss(str(train_cfg.get("loss", "huber")))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler_name = str(train_cfg.get("lr_scheduler", "none")).lower()
    scheduler = build_scheduler(scheduler_name, optimizer)
    checkpoint = resolve(root, train_cfg["checkpoint"])
    resume_checkpoint = resolve(root, train_cfg.get("resume_checkpoint", checkpoint.with_name("training_state.pt")))
    history_csv = resolve(root, train_cfg["history_csv"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0
    patience = int(train_cfg.get("patience", 20))
    smooth_weight = float(train_cfg.get("latent_smooth_weight", 0.0))
    corr_weight = float(train_cfg.get("latent_corr_weight", 0.0))
    clip_norm = float(train_cfg.get("gradient_clip_norm", 0.0))
    start_epoch = 1
    if resume:
        start_epoch, best_val, stale = load_resume_state(resume_checkpoint, model, optimizer, scheduler, device)
    for epoch in range(start_epoch, int(train_cfg.get("epochs", 250)) + 1):
        train_stats = train_epoch_with_clip(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            smooth_weight=smooth_weight,
            corr_weight=corr_weight,
            clip_norm=clip_norm,
        )
        train_eval = evaluate_metrics(model, train_loader, loss_fn, device)
        val_stats = evaluate_metrics(model, val_loader, loss_fn, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "gradient_clip_norm": clip_norm,
            "lr_scheduler": scheduler_name,
            "train_loss": train_stats["loss"],
            "train_pred_loss": train_stats["pred_loss"],
            "train_grad_norm_before_clip": train_stats["grad_norm"],
            "train_mse": train_eval["mse"],
            "train_mae": train_eval["mae"],
            "train_r2": train_eval["r2"],
            "val_loss": val_stats["loss"],
            "val_mse": val_stats["mse"],
            "val_mae": val_stats["mae"],
            "val_r2": val_stats["r2"],
        }
        append_csv(history_csv, row)
        print(
            f"epoch={epoch} lr={current_lr:.3g} clip={clip_norm:g} "
            f"train_r2={train_eval['r2']:.4f} val_r2={val_stats['r2']:.4f} "
            f"train_loss={train_stats['loss']:.6g} val_loss={val_stats['loss']:.6g}",
            flush=True,
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_val": best_val, "config": config}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best_val={best_val:.6g}", flush=True)
                save_resume_state(resume_checkpoint, model, optimizer, scheduler, epoch, best_val, stale)
                break
        if scheduler is not None:
            scheduler.step(val_stats["loss"])
        save_resume_state(resume_checkpoint, model, optimizer, scheduler, epoch, best_val, stale)


def read_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return out


def summarize(base_config: dict, lrs: list[float], clip_norms: list[float], scheduler: str) -> None:
    root = project_dir()
    rows = []
    for lr, clip_norm in itertools.product(lrs, clip_norms):
        cfg = prepare_config(base_config, lr, clip_norm, scheduler)
        summary_path = resolve(root, cfg["output"]["summary"])
        summary = read_summary(summary_path)
        history_path = resolve(root, cfg["train"]["history_csv"])
        best_epoch = np.nan
        best_val_loss = np.nan
        best_val_r2 = np.nan
        if history_path.exists():
            with history_path.open("r", newline="") as handle:
                hist_rows = list(csv.DictReader(handle))
            if hist_rows:
                best = min(hist_rows, key=lambda row: float(row["val_loss"]))
                best_epoch = int(best["epoch"])
                best_val_loss = float(best["val_loss"])
                best_val_r2 = float(best["val_r2"])
        rows.append({
            "experiment": experiment_name(lr, clip_norm, scheduler),
            "lr": lr,
            "gradient_clip_norm": clip_norm,
            "lr_scheduler": scheduler,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_r2_history": best_val_r2,
            "val_r2": summary.get("val_r2", np.nan),
            "test_r2": summary.get("test_r2", np.nan),
            "test_mse": summary.get("test_mse", np.nan),
            "test_mae": summary.get("test_mae", np.nan),
        })
    out_path = MODEL_ANALYSIS_DIR / "lr_clip_tuning" / "lr_clip_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = [row for row in rows if np.isfinite(row["test_r2"])]
    complete.sort(key=lambda row: row["test_r2"], reverse=True)
    print(f"Wrote {out_path}")
    if complete:
        print("Best completed runs by test_r2:")
        for row in complete[:10]:
            print(
                f"{row['experiment']}: val_r2={row['val_r2']:.4f} "
                f"test_r2={row['test_r2']:.4f} best_epoch={row['best_epoch']}"
            )


def parse_floats(values: list[str] | None, defaults: list[float]) -> list[float]:
    if not values:
        return defaults
    return [float(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune learning rate with optional gradient clipping for SubmesoFluxNet.")
    parser.add_argument("command", choices=["train", "evaluate", "train-evaluate", "resume", "resume-evaluate", "summarize"])
    parser.add_argument("--base-config", default=str(MODEL_ANALYSIS_DIR / "config.yaml"))
    parser.add_argument("--lrs", nargs="*", default=None, help="Learning rates to test.")
    parser.add_argument("--clip-norms", nargs="*", default=None, help="Gradient clip norms. Use 0 for no clipping.")
    parser.add_argument("--scheduler", choices=["plateau", "none"], default="plateau")
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Also rerun configs that already have evaluation_summary.txt.",
    )
    args = parser.parse_args()

    base_config = load_config(Path(args.base_config))
    lrs = parse_floats(args.lrs, DEFAULT_LRS)
    clip_norms = parse_floats(args.clip_norms, DEFAULT_CLIP_NORMS)

    if args.command == "summarize":
        summarize(base_config, lrs, clip_norms, args.scheduler)
        return

    for lr, clip_norm in itertools.product(lrs, clip_norms):
        cfg = prepare_config(base_config, lr, clip_norm, args.scheduler)
        name = experiment_name(lr, clip_norm, args.scheduler)
        summary_path = resolve(project_dir(), cfg["output"]["summary"])
        checkpoint_path = resolve(project_dir(), cfg["train"]["checkpoint"])
        resume_path = resolve(project_dir(), cfg["train"].get("resume_checkpoint", checkpoint_path.with_name("training_state.pt")))
        print(f"\n=== {name} ===", flush=True)
        if args.command in {"resume", "resume-evaluate"} and summary_path.exists() and not args.rerun_completed:
            print(f"Skipping completed config: {summary_path}", flush=True)
            continue
        if args.command in {"train", "train-evaluate", "resume", "resume-evaluate"}:
            can_resume = args.command in {"resume", "resume-evaluate"} and resume_path.exists()
            if args.command in {"resume", "resume-evaluate"} and not can_resume:
                print("No training_state.pt found; starting this unfinished config from epoch 1.", flush=True)
            train_with_clip(cfg, resume=can_resume)
        if args.command in {"evaluate", "train-evaluate", "resume-evaluate"}:
            if checkpoint_path.exists():
                evaluate(cfg)
            else:
                print(f"Skipping evaluation because checkpoint is missing: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
