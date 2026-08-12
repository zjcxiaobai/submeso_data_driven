#!/usr/bin/env python3
"""Random-search LfMin using a fixed tuned C_e on the training split.

LfMin changes the FK physics, so each candidate needs its own C_e=1 basis.
This script is resumable: rerun the same command and it continues the current
LfMin basis before scoring and moving to the next candidate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tune_ce_random_search import (
    ANALYSIS_DIR,
    FK_TUNING_DIR,
    PROJECT_DIR,
    align_train_samples,
    build_ce1_basis,
    load_config,
    load_fk_input_times,
    load_truth,
    overall_r2,
    per_snapshot_r2,
    submesofluxnet_train_indices,
)

OUT_DIR = ANALYSIS_DIR / "LfMin_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-e", type=float, default=2.25895966, help="Fixed tuned C_e to use while searching LfMin.")
    parser.add_argument("--lfmin-min", type=float, default=100.0)
    parser.add_argument("--lfmin-max", type=float, default=5000.0)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=("log", "linear"), default="log")
    parser.add_argument("--basis-batch-size", type=int, default=50, help="Compute at most this many missing snapshots per run for the active LfMin.")
    parser.add_argument("--max-train-snapshots", type=int, default=None)
    parser.add_argument("--config", type=Path, default=ANALYSIS_DIR / "config_mld_rho_split.yaml")
    parser.add_argument("--fk-inputs", type=Path, default=FK_TUNING_DIR / "fk_input_trimmed.npz")
    parser.add_argument("--truth", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/y.npy")
    parser.add_argument("--time-source", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/time_index.npy")
    parser.add_argument("--grid-run-dir", type=Path, default=Path("/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"))
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument("--crop", type=int, nargs=4, default=(92, 112, 12, 52), metavar=("J0", "J1", "I0", "I1"))
    parser.add_argument("--submeso-lmax", type=float, default=2751.0)
    parser.add_argument("--submeso-inv-tau", type=float, default=2.0e-6)
    parser.add_argument("--trials-csv", type=Path, default=OUT_DIR / "lfmin_random_search_trials.csv")
    parser.add_argument("--history-csv", type=Path, default=OUT_DIR / "lfmin_random_search_history.csv")
    parser.add_argument("--summary-output", type=Path, default=OUT_DIR / "lfmin_random_search_summary.txt")
    parser.add_argument("--plot-output", type=Path, default=OUT_DIR / "lfmin_random_search_r2.png")
    return parser.parse_args()


def lfmin_tag(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1.0e-6:
        return str(rounded)
    return f"{value:.3f}".replace(".", "p")


def make_trials(args: argparse.Namespace) -> np.ndarray:
    if args.trials_csv.exists():
        values = []
        with args.trials_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values.append(float(row["LfMin"]))
        return np.asarray(values, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    if args.sampling == "log":
        raw = np.exp(rng.uniform(np.log(args.lfmin_min), np.log(args.lfmin_max), size=args.n_trials))
    else:
        raw = rng.uniform(args.lfmin_min, args.lfmin_max, size=args.n_trials)

    values = np.unique(np.rint(np.concatenate([raw, np.asarray([1000.0])])).astype(np.int64)).astype(np.float64)
    args.trials_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.trials_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "LfMin"])
        writer.writeheader()
        for trial, value in enumerate(values):
            writer.writerow({"trial": trial, "LfMin": float(value)})
    return values


def load_history(path: Path) -> dict[float, dict[str, float | int]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lfmin = float(row["LfMin"])
            rows[lfmin] = {
                "LfMin": lfmin,
                "C_e": float(row["C_e"]),
                "n_snapshots": int(row["n_snapshots"]),
                "mean_per_snapshot_r2": float(row["mean_per_snapshot_r2"]),
                "median_per_snapshot_r2": float(row["median_per_snapshot_r2"]),
                "overall_r2": float(row["overall_r2"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
            }
    return rows


def write_history(path: Path, rows_by_lfmin: dict[float, dict[str, float | int]]) -> list[dict[str, float | int]]:
    rows = list(rows_by_lfmin.values())
    rows.sort(key=lambda row: float(row["mean_per_snapshot_r2"]), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "LfMin", "C_e", "n_snapshots", "mean_per_snapshot_r2", "median_per_snapshot_r2", "overall_r2", "mae", "rmse"],
        )
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)
    return rows


def write_summary(path: Path, args: argparse.Namespace, sample_index: np.ndarray, time_index: np.ndarray, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("FK LfMin random search with fixed tuned C_e\n")
        handle.write(f"C_e: {args.c_e:.10g}\n")
        handle.write(f"LfMin range: {args.lfmin_min:.10g}-{args.lfmin_max:.10g}\n")
        handle.write(f"sampling: {args.sampling}\n")
        handle.write(f"n_trials_requested: {args.n_trials}\n")
        handle.write(f"n_completed_trials: {len(rows)}\n")
        handle.write(f"n_aligned_train_snapshots: {sample_index.size}\n")
        handle.write(f"sample_index_range: {int(sample_index.min())}-{int(sample_index.max())}\n")
        handle.write(f"time_index_range: {int(time_index.min())}-{int(time_index.max())}\n")
        if rows:
            best = rows[0]
            handle.write("\nBest by mean per-snapshot R2\n")
            for key in ("LfMin", "C_e", "n_snapshots", "mean_per_snapshot_r2", "median_per_snapshot_r2", "overall_r2", "mae", "rmse"):
                value = best[key]
                if isinstance(value, int):
                    handle.write(f"{key}: {value}\n")
                else:
                    handle.write(f"{key}: {float(value):.10g}\n")


def plot_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    mpl.rcParams.update({"font.size": 7, "axes.spines.right": False, "axes.spines.top": False})
    lfmin = np.asarray([float(row["LfMin"]) for row in rows])
    r2 = np.asarray([float(row["mean_per_snapshot_r2"]) for row in rows])
    best = rows[0]
    fig, ax = plt.subplots(figsize=(3.2, 2.3), constrained_layout=True)
    ax.scatter(lfmin, r2, s=14, color="#4A6F8A", alpha=0.75, linewidths=0)
    ax.scatter([float(best["LfMin"])], [float(best["mean_per_snapshot_r2"])], s=70, marker="*", color="#B64A3B", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$L_{f,min}$ (m)")
    ax.set_ylabel(r"Mean training $R^2$")
    ax.set_title(r"FK $L_{f,min}$ random search", fontsize=7.5, pad=4)
    ax.grid(color="#D6D6D6", linewidth=0.4, alpha=0.55)
    ax.text(
        0.03,
        0.96,
        fr"best $L_{{f,min}}$={float(best['LfMin']):.0f} m\nmean $R^2$={float(best['mean_per_snapshot_r2']):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def basis_args(args: argparse.Namespace, lfmin: float, basis_output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        fk_inputs=args.fk_inputs,
        grid_run_dir=args.grid_run_dir,
        box_j=args.box_j,
        box_i=args.box_i,
        crop=args.crop,
        lf_min=float(lfmin),
        submeso_lmax=args.submeso_lmax,
        submeso_inv_tau=args.submeso_inv_tau,
        basis_output=basis_output,
        basis_batch_size=args.basis_batch_size,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_sample_all = submesofluxnet_train_indices(config, PROJECT_DIR)
    truth_time, truth_data = load_truth(args.truth, args.time_source)
    fk_input_time = load_fk_input_times(args.fk_inputs)
    sample_index, time_index = align_train_samples(train_sample_all, truth_time, fk_input_time, args.max_train_snapshots)
    truth_lookup = {int(t): n for n, t in enumerate(truth_time)}

    trials = make_trials(args)
    rows_by_lfmin = load_history(args.history_csv)
    completed = set(rows_by_lfmin)
    print(f"Aligned train snapshots: {sample_index.size}", flush=True)
    print(f"LfMin trials: {trials.size}; completed: {len(completed)}", flush=True)

    for lfmin in trials:
        lfmin = float(lfmin)
        if lfmin in completed:
            continue
        tag = lfmin_tag(lfmin)
        basis_output = OUT_DIR / f"fk_train_lfmin{tag}_ce1_basis.npz"
        print(f"\nLfMin={lfmin:.0f} m, basis={basis_output}", flush=True)
        basis_sample, basis_time, basis_flux = build_ce1_basis(basis_args(args, lfmin, basis_output), sample_index, time_index)
        if basis_time.size < sample_index.size:
            print(
                f"Basis for LfMin={lfmin:.0f} is incomplete: {basis_time.size}/{sample_index.size}. "
                "Rerun the same command to continue before scoring this LfMin.",
                flush=True,
            )
            break

        truth = np.stack([truth_data[truth_lookup[int(t)]] for t in basis_time], axis=0).astype(np.float64)
        pred = args.c_e * basis_flux.astype(np.float64)
        r2_series = per_snapshot_r2(pred, truth)
        rows_by_lfmin[lfmin] = {
            "LfMin": lfmin,
            "C_e": float(args.c_e),
            "n_snapshots": int(basis_time.size),
            "mean_per_snapshot_r2": float(np.nanmean(r2_series)),
            "median_per_snapshot_r2": float(np.nanmedian(r2_series)),
            "overall_r2": float(overall_r2(pred, truth)),
            "mae": float(np.nanmean(np.abs(pred - truth))),
            "rmse": float(np.sqrt(np.nanmean((pred - truth) ** 2))),
        }
        rows = write_history(args.history_csv, rows_by_lfmin)
        write_summary(args.summary_output, args, sample_index, time_index, rows)
        plot_history(args.plot_output, rows)
        best = rows[0]
        print(f"Scored LfMin={lfmin:.0f}; current best LfMin={float(best['LfMin']):.0f}, mean R2={float(best['mean_per_snapshot_r2']):.6f}", flush=True)

    rows = write_history(args.history_csv, rows_by_lfmin) if rows_by_lfmin else []
    write_summary(args.summary_output, args, sample_index, time_index, rows)
    plot_history(args.plot_output, rows)
    print(f"Saved {args.trials_csv}", flush=True)
    print(f"Saved {args.history_csv}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    if rows:
        print(f"Best LfMin so far: {float(rows[0]['LfMin']):.0f} m", flush=True)


if __name__ == "__main__":
    main()
