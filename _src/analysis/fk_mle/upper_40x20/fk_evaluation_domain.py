#!/usr/bin/env python3
"""Compute tuned-Ce FK heat flux for the final model_analysis test split.

The script reads sample indices from the final temporal-bottleneck prediction
file, computes FK on the same held-out raw-time-block samples, saves physical-unit
FK predictions, and writes per-snapshot R2.

Update --c-e after running tune_ce_random_search.py on the matching training set.
The FK calculation is resumable via the full-grid progress NPZ; rerun the same
command after interruption.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


PROJECT_DIR = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_DIR.parent
PER_SNAPSHOT_DIR = PROJECT_DIR / "analysis/per_snapshot_r2_time_series"
OUT_DIR = PROJECT_DIR / "analysis/fk_mle/upper_40x20/final_model_analysis_test"

if str(PER_SNAPSHOT_DIR) not in sys.path:
    sys.path.insert(0, str(PER_SNAPSHOT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fk_physics import compute_fk_submeso_heat_flux_one_time  # noqa: E402
from src.utils import build_coarse_grid, read_mitgcm  # noqa: E402


BEST_CE = 2.25895966  # Best value recorded in Ce_base_case_calculation/ce_random_search_summary.txt.


PALETTE = {
    "line": "#466A54",
    "smooth": "#B64A3B",
    "fill": "#E4ECE6",
    "hist": "#9DB7A5",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-e", type=float, default=BEST_CE)
    parser.add_argument("--predictions", type=Path, default=PROJECT_DIR / "analysis/model_analysis/outputs_final/test_predictions_denorm.npz")
    parser.add_argument("--time-source", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/time_index.npy")
    parser.add_argument(
        "--grid-run-dir",
        type=Path,
        default=Path("/scratch/jczhang/MITgcm_BI/verification/tutorial_reentrant_channel/run"),
    )
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    parser.add_argument("--nc-dir", type=Path, default=Path("/scratch/jczhang/data_5km/save_coarse/13901616_nc"))
    parser.add_argument("--engine", default="h5netcdf", choices=("h5netcdf", "netcdf4", "scipy"))
    parser.add_argument("--crop", type=int, nargs=4, default=(92, 112, 12, 52), metavar=("J0", "J1", "I0", "I1"), help="Crop on the 120x60 FK grid matching upper40x20: 56x56 y=30:50, x=10:50.")
    parser.add_argument("--lf-min", type=float, default=1374.0)
    parser.add_argument("--submeso-lmax", type=float, default=2751.0)
    parser.add_argument("--submeso-inv-tau", type=float, default=2.0e-6)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--max-snapshots", type=int, default=None, help="Compute only the next N missing snapshots.")
    parser.add_argument(
        "--skip-missing-nc",
        action="store_true",
        help="Skip requested test times that are not present in the FK input NetCDF files.",
    )
    parser.add_argument("--calc-only", action="store_true", help="Compute/save FK only; skip R2 plot.")
    parser.add_argument("--plot-only", action="store_true", help="Skip FK calculation and plot from an existing prediction NPZ.")
    parser.add_argument(
        "--allow-partial-output",
        action="store_true",
        help="Save/plot the currently available progress snapshots even if the full test set is incomplete.",
    )
    parser.add_argument("--progress", type=Path, default=OUT_DIR / "fk_progress_fullgrid.npz")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "fk_prediction_physical.npz")
    parser.add_argument("--r2-csv", type=Path, default=OUT_DIR / "fk_r2_series.csv")
    parser.add_argument("--output-prefix", type=Path, default=OUT_DIR / "fk_per_snapshot_r2")
    parser.add_argument("--summary-output", type=Path, default=OUT_DIR / "fk_per_snapshot_r2_summary.txt")
    parser.add_argument("--stats-csv", type=Path, default=OUT_DIR / "fk_per_snapshot_r2_stats.csv")
    parser.add_argument("--rolling-window", type=int, default=81)
    return parser.parse_args()


def load_test_times(predictions_path: Path, time_source_path: Path) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(predictions_path)
    time_source = np.load(time_source_path)
    sample_index = pred["sample_index"].astype(np.int64)
    if isinstance(time_source, np.lib.npyio.NpzFile):
        key = "time_index" if "time_index" in time_source.files else "time"
        all_time = time_source[key].astype(np.int64)
    else:
        all_time = np.asarray(time_source, dtype=np.int64)
    time_index = all_time[sample_index]
    return sample_index, time_index


def load_progress(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.exists():
        return np.array([], dtype=np.int64), None
    z = np.load(path)
    return z["time_index"].astype(np.int64), z["data"].astype(np.float32)


def save_progress(path: Path, time_index: np.ndarray, data: np.ndarray) -> None:
    order = np.argsort(time_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, time_index=time_index[order].astype(np.int64), data=data[order].astype(np.float32))
    os.replace(tmp_path, path)


def append_progress(path: Path, new_time: int, new_map: np.ndarray) -> None:
    old_time, old_data = load_progress(path)
    time_cat = np.concatenate([old_time, np.asarray([new_time], dtype=np.int64)])
    data_new = np.asarray([new_map], dtype=np.float32)
    data_cat = data_new if old_data is None else np.concatenate([old_data, data_new], axis=0)
    reverse = np.arange(time_cat.size - 1, -1, -1)
    _, first = np.unique(time_cat[reverse], return_index=True)
    keep = np.sort(reverse[first])
    save_progress(path, time_cat[keep], data_cat[keep])


def build_grid(grid_run_dir: Path, box_j: int, box_i: int) -> xr.Dataset:
    ds_fine = read_mitgcm.read_mitgrid_staggered(grid_run_dir)
    return build_coarse_grid.build_coarse_grid_staggered(ds_fine, box_j=box_j, box_i=box_i)


def calculate_fk(args: argparse.Namespace, sample_index: np.ndarray, target_time: np.ndarray) -> None:
    ds_grid = build_grid(args.grid_run_dir, args.box_j, args.box_i)
    ds_kpphbl = xr.open_dataset(args.nc_dir / "kpphbl_12hmean_coarse.nc", engine=args.engine, decode_times=False)
    ds_t = xr.open_dataset(args.nc_dir / "T_12hmean_coarse.nc", engine=args.engine, decode_times=False)
    ds_s = xr.open_dataset(args.nc_dir / "S_12hmean_coarse.nc", engine=args.engine, decode_times=False)
    ds_eta = xr.open_dataset(args.nc_dir / "Eta_12hmean_coarse.nc", engine=args.engine, decode_times=False)

    kpphbl_da = ds_kpphbl["kpphbl"]
    t_da = ds_t["T"]
    s_da = ds_s["S"]
    eta_da = ds_eta["Eta"]

    available = set(np.asarray(kpphbl_da["time"].values, dtype=np.int64).tolist())
    missing_from_nc = [int(t) for t in target_time if int(t) not in available]
    if missing_from_nc:
        if not args.skip_missing_nc:
            raise RuntimeError(f"NC files do not contain {len(missing_from_nc)} test times. First missing: {missing_from_nc[:5]}")
        print(
            f"Skipping {len(missing_from_nc)} test times missing from FK input NC files. "
            f"First missing: {missing_from_nc[:5]}",
            flush=True,
        )

    saved_time, _ = load_progress(args.progress)
    saved_set = set(map(int, saved_time))
    todo = [
        (int(s), int(t))
        for s, t in zip(sample_index, target_time)
        if int(t) in available and int(t) not in saved_set
    ]
    if args.max_snapshots is not None:
        todo = todo[: args.max_snapshots]

    print(f"Existing progress snapshots: {len(saved_set)}", flush=True)
    print(f"Need to compute in this run: {len(todo)}", flush=True)
    for n, (_, time) in enumerate(todo, start=1):
        print(f"[{n}/{len(todo)}] FK time_index={time}, C_e={args.c_e}", flush=True)
        fk_map = compute_fk_submeso_heat_flux_one_time(
            time_index=time,
            ds_grid=ds_grid,
            kpphbl_da=kpphbl_da,
            t_da=t_da,
            s_da=s_da,
            eta_da=eta_da,
            c_e=args.c_e,
            submeso_lmax=args.submeso_lmax,
            submeso_inv_tau=args.submeso_inv_tau,
            lf_min=args.lf_min,
            f0=args.f0,
            beta=args.beta,
        )
        append_progress(args.progress, time, fk_map)


def save_physical_prediction(args: argparse.Namespace, sample_index: np.ndarray, target_time: np.ndarray) -> bool:
    saved_time, saved_data = load_progress(args.progress)
    if saved_data is None:
        raise FileNotFoundError(f"No progress data found at {args.progress}")
    lookup = {int(t): n for n, t in enumerate(saved_time)}
    if args.allow_partial_output:
        keep = np.asarray([n for n, t in enumerate(target_time) if int(t) in lookup], dtype=np.int64)
    else:
        missing = [int(t) for t in target_time if int(t) not in lookup]
        if missing:
            print(f"Final physical output not saved yet; missing {len(missing)} snapshots. First missing: {missing[:5]}", flush=True)
            return False
        keep = np.arange(target_time.size, dtype=np.int64)

    if keep.size == 0:
        raise RuntimeError("No progress snapshots overlap the target test set.")

    j0, j1, i0, i1 = args.crop
    full = np.stack([saved_data[lookup[int(target_time[n])]] for n in keep], axis=0)
    cropped = full[:, None, j0:j1, i0:i1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_index=sample_index[keep].astype(np.int64),
        time_index=target_time[keep].astype(np.int64),
        prediction=cropped.astype(np.float32),
        fk_heat_flux=cropped.astype(np.float32),
        units=np.asarray(["W m^-2"]),
        c_e=np.asarray([args.c_e], dtype=np.float64),
        lf_min=np.asarray([args.lf_min], dtype=np.float64),
        submeso_lmax=np.asarray([args.submeso_lmax], dtype=np.float64),
        submeso_inv_tau=np.asarray([args.submeso_inv_tau], dtype=np.float64),
        crop_j=np.asarray([j0, j1], dtype=np.int64),
        crop_i=np.asarray([i0, i1], dtype=np.int64),
    )
    print(f"Saved physical-unit FK prediction: {args.output}", flush=True)
    return True


def load_prediction_and_truth(output_path: Path, predictions_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_fk = np.load(output_path)
    pred_nn = np.load(predictions_path)
    sample_index = pred_fk["sample_index"].astype(np.int64)
    time_index = pred_fk["time_index"].astype(np.int64)
    fk = pred_fk["prediction"].astype(np.float64)
    all_sample = pred_nn["sample_index"].astype(np.int64)
    all_target = pred_nn["target"].astype(np.float64)
    lookup = {int(s): n for n, s in enumerate(all_sample)}
    target = np.stack([all_target[lookup[int(s)]] for s in sample_index], axis=0)
    if fk.shape != target.shape:
        raise ValueError(f"FK prediction and target shapes differ: {fk.shape} != {target.shape}")
    return sample_index, time_index, fk, target


def per_snapshot_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_target = target.reshape(target.shape[0], -1)
    r2 = np.full(pred.shape[0], np.nan, dtype=np.float64)
    mse = np.full(pred.shape[0], np.nan, dtype=np.float64)
    mae = np.full(pred.shape[0], np.nan, dtype=np.float64)
    for n in range(pred.shape[0]):
        valid = np.isfinite(flat_pred[n]) & np.isfinite(flat_target[n])
        if not np.any(valid):
            continue
        p = flat_pred[n, valid]
        y = flat_target[n, valid]
        diff = p - y
        sse = float(np.sum(diff**2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        if sst > 0.0:
            r2[n] = 1.0 - sse / sst
        mse[n] = float(np.mean(diff**2))
        mae[n] = float(np.mean(np.abs(diff)))
    return r2, mse, mae


def overall_r2(pred: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return float("nan")
    p = pred[valid]
    y = target[valid]
    sse = float(np.sum((p - y) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst if sst > 0.0 else float("nan")


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    smooth = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        start = max(0, idx - half)
        stop = min(values.size, idx + half + 1)
        smooth[idx] = np.nanmedian(values[start:stop])
    return smooth


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def write_r2_csv(path: Path, sample_index: np.ndarray, time_index: np.ndarray, r2: np.ndarray, mse: np.ndarray, mae: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["test_position", "sample_index", "time_index", "r2", "mse", "mae"])
        writer.writeheader()
        for n, row in enumerate(zip(sample_index, time_index, r2, mse, mae)):
            sample, time, r2_n, mse_n, mae_n = row
            writer.writerow(
                {
                    "test_position": n,
                    "sample_index": int(sample),
                    "time_index": int(time),
                    "r2": float(r2_n),
                    "mse": float(mse_n),
                    "mae": float(mae_n),
                }
            )


def write_stats(path: Path, summary: dict[str, float | int], overall: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})
        writer.writerow({"metric": "overall_r2", "value": overall})


def write_summary(path: Path, args: argparse.Namespace, sample_index: np.ndarray, time_index: np.ndarray, summary: dict[str, float | int], overall: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Tuned-Ce FK MLD-rho final model_analysis raw-time-block test R2 summary\n")
        handle.write(f"C_e: {args.c_e:.10g}\n")
        handle.write(f"sample_index_range: {int(sample_index.min())}-{int(sample_index.max())}\n")
        handle.write(f"time_index_range: {int(time_index.min())}-{int(time_index.max())}\n")
        for key, value in summary.items():
            if isinstance(value, int):
                handle.write(f"{key}: {value}\n")
            else:
                handle.write(f"{key}: {value:.10g}\n")
        handle.write(f"overall_r2: {overall:.10g}\n")


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_r2(args: argparse.Namespace) -> None:
    sample_index, time_index, fk, target = load_prediction_and_truth(args.output, args.predictions)
    r2, mse, mae = per_snapshot_metrics(fk, target)
    overall = overall_r2(fk, target)
    summary = finite_summary(r2)
    smooth = rolling_median(r2, args.rolling_window)
    write_r2_csv(args.r2_csv, sample_index, time_index, r2, mse, mae)
    write_stats(args.stats_csv, summary, overall)
    write_summary(args.summary_output, args, sample_index, time_index, summary, overall)

    apply_plot_style()
    fig, (ax_ts, ax_hist) = plt.subplots(1, 2, figsize=(5.25, 2.45), width_ratios=(3.4, 1.0), constrained_layout=True)
    mean_r2 = float(summary["mean"])
    p05 = float(summary["p05"])
    p95 = float(summary["p95"])
    ax_ts.fill_between(sample_index, p05, p95, color=PALETTE["fill"], alpha=0.38, linewidth=0)
    ax_ts.plot(sample_index, r2, color=PALETTE["line"], linewidth=0.75, alpha=0.68)
    ax_ts.plot(sample_index, smooth, color=PALETTE["smooth"], linewidth=1.35)
    ax_ts.axhline(mean_r2, color=PALETTE["neutral_dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax_ts.set_xlim(int(sample_index.min()), int(sample_index.max()))
    ax_ts.set_ylim(float(np.nanmin(r2)) - 0.04, min(1.0, float(np.nanmax(r2)) + 0.04))
    ax_ts.set_xlabel("Held-out test data index")
    ax_ts.set_ylabel(r"Per-snapshot $R^2$")
    ax_ts.set_title(fr"FK final test $R^2$, $C_e$={args.c_e:.4g}", fontsize=7.5, pad=4)
    ax_ts.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.4, alpha=0.5)
    ax_ts.text(0.012, 0.96, fr"mean $R^2$={mean_r2:.3f}", transform=ax_ts.transAxes, ha="left", va="top", fontsize=6.5)

    finite_r2 = r2[np.isfinite(r2)]
    bins = np.linspace(float(np.nanmin(finite_r2)) - 0.02, float(np.nanmax(finite_r2)) + 0.02, 18)
    ax_hist.hist(finite_r2, bins=bins, orientation="horizontal", color=PALETTE["hist"], alpha=0.86)
    ax_hist.axhline(mean_r2, color=PALETTE["neutral_dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax_hist.set_ylim(ax_ts.get_ylim())
    ax_hist.set_xlabel("Snapshots")
    ax_hist.set_title("Distribution", fontsize=7.5, pad=4)
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.35, alpha=0.42)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R2 CSV: {args.r2_csv}", flush=True)
    print(f"Saved R2 figure: {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"mean R2: {summary['mean']:.6f}", flush=True)
    print(f"overall R2: {overall:.6f}", flush=True)


def main() -> None:
    args = parse_args()
    sample_index, target_time = load_test_times(args.predictions, args.time_source)
    print(f"Test snapshots: {sample_index.size}", flush=True)
    print(f"Test time range: {int(target_time.min())} ... {int(target_time.max())}", flush=True)
    print(f"Using tuned C_e={args.c_e}", flush=True)
    print(f"Using tuned lf_min={args.lf_min}", flush=True)

    if args.calc_only and args.plot_only:
        raise ValueError("Use only one of --calc-only or --plot-only, not both.")

    if not args.plot_only:
        calculate_fk(args, sample_index, target_time)
        output_ready = save_physical_prediction(args, sample_index, target_time)
    else:
        output_ready = args.output.exists()
        if not output_ready:
            raise FileNotFoundError(f"Missing physical prediction NPZ for --plot-only: {args.output}")

    if args.calc_only:
        return
    if output_ready:
        plot_r2(args)
    else:
        print("R2 plot skipped because full physical prediction output is not complete yet.", flush=True)
        print("Use --allow-partial-output if you want a partial test-set plot.", flush=True)


if __name__ == "__main__":
    main()
