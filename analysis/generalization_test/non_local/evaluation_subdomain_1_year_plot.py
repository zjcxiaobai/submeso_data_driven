#!/usr/bin/env python3
"""Find and plot the best one-year lower-domain evaluation window."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent

PALETTE = {
    "line": "#245E8A",
    "smooth": "#B55345",
    "fk": "#4F8A5B",
    "mean": "#2F2F2F",
    "zero": "#8A8A8A",
    "neutral_dark": "#2F2F2F",
    "neutral_mid": "#7A7A7A",
    "neutral_light": "#E3E3E3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r2-series",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_model_analysis_r2_series.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "evaluation_subdomain_best_1_year_window",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=HERE / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz",
        help="Evaluation NPZ containing lower-domain target maps.",
    )
    parser.add_argument(
        "--fk-predictions",
        type=Path,
        default=HERE / "evaluation_subdomain_fk_prediction.npz",
        help="Assembled FK/MLE prediction NPZ. Used first if it exists.",
    )
    parser.add_argument(
        "--fk-progress",
        type=Path,
        default=HERE / "evaluation_subdomain_fk_progress_fullgrid.npz",
        help="Full-grid FK/MLE progress NPZ used when --fk-predictions is not available.",
    )
    parser.add_argument(
        "--fk-crop",
        type=int,
        nargs=4,
        default=(72, 92, 12, 52),
        metavar=("J0", "J1", "I0", "I1"),
        help="Crop applied to full-grid FK maps before computing lower-domain R2.",
    )
    parser.add_argument(
        "--upper-truth",
        type=Path,
        default=HERE.parents[2] / "analysis/model_analysis/data/y.npy",
        help="Upper training-subdomain truth array with the same sample_index convention.",
    )
    parser.add_argument("--window-snapshots", type=int, default=None, help="Window length in snapshots. Overrides --window-days.")
    parser.add_argument("--window-days", type=float, default=365.0, help="Window length in model days.")
    parser.add_argument("--snapshots-per-day", type=float, default=2.0, help="12-hour data use 2 snapshots per day.")
    parser.add_argument("--smooth-window", type=int, default=41, help="Rolling median window for display.")
    parser.add_argument("--snapshot-choice", choices=("best", "worst", "center", "random-middle"), default="random-middle", help="Snapshot from the selected window shown in the domain map.")
    parser.add_argument("--snapshot-seed", type=int, default=42, help="Seed used when --snapshot-choice random-middle.")
    parser.add_argument("--middle-quantile-low", type=float, default=0.4, help="Lower quantile for random-middle snapshot pool.")
    parser.add_argument("--middle-quantile-high", type=float, default=0.6, help="Upper quantile for random-middle snapshot pool.")
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_r2_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing R2 CSV: {path}")
    test_position: list[int] = []
    sample_index: list[int] = []
    time_index: list[int] = []
    has_time = False
    r2: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"test_position", "sample_index", "r2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        has_time = "time_index" in (reader.fieldnames or [])
        for row in reader:
            test_position.append(int(row["test_position"]))
            sample_index.append(int(row["sample_index"]))
            if has_time:
                time_index.append(int(row["time_index"]) if row["time_index"] != "" else -1)
            r2.append(float(row["r2"]))
    time_array = np.asarray(time_index, dtype=np.int64) if has_time else None
    return (
        np.asarray(test_position, dtype=np.int64),
        np.asarray(sample_index, dtype=np.int64),
        time_array,
        np.asarray(r2, dtype=np.float64),
    )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or values.size < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        lo = max(0, idx - half)
        hi = min(values.size, idx + half + 1)
        out[idx] = np.nanmedian(values[lo:hi])
    return out


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = prediction - target
    reduce_axes = tuple(range(1, target.ndim))
    sse = np.sum(residual**2, axis=reduce_axes)
    target_mean = np.mean(target, axis=reduce_axes, keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=reduce_axes)
    r2 = np.full(sse.shape, np.nan, dtype=np.float64)
    np.divide(sse, sst, out=r2, where=sst > 0.0)
    r2[sst > 0.0] = 1.0 - r2[sst > 0.0]
    return r2


def load_fk_window_r2(
    predictions_path: Path,
    fk_predictions_path: Path,
    fk_progress_path: Path,
    fk_crop: tuple[int, int, int, int],
    window_time_index: np.ndarray,
) -> np.ndarray:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing model prediction file with targets: {predictions_path}")
    with np.load(predictions_path) as data:
        required = {"time_index", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{predictions_path} is missing arrays: {sorted(missing)}")
        target_time = np.asarray(data["time_index"], dtype=np.int64)
        target = np.asarray(data["target"], dtype=np.float32)
    target_lookup = {int(t): n for n, t in enumerate(target_time)}

    if fk_predictions_path.exists():
        with np.load(fk_predictions_path) as data:
            if "time_index" not in data.files:
                raise ValueError(f"{fk_predictions_path} is missing time_index")
            pred_key = "prediction" if "prediction" in data.files else "fk_heat_flux"
            if pred_key not in data.files:
                raise ValueError(f"{fk_predictions_path} is missing prediction/fk_heat_flux")
            fk_time = np.asarray(data["time_index"], dtype=np.int64)
            fk_pred = np.asarray(data[pred_key], dtype=np.float32)
    else:
        if not fk_progress_path.exists():
            raise FileNotFoundError(f"Missing FK prediction and progress files: {fk_predictions_path}, {fk_progress_path}")
        with np.load(fk_progress_path) as data:
            required = {"time_index", "data"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{fk_progress_path} is missing arrays: {sorted(missing)}")
            fk_time = np.asarray(data["time_index"], dtype=np.int64)
            full_grid = np.asarray(data["data"], dtype=np.float32)
        j0, j1, i0, i1 = fk_crop
        fk_pred = full_grid[:, None, j0:j1, i0:i1]

    fk_lookup = {int(t): n for n, t in enumerate(fk_time)}
    missing_fk = [int(t) for t in window_time_index if int(t) not in fk_lookup]
    missing_target = [int(t) for t in window_time_index if int(t) not in target_lookup]
    if missing_fk:
        raise ValueError(f"FK data miss {len(missing_fk)} best-window times. First missing: {missing_fk[:8]}")
    if missing_target:
        raise ValueError(f"Target data miss {len(missing_target)} best-window times. First missing: {missing_target[:8]}")

    pred = np.stack([fk_pred[fk_lookup[int(t)]] for t in window_time_index], axis=0)
    truth = np.stack([target[target_lookup[int(t)]] for t in window_time_index], axis=0)
    if pred.shape != truth.shape:
        raise ValueError(f"FK and target shapes differ after crop: {pred.shape} vs {truth.shape}")
    return sample_r2(pred, truth)


def best_window(r2: np.ndarray, window: int) -> tuple[int, int, float]:
    if window <= 0:
        raise ValueError("--window must be positive")
    if r2.size < window:
        raise ValueError(f"Window {window} exceeds R2 series length {r2.size}")
    best_start = 0
    best_mean = -np.inf
    for start in range(0, r2.size - window + 1):
        stop = start + window
        mean = float(np.nanmean(r2[start:stop]))
        if np.isfinite(mean) and mean > best_mean:
            best_start = start
            best_mean = mean
    return best_start, best_start + window, best_mean


def write_window_csv(
    path: Path,
    test_position: np.ndarray,
    sample_index: np.ndarray,
    time_index: np.ndarray | None,
    r2: np.ndarray,
    start: int,
    stop: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_position", "test_position", "sample_index", "time_index", "r2"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for n, idx in enumerate(range(start, stop)):
            writer.writerow(
                {
                    "window_position": n,
                    "test_position": int(test_position[idx]),
                    "sample_index": int(sample_index[idx]),
                    "time_index": int(time_index[idx]) if time_index is not None and int(time_index[idx]) >= 0 else "",
                    "r2": float(r2[idx]),
                }
            )


def write_summary(
    path: Path,
    r2_path: Path,
    test_position: np.ndarray,
    sample_index: np.ndarray,
    time_index: np.ndarray | None,
    r2: np.ndarray,
    start: int,
    stop: int,
    mean_r2: float,
    fk_r2: np.ndarray | None = None,
    map_snapshot_position: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    window_r2 = r2[start:stop]
    with path.open("w") as handle:
        handle.write("Best one-year lower-domain evaluation window\n")
        handle.write(f"input_r2_series: {r2_path}\n")
        handle.write(f"window_snapshots: {stop - start}\n")
        handle.write(f"start_test_position: {int(test_position[start])}\n")
        handle.write(f"end_test_position: {int(test_position[stop - 1])}\n")
        handle.write(f"start_sample_index: {int(sample_index[start])}\n")
        handle.write(f"end_sample_index: {int(sample_index[stop - 1])}\n")
        if time_index is not None:
            handle.write(f"start_time_index: {int(time_index[start])}\n")
            handle.write(f"end_time_index: {int(time_index[stop - 1])}\n")
        handle.write(f"mean_r2: {mean_r2:.10g}\n")
        handle.write(f"median_r2: {float(np.nanmedian(window_r2)):.10g}\n")
        handle.write(f"min_r2: {float(np.nanmin(window_r2)):.10g}\n")
        handle.write(f"max_r2: {float(np.nanmax(window_r2)):.10g}\n")
        handle.write(f"std_r2: {float(np.nanstd(window_r2)):.10g}\n")
        if fk_r2 is not None:
            handle.write(f"fk_mean_r2: {float(np.nanmean(fk_r2)):.10g}\n")
            handle.write(f"fk_median_r2: {float(np.nanmedian(fk_r2)):.10g}\n")
            handle.write(f"fk_min_r2: {float(np.nanmin(fk_r2)):.10g}\n")
            handle.write(f"fk_max_r2: {float(np.nanmax(fk_r2)):.10g}\n")
            handle.write(f"fk_std_r2: {float(np.nanstd(fk_r2)):.10g}\n")
        if map_snapshot_position is not None:
            handle.write(f"map_snapshot_position: {map_snapshot_position}\n")
            handle.write(f"map_snapshot_sample_index: {int(sample_index[map_snapshot_position])}\n")
            if time_index is not None:
                handle.write(f"map_snapshot_time_index: {int(time_index[map_snapshot_position])}\n")
            handle.write(f"map_snapshot_r2: {float(r2[map_snapshot_position]):.10g}\n")



def load_domain_map(predictions_path: Path, upper_truth_path: Path, sample: int) -> np.ndarray:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {predictions_path}")
    if not upper_truth_path.exists():
        raise FileNotFoundError(f"Missing upper truth file: {upper_truth_path}")
    with np.load(predictions_path) as data:
        required = {"sample_index", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{predictions_path} is missing arrays: {sorted(missing)}")
        lower_sample = np.asarray(data["sample_index"], dtype=np.int64)
        lower_target = np.asarray(data["target"], dtype=np.float32)
    lower_lookup = {int(s): n for n, s in enumerate(lower_sample)}
    if int(sample) not in lower_lookup:
        raise ValueError(f"sample_index {sample} was not found in {predictions_path}")
    lower = lower_target[lower_lookup[int(sample)], 0]

    upper_all = np.load(upper_truth_path, mmap_mode="r")
    if sample < 0 or sample >= upper_all.shape[0]:
        raise ValueError(f"sample_index {sample} is outside upper truth length {upper_all.shape[0]}")
    upper = np.asarray(upper_all[int(sample), 0], dtype=np.float32)
    if upper.shape != lower.shape:
        raise ValueError(f"Upper/lower map shapes differ: {upper.shape} vs {lower.shape}")
    return np.vstack([lower, upper])


def robust_symmetric_limit(values: np.ndarray, percentile: float = 99.0, min_limit: float = 1.0) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return min_limit
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(np.abs(finite)))
    return max(limit, min_limit)

def draw_figure(
    output_prefix: Path,
    sample_index: np.ndarray,
    r2: np.ndarray,
    start: int,
    stop: int,
    mean_r2: float,
    fk_r2: np.ndarray | None,
    smooth_window: int,
    domain_map: np.ndarray,
    best_sample: int,
) -> None:
    apply_style()
    title_fs = 10.5
    axis_label_fs = 10.5
    tick_fs = 7.8
    legend_fs = 8.5
    annotation_fs = 8.5
    cbar_label_fs = 8.5
    x = sample_index[start:stop]
    y = r2[start:stop]
    y_smooth = rolling_median(y, smooth_window)
    fk_smooth = rolling_median(fk_r2, smooth_window) if fk_r2 is not None else None

    # Allocate more horizontal space to the skill time series, which is the
    # primary result, while retaining a compact square domain map.
    fig = plt.figure(figsize=(9.8, 3.35))
    ax = fig.add_axes([0.065, 0.16, 0.59, 0.72])
    ax_map = fig.add_axes([0.735, 0.18, 0.18, 0.64])
    cax = fig.add_axes([0.945, 0.18, 0.016, 0.64])

    fluxnet_label = rf"SubmesoFluxNet, mean $R^2$={mean_r2:.3f}"
    ax.plot(x, y, color=PALETTE["line"], linewidth=0.55, alpha=0.42)
    ax.plot(x, y_smooth, color=PALETTE["smooth"], linewidth=1.35, label=fluxnet_label)
    if fk_r2 is not None and fk_smooth is not None:
        fk_label = rf"MLE parameterization, mean $R^2$={float(np.nanmean(fk_r2)):.3f}"
        ax.plot(x, fk_r2, color=PALETTE["fk"], linewidth=0.55, alpha=0.30)
        ax.plot(x, fk_smooth, color=PALETTE["fk"], linewidth=1.25, label=fk_label)
    ax.axhline(mean_r2, color=PALETTE["mean"], linewidth=0.85, linestyle=(0, (3, 2)))
    if fk_r2 is not None:
        ax.axhline(float(np.nanmean(fk_r2)), color=PALETTE["fk"], linewidth=0.85, linestyle=(0, (3, 2)), alpha=0.9)

    finite_values = [y[np.isfinite(y)]]
    if fk_r2 is not None:
        finite_values.append(fk_r2[np.isfinite(fk_r2)])
    finite = np.concatenate([v for v in finite_values if v.size])
    low = float(np.nanpercentile(finite, 1))
    high = float(np.nanpercentile(finite, 99))
    pad = max(0.06 * (high - low), 0.04)
    ax.set_xlim(int(x[0]), int(x[-1]))
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel("Evaluation subdomain data index", fontsize=axis_label_fs)
    ax.set_ylabel(r"Snapshot $R^2$", fontsize=axis_label_fs)
    ax.set_title(r"Non-local $R^2$ vs time", fontsize=title_fs, pad=5)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.65)
    ax.tick_params(labelsize=tick_fs, length=2.5, width=0.7)
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.43, 0.18),
        fontsize=legend_fs,
        handlelength=1.8,
        borderaxespad=0.0,
    )

    vmax = 60.0
    image = ax_map.imshow(domain_map, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_map.axhline(19.5, color="#5F5F5F", linewidth=0.85)
    ax_map.set_xlim(-0.5, 39.5)
    ax_map.set_ylim(-0.5, 39.5)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xticks([0, 20, 40])
    ax_map.set_yticks([0, 20, 40])
    ax_map.tick_params(labelsize=tick_fs, length=2.5, width=0.7)
    ax_map.set_xlabel("x index", fontsize=axis_label_fs)
    ax_map.set_ylabel("y index", fontsize=axis_label_fs)
    ax_map.set_title("Submesoscale VHF", fontsize=title_fs, pad=5)
    ax_map.text(
        -0.05,
        0.75,
        "Training\nsubdomain",
        transform=ax_map.transAxes,
        ha="right",
        va="center",
        fontsize=annotation_fs,
        color=PALETTE["neutral_dark"],
    )
    ax_map.text(
        -0.05,
        0.25,
        "Evaluation\nSubdomain",
        transform=ax_map.transAxes,
        ha="right",
        va="center",
        fontsize=annotation_fs,
        color=PALETTE["neutral_dark"],
    )

    cbar = fig.colorbar(image, cax=cax, ticks=[-60, -40, -20, 0, 20, 40, 60])
    cbar.ax.tick_params(labelsize=tick_fs, length=2.2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(r"W m$^{-2}$", fontsize=cbar_label_fs, color=PALETTE["neutral_dark"], labelpad=4)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

def main() -> None:
    args = parse_args()
    test_position, sample_index, time_index, r2 = load_r2_csv(args.r2_series)
    if time_index is None:
        raise ValueError("The R2 series must contain time_index to align the FK/MLE time series.")
    window = int(round(args.window_days * args.snapshots_per_day)) if args.window_snapshots is None else int(args.window_snapshots)
    start, stop, mean_r2 = best_window(r2, window)
    window_r2 = r2[start:stop]
    fk_r2 = load_fk_window_r2(
        args.predictions,
        args.fk_predictions,
        args.fk_progress,
        tuple(args.fk_crop),
        time_index[start:stop],
    )
    if args.snapshot_choice == "best":
        selected_snapshot_position = start + int(np.nanargmax(window_r2))
    elif args.snapshot_choice == "worst":
        selected_snapshot_position = start + int(np.nanargmin(window_r2))
    elif args.snapshot_choice == "random-middle":
        low = float(np.nanquantile(window_r2, args.middle_quantile_low))
        high = float(np.nanquantile(window_r2, args.middle_quantile_high))
        candidates = np.flatnonzero(np.isfinite(window_r2) & (window_r2 >= low) & (window_r2 <= high))
        if candidates.size == 0:
            candidates = np.flatnonzero(np.isfinite(window_r2))
        rng = np.random.default_rng(args.snapshot_seed)
        selected_snapshot_position = start + int(rng.choice(candidates))
    else:
        selected_snapshot_position = start + (stop - start) // 2
    selected_sample = int(sample_index[selected_snapshot_position])
    domain_map = load_domain_map(args.predictions, args.upper_truth, selected_sample)

    draw_figure(args.output_prefix, sample_index, r2, start, stop, mean_r2, fk_r2, args.smooth_window, domain_map, selected_sample)
    write_window_csv(args.output_prefix.with_name(args.output_prefix.name + ".csv"), test_position, sample_index, time_index, r2, start, stop)
    write_summary(args.output_prefix.with_name(args.output_prefix.name + "_summary.txt"), args.r2_series, test_position, sample_index, time_index, r2, start, stop, mean_r2, fk_r2=fk_r2, map_snapshot_position=selected_snapshot_position)

    print(f"Best window: positions {start}-{stop - 1}", flush=True)
    print(f"Window length: {window} snapshots ({window / args.snapshots_per_day:.1f} days)", flush=True)
    print(f"Sample index: {int(sample_index[start])}-{int(sample_index[stop - 1])}", flush=True)
    if time_index is not None:
        print(f"Time index: {int(time_index[start])}-{int(time_index[stop - 1])}", flush=True)
    print(f"Mean R2: {mean_r2:.6f}", flush=True)
    print(f"MLE mean R2: {float(np.nanmean(fk_r2)):.6f}", flush=True)
    print(f"Map snapshot choice: {args.snapshot_choice}", flush=True)
    print(f"Map snapshot in window: sample={selected_sample}, R2={float(r2[selected_snapshot_position]):.6f}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.output_prefix.with_name(args.output_prefix.name + '.csv')}", flush=True)
    print(f"Saved {args.output_prefix.with_name(args.output_prefix.name + '_summary.txt')}", flush=True)


if __name__ == "__main__":
    main()
