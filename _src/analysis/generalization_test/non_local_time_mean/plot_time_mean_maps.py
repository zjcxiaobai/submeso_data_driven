#!/usr/bin/env python3
"""Plot non-local truth/prediction snapshots and time-mean maps for the selected one-year window."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
NON_LOCAL_DIR = HERE.parent / "non_local"

PALETTE = {
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
}


@dataclass(frozen=True)
class RowSpec:
    title: str
    indices: np.ndarray
    r2: float
    extra: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=NON_LOCAL_DIR / "upper_model_on_lower_40x20_model_analysis_predictions_denorm.npz",
    )
    parser.add_argument(
        "--window-csv",
        type=Path,
        default=NON_LOCAL_DIR / "evaluation_subdomain_best_1_year_window.csv",
        help="Best one-year window CSV used to select the plotted year.",
    )
    parser.add_argument("--snapshots-per-day", type=float, default=2.0)
    parser.add_argument("--monthly-days", type=float, default=30.0)
    parser.add_argument("--seasonal-days", type=float, default=90.0)
    parser.add_argument("--monthly-selection", choices=("best", "center", "first"), default="best")
    parser.add_argument("--seasonal-selection", choices=("best", "center", "first"), default="best")
    parser.add_argument("--field-percentile", type=float, default=99.0)
    parser.add_argument("--output-prefix", type=Path, default=HERE / "time_mean_maps")
    parser.add_argument("--summary-output", type=Path, default=HERE / "time_mean_maps_summary.txt")
    parser.add_argument("--data-output", type=Path, default=HERE / "time_mean_maps_data.npz")
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_window_time_index(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing one-year window CSV: {path}")
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "time_index" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing time_index")
        values = [int(row["time_index"]) for row in reader if row["time_index"] != ""]
    if not values:
        raise ValueError(f"No time_index values found in {path}")
    return np.asarray(values, dtype=np.int64)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    with np.load(path) as data:
        required = {"sample_index", "time_index", "prediction", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        time_index = np.asarray(data["time_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected prediction shape (N,1,H,W), got {prediction.shape}")
    return sample_index, time_index, prediction[:, 0], target[:, 0]


def align_to_window(
    sample_index: np.ndarray,
    time_index: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    window_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lookup = {int(t): n for n, t in enumerate(time_index)}
    missing = [int(t) for t in window_time if int(t) not in lookup]
    if missing:
        raise ValueError(f"Prediction file misses {len(missing)} selected window times. First missing: {missing[:8]}")
    positions = np.asarray([lookup[int(t)] for t in window_time], dtype=np.int64)
    return sample_index[positions], time_index[positions], prediction[positions], target[positions]


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.nansum((prediction - target) ** 2))
    ss_tot = float(np.nansum((target - np.nanmean(target)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def sample_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = prediction - target
    sse = np.sum(residual**2, axis=(1, 2))
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    sst = np.sum((target - target_mean) ** 2, axis=(1, 2))
    out = np.full(sse.shape, np.nan, dtype=np.float64)
    valid = sst > 0.0
    out[valid] = 1.0 - sse[valid] / sst[valid]
    return out


def block_mean_rows(
    title: str,
    prediction: np.ndarray,
    target: np.ndarray,
    block: int,
    snapshots_per_day: float,
    selection: str,
) -> RowSpec:
    n_blocks = prediction.shape[0] // block
    if n_blocks < 1:
        raise ValueError(f"{title} block length {block} exceeds selected window length {prediction.shape[0]}")
    starts = np.arange(n_blocks, dtype=np.int64) * block
    block_r2 = []
    for start in starts:
        stop = start + block
        block_r2.append(r2_score(np.nanmean(prediction[start:stop], axis=0), np.nanmean(target[start:stop], axis=0)))
    block_r2 = np.asarray(block_r2, dtype=np.float64)
    if selection == "best":
        chosen = int(np.nanargmax(block_r2))
    elif selection == "center":
        chosen = int(n_blocks // 2)
    else:
        chosen = 0
    start = int(starts[chosen])
    indices = np.arange(start, start + block, dtype=np.int64)
    days = int(round(block / snapshots_per_day))
    return RowSpec(title=title, indices=indices, r2=float(block_r2[chosen]), extra=f"{days} model days")


def robust_symmetric_limit(values: np.ndarray, percentile: float, min_limit: float = 1.0) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return min_limit
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(np.abs(finite)))
    return max(limit, min_limit)


def format_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_row_label(ax: plt.Axes, row: RowSpec) -> None:
    label = f"{row.title}\n$R^2$={row.r2:.3f}"
    if row.extra:
        label += f"\n{row.extra}"
    ax.text(0.98, 0.5, label, ha="right", va="center", fontsize=8.2, color="black", transform=ax.transAxes, linespacing=1.05)
    ax.set_axis_off()


def draw_figure(
    rows: list[RowSpec],
    prediction: np.ndarray,
    target: np.ndarray,
    field_percentile: float,
    output_prefix: Path,
) -> None:
    apply_style()
    truth_maps = [np.nanmean(target[row.indices], axis=0) for row in rows]
    pred_maps = [np.nanmean(prediction[row.indices], axis=0) for row in rows]
    field_limit = robust_symmetric_limit(np.concatenate([m.ravel() for m in truth_maps + pred_maps]), field_percentile, min_limit=1.0)
    norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)

    fig = plt.figure(figsize=(7.3, 7.2))
    grid = fig.add_gridspec(
        len(rows),
        4,
        width_ratios=[0.90, 1.25, 1.25, 0.045],
        height_ratios=[1.0] * len(rows),
        left=0.055,
        right=0.94,
        bottom=0.04,
        top=0.955,
        wspace=0.055,
        hspace=0.36,
    )

    first_image = None
    for i, row in enumerate(rows):
        ax_label = fig.add_subplot(grid[i, 0])
        ax_truth = fig.add_subplot(grid[i, 1])
        ax_pred = fig.add_subplot(grid[i, 2])
        cax = fig.add_subplot(grid[i, 3])

        add_row_label(ax_label, row)
        first_image = ax_truth.imshow(truth_maps[i], origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax_pred.imshow(pred_maps[i], origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
        format_map_axis(ax_truth)
        format_map_axis(ax_pred)
        if i == 0:
            ax_truth.set_title("Truth", fontsize=10.2, pad=4)
            ax_pred.set_title("SubmesoFluxNet", fontsize=10.2, pad=4)
        cbar = fig.colorbar(first_image, cax=cax)
        cbar.ax.tick_params(labelsize=7.0, length=2.2, width=0.55, colors=PALETTE["neutral_dark"])
        cbar.outline.set_linewidth(0.55)
        cbar.set_label(r"$w'T'$", fontsize=7.2, color=PALETTE["neutral_dark"], labelpad=3)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_prefix}.png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_outputs(
    rows: list[RowSpec],
    sample_index: np.ndarray,
    time_index: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    summary_output: Path,
    data_output: Path,
) -> None:
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w") as handle:
        handle.write("Non-local selected-year truth/prediction time-mean maps\n")
        handle.write("time_window: evaluation_subdomain_best_1_year_window.csv\n")
        handle.write(f"n_window_snapshots: {time_index.size}\n")
        handle.write(f"start_time_index: {int(time_index[0])}\n")
        handle.write(f"end_time_index: {int(time_index[-1])}\n")
        for row in rows:
            handle.write(f"\n{row.title}\n")
            handle.write(f"r2: {row.r2:.10g}\n")
            handle.write(f"n_snapshots: {row.indices.size}\n")
            handle.write(f"start_sample_index: {int(sample_index[row.indices[0]])}\n")
            handle.write(f"end_sample_index: {int(sample_index[row.indices[-1]])}\n")
            handle.write(f"start_time_index: {int(time_index[row.indices[0]])}\n")
            handle.write(f"end_time_index: {int(time_index[row.indices[-1]])}\n")
    np.savez_compressed(
        data_output,
        row_title=np.asarray([row.title for row in rows]),
        row_extra=np.asarray([row.extra for row in rows]),
        row_r2=np.asarray([row.r2 for row in rows], dtype=np.float64),
        truth=np.stack([np.nanmean(target[row.indices], axis=0) for row in rows]).astype(np.float32),
        prediction=np.stack([np.nanmean(prediction[row.indices], axis=0) for row in rows]).astype(np.float32),
        sample_index=sample_index.astype(np.int64),
        time_index=time_index.astype(np.int64),
    )


def main() -> None:
    args = parse_args()
    sample_index, time_index, prediction, target = load_predictions(args.predictions)
    window_time = load_window_time_index(args.window_csv)
    sample_index, time_index, prediction, target = align_to_window(sample_index, time_index, prediction, target, window_time)

    r2 = sample_r2(prediction, target)
    best_idx = int(np.nanargmax(r2))
    worst_idx = int(np.nanargmin(r2))
    monthly_block = max(1, int(round(args.monthly_days * args.snapshots_per_day)))
    seasonal_block = max(1, int(round(args.seasonal_days * args.snapshots_per_day)))
    annual_block = time_index.size

    rows = [
        RowSpec("Best snapshot", np.asarray([best_idx], dtype=np.int64), float(r2[best_idx])),
        RowSpec("Worst snapshot", np.asarray([worst_idx], dtype=np.int64), float(r2[worst_idx])),
        block_mean_rows("Monthly mean", prediction, target, monthly_block, args.snapshots_per_day, args.monthly_selection),
        block_mean_rows("Seasonal mean", prediction, target, seasonal_block, args.snapshots_per_day, args.seasonal_selection),
        RowSpec("Annual mean", np.arange(annual_block, dtype=np.int64), r2_score(np.nanmean(prediction, axis=0), np.nanmean(target, axis=0)), "365 model days"),
    ]

    draw_figure(rows, prediction, target, args.field_percentile, args.output_prefix)
    write_outputs(rows, sample_index, time_index, prediction, target, args.summary_output, args.data_output)

    print(f"Window snapshots: {time_index.size}", flush=True)
    print(f"Time index: {int(time_index[0])}-{int(time_index[-1])}", flush=True)
    for row in rows:
        print(f"{row.title}: R2={row.r2:.6f}, n={row.indices.size}", flush=True)
    print(f"Saved {args.output_prefix}.png", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.data_output}", flush=True)


if __name__ == "__main__":
    main()
