#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "analysis/per_snapshot_r2_time_series"

PALETTE = {
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
}


@dataclass(frozen=True)
class SnapshotCase:
    sample_index: int
    time_index: int
    test_position: int
    r2: float
    truth: np.ndarray
    fk: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the best FK snapshot as truth versus FK prediction.")
    parser.add_argument(
        "--fk",
        type=Path,
        default=ANALYSIS_DIR / "no_shuffle_overfitting_test_fk_heat_flux_progress_fullgrid.npz",
        help="FK progress NPZ with keys time_index and data, or aligned FK NPZ with fk_heat_flux.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_predictions_denorm.npz",
        help="Evaluation NPZ containing sample_index and target.",
    )
    parser.add_argument(
        "--wt",
        type=Path,
        default=PROJECT_DIR.parent / "data/mld/wT.npz",
        help="Truth heat-flux NPZ with global time_index, used for progress-file alignment.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ANALYSIS_DIR / "fk_best_snapshot_truth_prediction",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ANALYSIS_DIR / "fk_best_snapshot_truth_prediction_summary.txt",
    )
    parser.add_argument("--crop", type=int, nargs=4, default=(62, 118, 2, 58), metavar=("J0", "J1", "I0", "I1"))
    parser.add_argument("--percentile", type=float, default=99.0, help="Robust symmetric color-limit percentile.")
    return parser.parse_args()


def apply_publication_style() -> None:
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
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def resolve_path(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    candidate = ANALYSIS_DIR / path
    return candidate if candidate.exists() else path


def snapshot_r2(prediction: np.ndarray, truth: np.ndarray) -> float:
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not np.any(valid):
        return float("nan")
    pred = prediction[valid]
    y = truth[valid]
    sse = float(np.sum((pred - y) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst if sst > 0.0 else float("nan")


def load_aligned_fk_and_truth(
    fk_path: Path,
    predictions_path: Path,
    wt_path: Path,
    crop: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fk_path = resolve_path(fk_path)
    predictions_path = resolve_path(predictions_path)
    wt_path = resolve_path(wt_path)

    if not fk_path.exists():
        raise FileNotFoundError(f"Missing FK file: {fk_path}")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing target file: {predictions_path}")
    if not wt_path.exists():
        raise FileNotFoundError(f"Missing time-index file: {wt_path}")

    fk_npz = np.load(fk_path)
    pred_npz = np.load(predictions_path)
    truth_all = pred_npz["target"].astype(np.float64)[:, 0]
    sample_all = pred_npz["sample_index"].astype(np.int64)

    keys = set(fk_npz.files)
    if {"sample_index", "time_index", "fk_heat_flux"}.issubset(keys):
        sample_index = fk_npz["sample_index"].astype(np.int64)
        time_index = fk_npz["time_index"].astype(np.int64)
        fk = fk_npz["fk_heat_flux"].astype(np.float64)[:, 0]
        if not np.array_equal(sample_index, sample_all):
            raise ValueError("Aligned FK sample_index does not match target sample_index.")
        truth = truth_all
        return sample_index, time_index, fk, truth

    if {"time_index", "data"}.issubset(keys):
        wt_npz = np.load(wt_path)
        test_time_all = wt_npz["time_index"].astype(np.int64)[sample_all]
        fk_time = fk_npz["time_index"].astype(np.int64)
        fk_data = fk_npz["data"].astype(np.float64)
        fk_lookup = {int(t): n for n, t in enumerate(fk_time)}
        target_pos = [n for n, t in enumerate(test_time_all) if int(t) in fk_lookup]
        if not target_pos:
            raise ValueError("No FK progress times overlap the no-shuffle test-set times.")
        j0, j1, i0, i1 = crop
        fk = np.stack([fk_data[fk_lookup[int(test_time_all[n])], j0:j1, i0:i1] for n in target_pos], axis=0)
        truth = truth_all[target_pos]
        sample_index = sample_all[target_pos]
        time_index = test_time_all[target_pos]
        return sample_index, time_index, fk, truth

    raise ValueError(f"Unsupported FK NPZ keys: {sorted(keys)}")


def select_best_fk(sample_index: np.ndarray, time_index: np.ndarray, fk: np.ndarray, truth: np.ndarray) -> SnapshotCase:
    r2 = np.asarray([snapshot_r2(fk[n], truth[n]) for n in range(fk.shape[0])], dtype=np.float64)
    best_pos = int(np.nanargmax(r2))
    return SnapshotCase(
        sample_index=int(sample_index[best_pos]),
        time_index=int(time_index[best_pos]),
        test_position=best_pos,
        r2=float(r2[best_pos]),
        truth=truth[best_pos],
        fk=fk[best_pos],
    )


def robust_symmetric_limit(values: np.ndarray, percentile: float, min_limit: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return min_limit
    limit = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.nanmax(np.abs(finite)))
    return max(limit, min_limit)


def format_panel(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_figure(case: SnapshotCase, output_prefix: Path, percentile: float) -> None:
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.9), constrained_layout=True)
    combined = np.concatenate([case.truth.ravel(), case.fk.ravel()])
    limit = robust_symmetric_limit(combined, percentile=percentile)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    for ax, title, values in zip(axes, ("Truth", "FK prediction"), (case.truth, case.fk)):
        image = ax.imshow(values, origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=7.5, pad=4)
        format_panel(ax)

    cbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(r"Heat flux (W m$^{-2}$)", fontsize=6.5, color=PALETTE["neutral_dark"], labelpad=2)

    fig.suptitle(
        fr"Best FK snapshot: $R^2$={case.r2:.3f}, sample {case.sample_index}",
        fontsize=8,
        color=PALETTE["neutral_dark"],
        y=1.03,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, case: SnapshotCase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Best FK snapshot truth-prediction diagnostic\n")
        handle.write(f"sample_index: {case.sample_index}\n")
        handle.write(f"time_index: {case.time_index}\n")
        handle.write(f"test_position_among_available_fk: {case.test_position}\n")
        handle.write(f"r2: {case.r2:.10g}\n")
        handle.write(f"truth_min: {float(np.nanmin(case.truth)):.10g}\n")
        handle.write(f"truth_max: {float(np.nanmax(case.truth)):.10g}\n")
        handle.write(f"fk_min: {float(np.nanmin(case.fk)):.10g}\n")
        handle.write(f"fk_max: {float(np.nanmax(case.fk)):.10g}\n")
        handle.write(f"truth_mean: {float(np.nanmean(case.truth)):.10g}\n")
        handle.write(f"fk_mean: {float(np.nanmean(case.fk)):.10g}\n")


def main() -> None:
    args = parse_args()
    sample_index, time_index, fk, truth = load_aligned_fk_and_truth(
        args.fk,
        args.predictions,
        args.wt,
        tuple(args.crop),
    )
    case = select_best_fk(sample_index, time_index, fk, truth)
    draw_figure(case, args.output_prefix, args.percentile)
    write_summary(args.summary_output, case)
    print(f"Best FK snapshot: sample={case.sample_index}, time={case.time_index}, R2={case.r2:.6f}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
