#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parents[2]

PALETTE = {
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


@dataclass(frozen=True)
class SnapshotCase:
    label: str
    sample_index: int
    test_position: int
    r2: float
    prediction: np.ndarray
    target: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best and worst per-snapshot R2 maps: truth, prediction, and normalized squared error."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_predictions_denorm.npz",
        help="NPZ produced by evaluate.py with prediction, target, and sample_index arrays.",
    )
    parser.add_argument(
        "--r2-series",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle_overfitting_test/test_r2_series.csv",
        help="CSV produced by evaluate.py with per-snapshot R2 values.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_DIR
        / "analysis/per_snapshot_r2_time_series/no_shuffle_overfitting_test_best_worst_maps",
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR
        / "analysis/per_snapshot_r2_time_series/no_shuffle_overfitting_test_best_worst_maps_summary.txt",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Robust percentile for truth/prediction color scale.",
    )
    parser.add_argument(
        "--error-percentile",
        type=float,
        default=99.0,
        help="Robust percentile for normalized squared-error color scale.",
    )
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
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_r2_rows(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run evaluate.py first.")

    rows: list[dict[str, float | int]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"test_position", "sample_index", "r2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "test_position": int(row["test_position"]),
                    "sample_index": int(row["sample_index"]),
                    "r2": float(row["r2"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run evaluate.py first.")
    with np.load(path) as data:
        required = {"sample_index", "prediction", "target"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        sample_index = np.asarray(data["sample_index"], dtype=np.int64)
        prediction = np.asarray(data["prediction"], dtype=np.float64)
        target = np.asarray(data["target"], dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")
    if sample_index.shape[0] != prediction.shape[0]:
        raise ValueError("sample_index length does not match prediction sample count")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(f"Expected prediction/target arrays shaped (N,1,H,W), got {prediction.shape}")
    return sample_index, prediction[:, 0], target[:, 0]


def select_cases(r2_rows: list[dict[str, float | int]], predictions_path: Path) -> tuple[SnapshotCase, SnapshotCase]:
    sample_index, prediction, target = load_predictions(predictions_path)
    lookup = {int(sample): pos for pos, sample in enumerate(sample_index)}

    best_row = max(r2_rows, key=lambda row: float(row["r2"]))
    worst_row = min(r2_rows, key=lambda row: float(row["r2"]))

    cases: list[SnapshotCase] = []
    for label, row in (("Best case", best_row), ("Worst case", worst_row)):
        sample = int(row["sample_index"])
        if sample not in lookup:
            raise ValueError(f"sample_index {sample} from R2 CSV was not found in {predictions_path}")
        pos = lookup[sample]
        cases.append(
            SnapshotCase(
                label=label,
                sample_index=sample,
                test_position=int(row["test_position"]),
                r2=float(row["r2"]),
                prediction=prediction[pos],
                target=target[pos],
            )
        )
    return cases[0], cases[1]


def robust_symmetric_limit(values: np.ndarray, percentile: float, min_limit: float = 1.0) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return min_limit
    limit = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(np.abs(finite)))
    return max(limit, min_limit)


def normalized_squared_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_std = float(np.nanstd(target))
    if not np.isfinite(target_std) or target_std <= 0.0:
        target_std = 1.0
    return ((prediction - target) / target_std) ** 2


def format_panel(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_colorbar(fig: plt.Figure, image, ax: plt.Axes, label: str) -> None:
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=6, length=2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(label, fontsize=6.5, color=PALETTE["neutral_dark"], labelpad=2)


def draw_figure(
    best: SnapshotCase,
    worst: SnapshotCase,
    output_prefix: Path,
    percentile: float,
    error_percentile: float,
) -> None:
    apply_publication_style()
    cases = [best, worst]
    fig, axes = plt.subplots(2, 3, figsize=(4.8, 2.2), constrained_layout=True)

    error_maps = [normalized_squared_error(case.prediction, case.target) for case in cases]
    error_values = np.concatenate([error.ravel() for error in error_maps])
    error_limit = float(np.nanpercentile(error_values, error_percentile))
    if not np.isfinite(error_limit) or error_limit <= 0.0:
        error_limit = float(np.nanmax(error_values))
    error_limit = max(error_limit, 1.0e-6)
    shared_error_norm = mpl.colors.Normalize(vmin=0.0, vmax=error_limit)

    for row, case in enumerate(cases):
        combined = np.concatenate([case.target.ravel(), case.prediction.ravel()])
        field_limit = robust_symmetric_limit(combined, percentile=percentile, min_limit=1.0)
        field_norm = TwoSlopeNorm(vmin=-field_limit, vcenter=0.0, vmax=field_limit)
        squared_error = error_maps[row]

        panels = (
            ("Truth", case.target, "RdBu_r", field_norm, "Heat flux"),
            ("Prediction", case.prediction, "RdBu_r", field_norm, "Heat flux"),
            ("Normalized squared error", squared_error, "magma", shared_error_norm, r"$(\Delta/\sigma_y)^2$"),
        )

        for col, (title, values, cmap, norm, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(values, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
            format_panel(ax)
            if row == 0:
                ax.set_title(title, fontsize=7.5, pad=4)
            add_colorbar(fig, image, ax, cbar_label)

        axes[row, 0].text(
            -0.12,
            0.5,
            f"{case.label}\n"
            fr"$R^2$={case.r2:.3f}" "\n"
            f"sample {case.sample_index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7,
            color=PALETTE["neutral_dark"],
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, best: SnapshotCase, worst: SnapshotCase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("Best and worst no-shuffle overfitting-test snapshots\n")
        handle.write("error_definition: normalized_squared_error = ((prediction - truth) / std(truth_snapshot))^2\n")
        for case in (best, worst):
            err = normalized_squared_error(case.prediction, case.target)
            handle.write(f"\n{case.label}\n")
            handle.write(f"sample_index: {case.sample_index}\n")
            handle.write(f"test_position: {case.test_position}\n")
            handle.write(f"r2: {case.r2:.10g}\n")
            handle.write(f"truth_min: {float(np.nanmin(case.target)):.10g}\n")
            handle.write(f"truth_max: {float(np.nanmax(case.target)):.10g}\n")
            handle.write(f"prediction_min: {float(np.nanmin(case.prediction)):.10g}\n")
            handle.write(f"prediction_max: {float(np.nanmax(case.prediction)):.10g}\n")
            handle.write(f"normalized_squared_error_mean: {float(np.nanmean(err)):.10g}\n")
            handle.write(f"normalized_squared_error_p95: {float(np.nanpercentile(err, 95)):.10g}\n")
            handle.write(f"normalized_squared_error_p99: {float(np.nanpercentile(err, 99)):.10g}\n")


def main() -> None:
    args = parse_args()
    r2_rows = read_r2_rows(args.r2_series)
    best, worst = select_cases(r2_rows, args.predictions)
    draw_figure(
        best,
        worst,
        args.output_prefix,
        percentile=args.percentile,
        error_percentile=args.error_percentile,
    )
    write_summary(args.summary_output, best, worst)

    print(f"Best case: sample={best.sample_index}, R2={best.r2:.6f}", flush=True)
    print(f"Worst case: sample={worst.sample_index}, R2={worst.r2:.6f}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {args.output_prefix.with_suffix('.tiff')}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)


if __name__ == "__main__":
    main()
