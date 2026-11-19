#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_DIR / "results/model/test_predictions_denorm.npz"
OUTPUT_PATH = (
    PROJECT_DIR
    / "results/analysis_outputs/snapshot_level_heat_flux_prediction_skill/best_worst_maps.png"
)
PERCENTILE = 99.0

PALETTE = {
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.linewidth"] = 0.65
mpl.rcParams["axes.labelcolor"] = PALETTE["neutral_dark"]
mpl.rcParams["xtick.color"] = PALETTE["neutral_dark"]
mpl.rcParams["ytick.color"] = PALETTE["neutral_dark"]
mpl.rcParams["xtick.major.width"] = 0.65
mpl.rcParams["ytick.major.width"] = 0.65
mpl.rcParams["xtick.major.size"] = 2.5
mpl.rcParams["ytick.major.size"] = 2.5


@dataclass(frozen=True)
class SnapshotCase:
    label: str
    sample_index: int
    r2: float
    prediction: np.ndarray
    target: np.ndarray

def load_cases() -> tuple[SnapshotCase, SnapshotCase]:
    with np.load(DATA_PATH) as data:
        sample_index = data["sample_index"]
        prediction = data["prediction"][:, 0]
        target = data["target"][:, 0]
        r2 = data["r2"]

    cases = []
    for label, position in (
        ("Best case", int(np.nanargmax(r2))),
        ("Worst case", int(np.nanargmin(r2))),
    ):
        cases.append(
            SnapshotCase(
                label=label,
                sample_index=int(sample_index[position]),
                r2=float(r2[position]),
                prediction=prediction[position],
                target=target[position],
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
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.55, colors=PALETTE["neutral_dark"])
    cbar.outline.set_linewidth(0.55)
    cbar.set_label(label, fontsize=4.8, color=PALETTE["neutral_dark"], labelpad=2)


def draw_figure(
    best: SnapshotCase,
    worst: SnapshotCase,
) -> None:
    cases = [best, worst]
    fig, axes = plt.subplots(2, 3, figsize=(4.8, 1.3), constrained_layout=True)

    error_maps = [normalized_squared_error(case.prediction, case.target) for case in cases]
    error_values = np.concatenate([error.ravel() for error in error_maps])
    error_limit = float(np.nanpercentile(error_values, PERCENTILE))
    if not np.isfinite(error_limit) or error_limit <= 0.0:
        error_limit = float(np.nanmax(error_values))
    error_limit = max(error_limit, 1.0e-6)
    shared_error_norm = mpl.colors.Normalize(vmin=0.0, vmax=error_limit)

    for row, case in enumerate(cases):
        combined = np.concatenate([case.target.ravel(), case.prediction.ravel()])
        field_limit = robust_symmetric_limit(
            combined, percentile=PERCENTILE, min_limit=1.0
        )
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
                ax.set_title(title, fontsize=5.5, pad=4)
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
            fontsize=6,
            color=PALETTE["neutral_dark"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    best, worst = load_cases()
    draw_figure(best, worst)

    print(f"Best case: sample={best.sample_index}, R2={best.r2:.6f}", flush=True)
    print(f"Worst case: sample={worst.sample_index}, R2={worst.r2:.6f}", flush=True)
    print(f"Saved {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
