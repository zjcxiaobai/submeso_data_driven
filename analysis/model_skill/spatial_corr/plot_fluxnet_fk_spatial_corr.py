#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

PALETTE = {
    "fluxnet": "#245E8A",
    "fk": "#B55345",
    "neutral_dark": "#2F2F2F",
    "neutral_mid": "#7A7A7A",
    "neutral_light": "#E3E3E3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SubmesoFluxNet and MLE parameterization spatial correlations together.")
    parser.add_argument(
        "--fluxnet-csv",
        type=Path,
        default=HERE / "submesofluxnet_spatial_corr.csv",
    )
    parser.add_argument(
        "--fk-csv",
        type=Path,
        default=HERE / "fk_spatial_corr.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "submesofluxnet_mle_spatial_corr",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=HERE / "submesofluxnet_mle_spatial_corr_summary.txt",
    )
    parser.add_argument("--fk-display-label", default="MLE parameterization", help="Display label for the FK/MLE baseline in plots and summaries.")
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
            "axes.linewidth": 0.8,
            "axes.labelcolor": PALETTE["neutral_dark"],
            "xtick.color": PALETTE["neutral_dark"],
            "ytick.color": PALETTE["neutral_dark"],
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_corr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing spatial-correlation CSV: {path}")
    sample: list[int] = []
    corr: list[float] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_index", "spatial_corr"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            sample.append(int(row["sample_index"]))
            corr.append(float(row["spatial_corr"]))
    sample_arr = np.asarray(sample, dtype=np.int64)
    corr_arr = np.asarray(corr, dtype=np.float64)
    order = np.argsort(sample_arr)
    return sample_arr[order], corr_arr[order]


def align_series(
    flux_sample: np.ndarray,
    flux_corr: np.ndarray,
    fk_sample: np.ndarray,
    fk_corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if flux_sample.shape != fk_sample.shape or not np.array_equal(flux_sample, fk_sample):
        flux_set = set(map(int, flux_sample))
        fk_set = set(map(int, fk_sample))
        raise ValueError(
            "SubmesoFluxNet and MLE spatial-correlation sample_index values do not match exactly: "
            f"fluxnet_rows={flux_sample.size}, mle_rows={fk_sample.size}, common={len(flux_set & fk_set)}, "
            f"first_missing_from_mle={sorted(flux_set - fk_set)[:5]}, "
            f"first_missing_from_fluxnet={sorted(fk_set - flux_set)[:5]}"
        )
    return flux_sample, flux_corr, fk_corr


def summary(values: np.ndarray) -> dict[str, float | int]:
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
        "p95": float(np.percentile(finite, 95)),
    }


def write_summary(
    path: Path,
    sample: np.ndarray,
    flux_summary: dict[str, float | int],
    fk_summary: dict[str, float | int],
    fk_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"SubmesoFluxNet versus {fk_label} spatial correlation\n")
        handle.write("comparison_scope: exact final model_analysis raw-time-block test split\n")
        handle.write("note: SubmesoFluxNet and MLE sample_index values matched exactly before statistics were computed\n")
        handle.write(f"sample_index_range: {int(sample.min())}-{int(sample.max())}\n")
        handle.write(f"n_common_snapshots: {sample.size}\n")
        for label, vals in (("SubmesoFluxNet", flux_summary), (fk_label, fk_summary)):
            handle.write(f"\n{label}\n")
            for key, value in vals.items():
                if isinstance(value, int):
                    handle.write(f"{key}: {value}\n")
                else:
                    handle.write(f"{key}: {value:.10g}\n")


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(f"{output_prefix}.png"), dpi=600, bbox_inches="tight")


def rolling_median(values: np.ndarray, window: int = 81) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    window = min(window, values.size if values.size % 2 == 1 else values.size - 1)
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    smooth = np.empty_like(values, dtype=np.float64)
    for idx in range(values.size):
        lo = max(0, idx - half)
        hi = min(values.size, idx + half + 1)
        smooth[idx] = np.nanmedian(values[lo:hi])
    return smooth


def draw_figure(
    sample: np.ndarray,
    flux_corr: np.ndarray,
    fk_corr: np.ndarray,
    flux_summary: dict[str, float | int],
    fk_summary: dict[str, float | int],
    output_prefix: Path,
    fk_label: str,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.9, 2.35), constrained_layout=True)
    x = np.arange(sample.size, dtype=np.int64)

    flux_smooth = rolling_median(flux_corr)
    fk_smooth = rolling_median(fk_corr)

    ax.plot(x, flux_corr, color=PALETTE["fluxnet"], linewidth=0.45, alpha=0.18, zorder=1)
    ax.plot(x, fk_corr, color=PALETTE["fk"], linewidth=0.45, alpha=0.18, zorder=1)
    ax.plot(
        x,
        flux_smooth,
        color=PALETTE["fluxnet"],
        linewidth=1.35,
        label=fr"SubmesoFluxNet, mean $\rho$={float(flux_summary['mean']):.3f}",
        zorder=3,
    )
    ax.plot(
        x,
        fk_smooth,
        color=PALETTE["fk"],
        linewidth=1.35,
        label=fr"{fk_label}, mean $\rho$={float(fk_summary['mean']):.3f}",
        zorder=3,
    )

    ax.axhline(float(flux_summary["mean"]), color=PALETTE["fluxnet"], linewidth=0.75, linestyle=(0, (3, 2)), alpha=0.75, zorder=2)
    ax.axhline(float(fk_summary["mean"]), color=PALETTE["fk"], linewidth=0.75, linestyle=(0, (3, 2)), alpha=0.75, zorder=2)

    all_values = np.concatenate([flux_corr[np.isfinite(flux_corr)], fk_corr[np.isfinite(fk_corr)]])
    ymin = max(-1.0, float(np.nanpercentile(all_values, 1)) - 0.04)
    ymax = min(1.0, float(np.nanpercentile(all_values, 99)) + 0.035)
    ax.set_xlim(0, int(x.max()))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Held-out test sample number")
    ax.set_ylabel(r"Spatial correlation, $\rho_n$")
    ax.set_title("Spatial pattern correlation", fontsize=8.2, pad=4)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.65, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="center",
        bbox_to_anchor=(0.7, 0.65),
        fontsize=6.7,
        handlelength=1.8,
        borderaxespad=0.0,
        frameon=False,
    )

    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])
    ax.tick_params(labelsize=6.8)

    save_figure(fig, output_prefix)
    plt.close(fig)

def main() -> None:
    args = parse_args()
    flux_sample, flux_corr = read_corr(args.fluxnet_csv)
    fk_sample, fk_corr = read_corr(args.fk_csv)
    sample, flux_aligned, fk_aligned = align_series(flux_sample, flux_corr, fk_sample, fk_corr)
    flux_summary = summary(flux_aligned)
    fk_summary = summary(fk_aligned)

    draw_figure(sample, flux_aligned, fk_aligned, flux_summary, fk_summary, args.output_prefix, args.fk_display_label)
    write_summary(args.summary_output, sample, flux_summary, fk_summary, args.fk_display_label)

    print(f"Saved {args.output_prefix}.png", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"SubmesoFluxNet mean spatial corr: {float(flux_summary['mean']):.6f}", flush=True)
    print(f"{args.fk_display_label} mean spatial corr: {float(fk_summary['mean']):.6f}", flush=True)


if __name__ == "__main__":
    main()
