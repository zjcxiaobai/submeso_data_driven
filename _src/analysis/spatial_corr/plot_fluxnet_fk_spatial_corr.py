#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

PALETTE = {
    "fluxnet": "#1F5A85",
    "fk": "#B64A3B",
    "neutral_dark": "#343434",
    "neutral_mid": "#777777",
    "neutral_light": "#D6D6D6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SubmesoFluxNet and FK spatial correlations together.")
    parser.add_argument(
        "--fluxnet-csv",
        type=Path,
        default=HERE / "no_shuffle_overfitting_test_spatial_corr.csv",
    )
    parser.add_argument(
        "--fk-csv",
        type=Path,
        default=HERE / "fk_Ce2p13062237_spatial_corr.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "fluxnet_fk_spatial_corr",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=HERE / "fluxnet_fk_spatial_corr_summary.txt",
    )
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
            "axes.linewidth": 0.75,
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
    flux_lookup = {int(s): i for i, s in enumerate(flux_sample)}
    fk_lookup = {int(s): i for i, s in enumerate(fk_sample)}
    common = np.asarray(sorted(set(flux_lookup).intersection(fk_lookup)), dtype=np.int64)
    if common.size == 0:
        raise ValueError("No common sample_index values between SubmesoFluxNet and FK spatial-correlation CSVs.")
    flux = np.asarray([flux_corr[flux_lookup[int(s)]] for s in common], dtype=np.float64)
    fk = np.asarray([fk_corr[fk_lookup[int(s)]] for s in common], dtype=np.float64)
    return common, flux, fk


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("SubmesoFluxNet versus FK spatial correlation\n")
        handle.write(f"sample_index_range: {int(sample.min())}-{int(sample.max())}\n")
        handle.write(f"n_common_snapshots: {sample.size}\n")
        for label, vals in (("SubmesoFluxNet", flux_summary), ("FK", fk_summary)):
            handle.write(f"\n{label}\n")
            for key, value in vals.items():
                if isinstance(value, int):
                    handle.write(f"{key}: {value}\n")
                else:
                    handle.write(f"{key}: {value:.10g}\n")


def save_figure(fig: plt.Figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".svg", ".pdf", ".png", ".tiff"):
        kwargs = {"bbox_inches": "tight"}
        if ext in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(Path(f"{output_prefix}{ext}"), **kwargs)


def draw_figure(
    sample: np.ndarray,
    flux_corr: np.ndarray,
    fk_corr: np.ndarray,
    flux_summary: dict[str, float | int],
    fk_summary: dict[str, float | int],
    output_prefix: Path,
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.85, 2.55), constrained_layout=True)

    ax.plot(
        sample,
        flux_corr,
        color=PALETTE["fluxnet"],
        linewidth=0.95,
        label=fr"SubmesoFluxNet, mean $\rho$={float(flux_summary['mean']):.3f}",
    )
    ax.plot(
        sample,
        fk_corr,
        color=PALETTE["fk"],
        linewidth=0.95,
        label=fr"FK, mean $\rho$={float(fk_summary['mean']):.3f}",
    )
    ax.set_xlim(int(sample.min()), int(sample.max()))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Held-out test data index")
    ax.set_ylabel(r"Spatial correlation $\rho_n$")
    ax.set_title("Spatial pattern correlation", fontsize=7.5, pad=4)
    ax.grid(axis="y", color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
    ax.spines["left"].set_color(PALETTE["neutral_mid"])
    ax.spines["bottom"].set_color(PALETTE["neutral_mid"])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        fontsize=6.7,
        handlelength=1.7,
        columnspacing=1.4,
    )

    save_figure(fig, output_prefix)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    flux_sample, flux_corr = read_corr(args.fluxnet_csv)
    fk_sample, fk_corr = read_corr(args.fk_csv)
    sample, flux_aligned, fk_aligned = align_series(flux_sample, flux_corr, fk_sample, fk_corr)
    flux_summary = summary(flux_aligned)
    fk_summary = summary(fk_aligned)

    draw_figure(sample, flux_aligned, fk_aligned, flux_summary, fk_summary, args.output_prefix)
    write_summary(args.summary_output, sample, flux_summary, fk_summary)

    for ext in (".svg", ".pdf", ".png", ".tiff"):
        print(f"Saved {args.output_prefix}{ext}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"SubmesoFluxNet mean spatial corr: {float(flux_summary['mean']):.6f}", flush=True)
    print(f"FK mean spatial corr: {float(fk_summary['mean']):.6f}", flush=True)


if __name__ == "__main__":
    main()
