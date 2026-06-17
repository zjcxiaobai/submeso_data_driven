#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


PROJECT_DIR = Path(__file__).resolve().parents[2]
PALETTE = {
    "truth": "#0F4D92",
    "prediction": "#B64342",
    "ratio": "#42949E",
    "reference": "#4D4D4D",
    "neutral_mid": "#767676",
    "neutral_light": "#CFCECE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate isotropic spectral ratio E_pred(k) / E_truth(k) for no-shuffle test predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_DIR / "outputs_no_shuffle/test_predictions_denorm.npz",
        help="NPZ produced by evaluate.py containing prediction, target, and sample_index.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spectra_analysis/no_shuffle_spectra_ratio.csv",
        help="Output CSV for annularly binned spectra and spectrum ratio.",
    )
    parser.add_argument(
        "--npz-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spectra_analysis/no_shuffle_spectra_ratio.npz",
        help="Output NPZ with mean and per-snapshot spectra.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR / "analysis/spectra_analysis/no_shuffle_spectra_ratio_summary.txt",
        help="Output text file with spectral-ratio band summaries.",
    )
    parser.add_argument(
        "--plot-output-prefix",
        type=Path,
        default=PROJECT_DIR / "analysis/spectra_analysis/no_shuffle_spectra_ratio",
        help="Output plot prefix without extension.",
    )
    parser.add_argument(
        "--keep-mean",
        action="store_true",
        help="Keep each snapshot's spatial mean, including k=0 energy. Default removes the mean and omits k=0.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run _src/evaluate.py for the no-shuffle model first.")

    with np.load(path) as data:
        required = {"prediction", "target", "sample_index"}
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
    if prediction.ndim != 4:
        raise ValueError(f"expected arrays shaped (sample, channel, y, x), got {prediction.shape}")
    return sample_index, prediction, target


def radial_bin_index(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    radial_k = np.sqrt(kx_grid**2 + ky_grid**2)
    bin_index = np.rint(radial_k).astype(np.int64)
    bins = np.arange(0, int(bin_index.max()) + 1, dtype=np.int64)
    bin_counts = np.bincount(bin_index.ravel(), minlength=bins.size).astype(np.int64)
    return bin_index, bins, bin_counts


def binned_isotropic_spectra(fields: np.ndarray, remove_mean: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, ny, nx = fields.shape
    bin_index, bins, bin_counts = radial_bin_index(ny, nx)
    spectra = np.zeros((fields.shape[0], bins.size), dtype=np.float64)

    for sample_id, field in enumerate(fields):
        sample_power = np.zeros(bins.size, dtype=np.float64)
        for channel in field:
            channel_field = channel
            if remove_mean:
                channel_field = channel_field - np.mean(channel_field)
            fft = np.fft.fft2(channel_field)
            power = np.abs(fft) ** 2 / (ny * nx)
            sample_power += np.bincount(bin_index.ravel(), weights=power.ravel(), minlength=bins.size)
        spectra[sample_id] = sample_power / field.shape[0]

    return bins, bin_counts, spectra


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0.0)
    return ratio


def write_csv(
    path: Path,
    k_mode: np.ndarray,
    bin_counts: np.ndarray,
    truth_spectrum: np.ndarray,
    pred_spectrum: np.ndarray,
    ratio: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["k_mode", "bin_count", "E_truth", "E_prediction", "E_prediction_over_E_truth"],
        )
        writer.writeheader()
        for row in zip(k_mode, bin_counts, truth_spectrum, pred_spectrum, ratio):
            writer.writerow(
                {
                    "k_mode": int(row[0]),
                    "bin_count": int(row[1]),
                    "E_truth": float(row[2]),
                    "E_prediction": float(row[3]),
                    "E_prediction_over_E_truth": float(row[4]),
                }
            )


def summarize_bands(k_mode: np.ndarray, ratio: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(ratio)
    summary = {
        "ratio_mean_all_k": float(np.mean(ratio[finite])) if np.any(finite) else float("nan"),
        "ratio_median_all_k": float(np.median(ratio[finite])) if np.any(finite) else float("nan"),
    }
    max_k = int(np.max(k_mode))
    bands = {
        "large_scale_k_1_4": (1, 4),
        "mesoscale_k_5_12": (5, 12),
        "small_scale_k_13_max": (13, max_k),
    }
    for name, (lo, hi) in bands.items():
        keep = (k_mode >= lo) & (k_mode <= hi) & finite
        summary[f"{name}_mean_ratio"] = float(np.mean(ratio[keep])) if np.any(keep) else float("nan")
        summary[f"{name}_median_ratio"] = float(np.median(ratio[keep])) if np.any(keep) else float("nan")
    return summary


def write_summary(
    path: Path,
    sample_count: int,
    shape: tuple[int, ...],
    remove_mean: bool,
    summary: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("No-shuffle isotropic spectrum-ratio summary\n")
        handle.write(f"samples: {sample_count}\n")
        handle.write(f"field_shape: {shape}\n")
        handle.write(f"remove_spatial_mean: {remove_mean}\n")
        handle.write("k0_included: false\n" if remove_mean else "k0_included: true\n")
        for key, value in summary.items():
            handle.write(f"{key}: {value:.10g}\n")


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def plot_spectra(
    path_prefix: Path,
    k_mode: np.ndarray,
    truth_spectrum: np.ndarray,
    pred_spectrum: np.ndarray,
    ratio: np.ndarray,
) -> None:
    apply_publication_style()
    fig, (ax_spec, ax_ratio) = plt.subplots(
        1,
        2,
        figsize=(5.2, 2.35),
        width_ratios=(1.15, 1.0),
        constrained_layout=True,
    )

    ax_spec.loglog(k_mode, truth_spectrum, color=PALETTE["truth"], linewidth=1.1, label=r"Truth $E_y(k)$")
    ax_spec.loglog(
        k_mode,
        pred_spectrum,
        color=PALETTE["prediction"],
        linewidth=1.1,
        label=r"Prediction $E_{\hat{y}}(k)$",
    )
    ax_spec.set_xlabel("Radial wavenumber mode")
    ax_spec.set_ylabel("Annular spectral energy")
    ax_spec.set_title("Mean isotropic spectra")
    ax_spec.legend(loc="best", handlelength=2.0)

    ax_ratio.semilogx(k_mode, ratio, color=PALETTE["ratio"], linewidth=1.2)
    ax_ratio.axhline(1.0, color=PALETTE["reference"], linestyle="--", linewidth=0.8)
    ax_ratio.set_xlabel("Radial wavenumber mode")
    ax_ratio.set_ylabel(r"$E_{\hat{y}}(k) / E_y(k)$")
    ax_ratio.set_title("Prediction/truth spectrum ratio")
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        upper = max(1.2, float(np.percentile(finite, 98)) * 1.1)
        ax_ratio.set_ylim(0.0, upper)

    for ax in (ax_spec, ax_ratio):
        ax.grid(True, color=PALETTE["neutral_light"], linewidth=0.45, alpha=0.55)
        ax.tick_params(colors=PALETTE["neutral_mid"])

    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_index, prediction, target = load_predictions(args.predictions)
    remove_mean = not args.keep_mean

    bins, bin_counts, pred_spectra_all = binned_isotropic_spectra(prediction, remove_mean=remove_mean)
    _, _, truth_spectra_all = binned_isotropic_spectra(target, remove_mean=remove_mean)

    keep = bin_counts > 0
    if remove_mean:
        keep &= bins > 0
    k_mode = bins[keep]
    bin_counts = bin_counts[keep]
    pred_spectra_all = pred_spectra_all[:, keep]
    truth_spectra_all = truth_spectra_all[:, keep]

    pred_spectrum = np.mean(pred_spectra_all, axis=0)
    truth_spectrum = np.mean(truth_spectra_all, axis=0)
    ratio = safe_ratio(pred_spectrum, truth_spectrum)
    ratio_all = safe_ratio(pred_spectra_all, truth_spectra_all)
    summary = summarize_bands(k_mode, ratio)

    write_csv(args.csv_output, k_mode, bin_counts, truth_spectrum, pred_spectrum, ratio)
    args.npz_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.npz_output,
        sample_index=sample_index,
        k_mode=k_mode,
        bin_count=bin_counts,
        E_truth=truth_spectrum,
        E_prediction=pred_spectrum,
        E_prediction_over_E_truth=ratio,
        E_truth_per_snapshot=truth_spectra_all,
        E_prediction_per_snapshot=pred_spectra_all,
        E_prediction_over_E_truth_per_snapshot=ratio_all,
        remove_spatial_mean=np.asarray(remove_mean),
    )
    write_summary(args.summary_output, sample_index.size, prediction.shape, remove_mean, summary)
    plot_spectra(args.plot_output_prefix, k_mode, truth_spectrum, pred_spectrum, ratio)

    print(f"Saved {args.csv_output}", flush=True)
    print(f"Saved {args.npz_output}", flush=True)
    print(f"Saved {args.summary_output}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.svg')}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.pdf')}", flush=True)
    print(f"Saved {args.plot_output_prefix.with_suffix('.png')}", flush=True)
    print(f"mean spectrum ratio across k: {summary['ratio_mean_all_k']:.6f}", flush=True)


if __name__ == "__main__":
    main()
