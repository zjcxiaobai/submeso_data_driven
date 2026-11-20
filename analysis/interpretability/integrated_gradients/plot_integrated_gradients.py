#!/usr/bin/env python3
"""Plot the three Integrated Gradients diagnostics used in the paper."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT = Path(__file__).resolve().with_name("top10_gridcell_ig.npz")
OUTPUT_DIR = PROJECT_ROOT / "results/analysis_outputs/interpretability/integrated_gradients"

FEATURE_LABELS = {
    "U velocity": r"$u$",
    "V velocity": r"$v$",
    "W velocity": r"$w$",
    "Vorticity": r"$\zeta$",
    "Strain": r"$S$",
    "Potential temperature": r"$T$",
    "Density": r"$\rho$",
    "KPP boundary layer depth": r"$h_{\mathrm{bl}}$",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def target_centered_plot(
    centered: np.ndarray,
    names: np.ndarray,
    dx_km: float,
    dy_km: float,
) -> None:
    _, height, width = centered.shape
    x = (np.arange(width) - width // 2) * dx_km
    y = (np.arange(height) - height // 2) * dy_km
    extent = (x[0] - dx_km / 2, x[-1] + dx_km / 2, y[0] - dy_km / 2, y[-1] + dy_km / 2)
    vmax = float(np.nanpercentile(centered, 99))

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 2.1), constrained_layout=True)
    for index, axis in enumerate(axes.ravel()):
        image = axis.imshow(
            centered[index],
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=vmax,
            extent=extent,
            interpolation="nearest",
        )
        axis.set_title(f"{index + 1}. {names[index]}", fontsize=10, pad=2)
        if index >= 4:
            axis.set_xlabel(r"$\Delta x$ (km)")
        if index % 4 == 0:
            axis.set_ylabel(r"$\Delta y$ (km)")
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
        if index % 4 == 3:
            colorbar.set_label("Mean |IG|", fontsize=10, labelpad=2)

    fig.suptitle("Feature influence around the prediction location", fontsize=10)
    fig.savefig(OUTPUT_DIR / "top10_gridcell_ig_target_centered_feature_maps.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def signed_plot(
    signed: np.ndarray,
    names: np.ndarray,
    dx_km: float,
    dy_km: float,
) -> None:
    _, height, width = signed.shape
    extent = (0, width * dx_km, 0, height * dy_km)
    vmax = float(np.nanpercentile(np.abs(signed), 99))

    fig, axes = plt.subplots(2, 4, figsize=(10.4, 3.35))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.88, wspace=0.16, hspace=0.02)
    for index, axis in enumerate(axes.ravel()):
        image = axis.imshow(
            signed[index],
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
            interpolation="nearest",
        )
        axis.set_title(f"{index + 1}. {names[index]}", fontsize=11, pad=3)
        if index >= 4:
            axis.set_xlabel("x distance (km)")
        else:
            axis.set_xticklabels([])
        if index % 4 == 0:
            axis.set_ylabel("y distance (km)")
        else:
            axis.set_yticklabels([])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.014)
        if index % 4 == 3:
            colorbar.set_label("IG", fontsize=10, labelpad=2)

    fig.suptitle("Signed IG contribution of input features", fontsize=13, y=0.98)
    fig.savefig(OUTPUT_DIR / "top10_gridcell_ig_signed_contribution_maps.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def lag_plot(lag: np.ndarray, offsets: np.ndarray, names: np.ndarray) -> None:
    total_fraction = lag.sum(axis=1) / lag.sum()
    denominator = lag.sum(axis=0, keepdims=True)
    normalized = np.divide(lag, denominator, out=np.zeros_like(lag), where=denominator > 0)
    labels = [FEATURE_LABELS.get(name, name) for name in names]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.2, 2.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1, 1.65)},
    )
    axes[0].plot(offsets, total_fraction, color="#1F5A85", marker="o", linewidth=1.2)
    axes[0].set_xlabel("Temporal offset", fontsize=8)
    axes[0].set_ylabel("Fraction of total |IG|", fontsize=8)
    axes[0].set_title("Total attribution by lag", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].grid(axis="y", color="#D0D0D0", linewidth=0.4)

    image = axes[1].imshow(
        normalized,
        origin="lower",
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=float(np.nanpercentile(normalized, 99)),
    )
    axes[1].set_xticks(np.arange(len(labels)), labels)
    axes[1].set_yticks(np.arange(len(offsets)), offsets)
    axes[1].set_ylabel("Temporal offset", fontsize=8)
    axes[1].set_title("Feature-wise attribution across temporal offsets", fontsize=8)
    axes[1].tick_params(labelsize=7)
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(labelsize=7)
    fig.savefig(OUTPUT_DIR / "top10_gridcell_ig_time_lag_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with np.load(INPUT) as data:
        names = data["feature_names"].astype(str)
        centered = data["target_centered_feature_attribution"].astype(float)
        lag = data["lag_feature_attribution"].astype(float)
        signed = data["signed_target_feature_attribution"].astype(float)
        offsets = data["temporal_offsets"].astype(int)
        dx_km = float(data["dx_km"])
        dy_km = float(data["dy_km"])
        ranking = (
            np.nansum(np.abs(data["geographic_feature_attribution"]), axis=(1, 2))
            if "geographic_feature_attribution" in data.files
            else np.nansum(centered, axis=(1, 2))
        )

    order = np.argsort(ranking)[::-1]
    names = names[order]
    centered = centered[order]
    signed = signed[order]
    lag = lag[:, order]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_centered_plot(centered, names, dx_km, dy_km)
    signed_plot(signed, names, dx_km, dy_km)
    lag_plot(lag, offsets, names)
    print(f"Saved three IG figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
