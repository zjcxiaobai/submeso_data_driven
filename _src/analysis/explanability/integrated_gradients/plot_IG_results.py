#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot compact grid-cell IG figures from saved arrays.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DIR / "top10_gridcell_ig.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def feature_order(names: list[str], geographic: np.ndarray) -> np.ndarray:
    scores = np.nansum(np.abs(geographic), axis=(1, 2))
    return np.asarray(np.argsort(-scores, kind="stable"), dtype=int)


def feature_maps(
    values, names, output, title, cmap, signed=False, shared_scale=False,
    cbar_label=None, figsize=(10.4, 4.6),
):
    style()
    # Rank signed maps after cancellation, using magnitude rather than sign.
    order = feature_order(names, values)
    values = values[order]
    names = [names[index] for index in order]
    fig, axes = plt.subplots(2, 4, figsize=figsize, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.035, wspace=0.025, hspace=0.04)
    global_finite = np.abs(values[np.isfinite(values)])
    shared_vmax = float(np.percentile(global_finite, 99)) if global_finite.size else 1.0
    for index, ax in enumerate(axes.ravel()):
        image = values[index]
        finite = np.abs(image[np.isfinite(image)])
        vmax = shared_vmax if shared_scale else (float(np.percentile(finite, 99)) if finite.size else 1.0)
        mesh = ax.imshow(
            image, origin="lower", aspect="equal", cmap=cmap,
            vmin=-vmax if signed else 0, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(f"{index + 1}. {names[index]}", fontsize=8.4, pad=3)
        ax.set_xticks([0, 20])
        ax.set_yticks([0, 10])
        ax.tick_params(labelsize=7.4, length=2.2, width=0.6)
        if index >= 4:
            ax.set_xlabel("x index", fontsize=8.0)
        else:
            ax.set_xticklabels([])
        if index % 4 == 0:
            ax.set_ylabel("y index", fontsize=8.0)
        else:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.014)
        cbar.ax.tick_params(labelsize=6.8, length=2.0, width=0.55)
        cbar.outline.set_linewidth(0.55)
        if cbar_label is not None and index % 4 == 3:
            cbar.set_label(cbar_label, fontsize=7.0, labelpad=2)
    fig.suptitle(title, fontsize=10.0, y=1.02)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def radial_curves(centered, dx_km, dy_km):
    _, height, width = centered.shape
    dy = (np.arange(height) - height // 2) * dy_km
    dx = (np.arange(width) - width // 2) * dx_km
    distance = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    curves = []
    for image in centered:
        valid = np.isfinite(image) & (image >= 0)
        radii = distance[valid]
        weights = image[valid]
        order = np.argsort(radii)
        radii = radii[order]
        cumulative = np.cumsum(weights[order])
        curves.append((radii, cumulative / cumulative[-1]))
    return curves


def target_centered_with_distance(centered, names, curves, dx_km, dy_km, output):
    style()
    order = feature_order(names, centered)
    centered = centered[order]
    names = [names[index] for index in order]
    _, height, width = centered.shape
    x_offsets = (np.arange(width) - width // 2) * dx_km
    y_offsets = (np.arange(height) - height // 2) * dy_km
    extent = (
        x_offsets[0] - dx_km / 2, x_offsets[-1] + dx_km / 2,
        y_offsets[0] - dy_km / 2, y_offsets[-1] + dy_km / 2,
    )
    finite = centered[np.isfinite(centered)]
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 2.10), constrained_layout=True)
    for index, ax in enumerate(axes.ravel()):
        mesh = ax.imshow(
            centered[index], origin="lower", aspect="equal", cmap="magma",
            vmin=0, vmax=vmax, extent=extent, interpolation="nearest",
        )
        ax.set_title(f"{index + 1}. {names[index]}", fontsize=7, pad=2)
        if index >= 4:
            ax.set_xlabel(r"$\Delta x$ (km)")
        if index % 4 == 0:
            ax.set_ylabel(r"$\Delta y$ (km)")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
        if index % 4 == 3:
            cbar.set_label("Mean |IGs|", fontsize=6.5, labelpad=2)
    fig.suptitle("Feature influence around the prediction location", fontsize=9)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)

def lag_plot(lag_feature, offsets, names, output):
    style()
    total_fraction = lag_feature.sum(axis=1) / lag_feature.sum()
    denominator = lag_feature.sum(axis=0, keepdims=True)
    normalized = np.divide(lag_feature, denominator, out=np.zeros_like(lag_feature), where=denominator > 0)
    fig, axes = plt.subplots(
        1, 2, figsize=(6.2, 2.35), constrained_layout=True,
        gridspec_kw={"width_ratios": (1, 1.65)},
    )
    axes[0].plot(offsets, total_fraction, color="#1F5A85", marker="o", linewidth=1.2)
    axes[0].set_xlabel("Temporal offset")
    axes[0].set_ylabel("Fraction of total |IG|")
    axes[0].set_title("Total attribution by lag", fontsize=8)
    axes[0].set_ylim(0, max(0.05, float(np.nanmax(total_fraction)) * 1.15))
    axes[0].grid(axis="y", color="#D0D0D0", linewidth=0.4)
    mesh = axes[1].imshow(
        normalized, origin="lower", aspect="auto", cmap="Blues",
        vmin=0, vmax=float(np.nanpercentile(normalized, 99)),
    )
    axes[1].set_xticks(np.arange(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(offsets)))
    axes[1].set_yticklabels(offsets)
    axes[1].set_ylabel("Temporal offset")
    axes[1].set_title("Feature-wise attribution across temporal offsets", fontsize=8)
    fig.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.02)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def replot(input_path: Path, output_dir: Path) -> None:
    with np.load(input_path, allow_pickle=True) as data:
        names = [str(value) for value in data["feature_names"]]
        geographic = np.asarray(data["geographic_feature_attribution"], dtype=float)
        centered = np.asarray(data["target_centered_feature_attribution"], dtype=float)
        lag_feature = np.asarray(data["lag_feature_attribution"], dtype=float)
        signed = np.asarray(data["signed_target_feature_attribution"], dtype=float)
        offsets = np.asarray(data["temporal_offsets"], dtype=int)
        dx_km = float(data["dx_km"])
        dy_km = float(data["dy_km"])
    raw_names = names
    raw_geographic = geographic
    order = feature_order(names, geographic)
    scores = np.nansum(np.abs(raw_geographic), axis=(1, 2))
    names = [names[index] for index in order]
    geographic = geographic[order]
    centered = centered[order]
    signed = signed[order]
    lag_feature = lag_feature[:, order]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "top10_gridcell_ig"
    with (output_dir / "top10_gridcell_ig_geographical_feature_rank.csv").open("w") as handle:
        handle.write("rank,feature_index,feature_name,total_abs_ig,fraction_abs_ig\n")
        total_score = float(np.nansum(scores))
        for rank, feature_index in enumerate(order, start=1):
            score = float(scores[feature_index])
            fraction = score / total_score if total_score > 0 else float("nan")
            handle.write(f"{rank},{int(feature_index)},{raw_names[feature_index]},{score:.10g},{fraction:.10g}\n")
    feature_maps(
        geographic, names, prefix.with_name(prefix.name + "_geographical_feature_maps.png"),
        "|IGs| contribution of Features", "magma", shared_scale=True, cbar_label="Mean |IGs|",
    )
    feature_maps(
        signed, names, prefix.with_name(prefix.name + "_signed_contribution_maps.png"),
        "Signed |IGs| contribution of Features", "RdBu_r", signed=True, shared_scale=True,
        cbar_label="Mean signed IGs", figsize=(10.4, 4.4),
    )
    curves = radial_curves(centered, dx_km, dy_km)
    target_centered_with_distance(
        centered, names, curves, dx_km, dy_km,
        prefix.with_name(prefix.name + "_target_centered_feature_maps.png"),
    )
    lag_plot(
        lag_feature, offsets, names,
        prefix.with_name(prefix.name + "_time_lag_sensitivity.png"),
    )
    print(f"Replotted compact PNG figures under {output_dir}")


def main() -> None:
    args = parse_args()
    replot(args.input, args.output_dir)


if __name__ == "__main__":
    main()
