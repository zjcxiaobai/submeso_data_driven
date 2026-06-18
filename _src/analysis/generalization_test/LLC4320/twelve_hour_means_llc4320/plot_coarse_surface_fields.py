#!/usr/bin/env python3
"""Plot eight LLC4320 coarse-grid surface fields for each 12-hour mean."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm


SCRIPT_DIR = Path(__file__).resolve().parent
LLC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
for path in (PROJECT_ROOT, LLC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils import build_coarse_grid, make_grid  # noqa: E402

from calculate_coarse_velocity_diagnostics import normalize_staggered_sizes  # noqa: E402
from save_12hour_means_and_coarse_llc4320 import read_static_grid  # noqa: E402


DEFAULT_INPUT = SCRIPT_DIR / "LLC4320_12hour_mean_coarse_50km.nc"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "coarse_surface_plots"

PANEL_SPECS = (
    ("U", "Coarse surface U", "m s$^{-1}$", "RdBu_r", True),
    ("V", "Coarse surface V", "m s$^{-1}$", "RdBu_r", True),
    ("W", "Coarse surface W", "m s$^{-1}$", "RdBu_r", True),
    ("T", "Coarse surface temperature", "$^{\\circ}$C", "inferno", False),
    ("S", "Coarse surface salinity", "g kg$^{-1}$", "viridis", False),
    ("Eta", "Coarse sea-surface height", "m", "RdBu_r", True),
    ("kpphbl", "Coarse KPP boundary-layer depth", "m", "cividis", False),
    (
        "wT_submeso",
        "Coarse surface submesoscale heat flux",
        "W m$^{-2}$",
        "RdBu_r",
        True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save one eight-panel coarse-surface PNG per 12-hour block."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional zero-based time indices. Default: plot every block.",
    )
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def robust_norm(values: np.ndarray, percentile: float, symmetric: bool):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(0.0, 1.0)
    if symmetric:
        scale = float(np.percentile(np.abs(finite), percentile))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.max(np.abs(finite)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)

    lower = 100.0 - percentile
    vmin = float(np.percentile(finite, lower))
    vmax = float(np.percentile(finite, percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def timestamp_label(timestamp: int) -> str:
    text = f"{int(timestamp):010d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:8]} {text[8:10]}:00"


def build_grid(source: xr.Dataset) -> xr.Dataset:
    source_dir = Path(str(source.attrs["source_data_dir"]))
    source_file = source_dir / str(source.attrs["source_first_file"])
    fine_grid = read_static_grid(source_file)
    coarse_grid = build_coarse_grid.build_coarse_grid_staggered(
        fine_grid,
        box_j=int(source.attrs["box_j"]),
        box_i=int(source.attrs["box_i"]),
    )
    return normalize_staggered_sizes(
        coarse_grid,
        center_j=int(source.sizes["j"]),
        center_i=int(source.sizes["i"]),
    )


def load_surface_fields(
    source: xr.Dataset,
    coarse_grid: xr.Dataset,
    time_index: int,
    surface_index: int,
) -> dict[str, xr.DataArray]:
    grid = make_grid.make_grid_periodic_x(coarse_grid)
    u = source["U"].isel(
        time=time_index,
        k=surface_index,
        i_g=slice(0, source.sizes["i"]),
        drop=True,
    ).load()
    v = source["V"].isel(
        time=time_index,
        k=surface_index,
        j_g=slice(0, source.sizes["j"]),
        drop=True,
    ).load()
    wet = coarse_grid["hFacC"].isel(k=surface_index, drop=True) > 0

    return {
        "U": grid.interp(u, "X", to="center").rename("U").where(wet),
        "V": grid.interp(v, "Y", to="center", boundary="fill").rename("V").where(wet),
        "W": source["W"]
        .isel(time=time_index, k_l=surface_index, drop=True)
        .load()
        .where(wet),
        "T": source["T"]
        .isel(time=time_index, k=surface_index, drop=True)
        .load()
        .where(wet),
        "S": source["S"]
        .isel(time=time_index, k=surface_index, drop=True)
        .load()
        .where(wet),
        "Eta": source["Eta"].isel(time=time_index, drop=True).load().where(wet),
        "kpphbl": source["kpphbl"].isel(time=time_index, drop=True).load().where(wet),
        "wT_submeso": source["wT_submeso"]
        .isel(time=time_index, k=surface_index, drop=True)
        .load()
        .where(wet),
    }


def plot_block(
    fields: dict[str, xr.DataArray],
    coarse_grid: xr.Dataset,
    time_start: int,
    time_end: int,
    output: Path,
    percentile: float,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    xc = coarse_grid["XC"].values
    yc = coarse_grid["YC"].values
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), constrained_layout=True)
    for panel_number, (ax, spec) in enumerate(zip(axes.flat, PANEL_SPECS), start=1):
        name, title, units, cmap, symmetric = spec
        values = fields[name].values
        mesh = ax.pcolormesh(
            xc,
            yc,
            values,
            shading="auto",
            cmap=cmap,
            norm=robust_norm(values, percentile, symmetric),
        )
        colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        colorbar.set_label(units)
        colorbar.ax.tick_params(length=2)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.text(
            0.02,
            0.97,
            chr(96 + panel_number),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )

    fig.suptitle(
        f"LLC4320 coarse 12-hour mean | {timestamp_label(time_start)} to "
        f"{timestamp_label(time_end)}",
        fontsize=11,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not 50.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (50, 100]")

    with xr.open_dataset(input_path, engine="h5netcdf", decode_times=False) as source:
        required = {"time_start", "time_end"} | {spec[0] for spec in PANEL_SPECS}
        missing = sorted(required - set(source.variables))
        if missing:
            raise KeyError(f"{input_path} is missing variables: {missing}")
        coarse_grid = build_grid(source)

        indices = (
            list(range(source.sizes["time"]))
            if args.indices is None
            else args.indices
        )
        invalid = [
            index for index in indices if index < 0 or index >= source.sizes["time"]
        ]
        if invalid:
            raise IndexError(
                f"Time indices outside 0..{source.sizes['time'] - 1}: {invalid}"
            )

        for number, index in enumerate(indices, start=1):
            time_start = int(source["time_start"].isel(time=index).item())
            time_end = int(source["time_end"].isel(time=index).item())
            output = output_dir / f"coarse_surface_fields_{time_start}_{time_end}.png"
            if output.exists() and not args.overwrite:
                print(f"[{number}/{len(indices)}] skip existing {output.name}", flush=True)
                continue

            print(f"[{number}/{len(indices)}] plotting {time_start}-{time_end}", flush=True)
            fields = load_surface_fields(
                source,
                coarse_grid,
                index,
                args.surface_index,
            )
            plot_block(
                fields,
                coarse_grid,
                time_start,
                time_end,
                output,
                args.percentile,
                args.dpi,
            )
            print(f"    saved {output}", flush=True)


if __name__ == "__main__":
    main()
