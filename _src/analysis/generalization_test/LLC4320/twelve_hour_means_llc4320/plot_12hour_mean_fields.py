#!/usr/bin/env python3
"""Plot LLC4320 12-hour mean surface fields and submesoscale heat flux."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import build_coarse_grid, make_grid  # noqa: E402

from save_12hour_means_and_coarse_llc4320 import read_static_grid  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "twelve_hour_means_llc4320" / "LLC4320_12hour_mean.nc"
DEFAULT_COARSE_INPUT = (
    SCRIPT_DIR / "twelve_hour_means_llc4320" / "LLC4320_12hour_mean_coarse_50km.nc"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "twelve_hour_means_llc4320" / "field_plots"

PANEL_SPECS = (
    ("U", "Surface U", "m s$^{-1}$", "RdBu_r", True),
    ("V", "Surface V", "m s$^{-1}$", "RdBu_r", True),
    ("W", "Surface W", "m s$^{-1}$", "RdBu_r", True),
    ("T", "Surface temperature", "$^{\\circ}$C", "inferno", False),
    ("S", "Surface salinity", "g kg$^{-1}$", "viridis", False),
    ("Eta", "Sea-surface height", "m", "RdBu_r", True),
    ("kpphbl", "KPP boundary-layer depth", "m", "cividis", False),
    ("wT_submeso", "Surface submesoscale heat flux", "W m$^{-2}$", "RdBu_r", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save one eight-panel PNG for each LLC4320 12-hour mean block."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coarse-input", type=Path, default=DEFAULT_COARSE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional zero-based time indices. Default: plot every saved block.",
    )
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def surface_grid(ds_grid: xr.Dataset, surface_index: int) -> xr.Dataset:
    return xr.Dataset(
        {
            "hFacW": ds_grid["hFacW"].isel(k=surface_index, drop=True),
            "hFacS": ds_grid["hFacS"].isel(k=surface_index, drop=True),
            "hFacC": ds_grid["hFacC"].isel(k=surface_index, drop=True),
            "DXC": ds_grid["DXC"],
            "DYC": ds_grid["DYC"],
            "RAZ": ds_grid["RAZ"],
            "XC": ds_grid["XC"],
            "YC": ds_grid["YC"],
        }
    )


def center_surface_velocity(
    ds_surface_grid: xr.Dataset,
    u: xr.DataArray,
    v: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    ds = xr.merge([ds_surface_grid, xr.Dataset({"U": u, "V": v})])
    grid = make_grid.make_grid_periodic_x(ds)
    u_center = grid.interp(u, "X", to="center").rename("U")
    v_center = grid.interp(v, "Y", to="center", boundary="fill").rename("V")
    wet = ds_surface_grid["hFacC"] > 0
    return u_center.where(wet), v_center.where(wet)


def robust_norm(
    values: np.ndarray,
    percentile: float,
    symmetric: bool,
) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    if symmetric:
        scale = float(np.nanpercentile(np.abs(finite), percentile))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        return TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)

    tail = 100.0 - percentile
    vmin = float(np.nanpercentile(finite, tail))
    vmax = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def format_timestamp(value: int) -> str:
    text = f"{int(value):010d}"
    return f"{text[:4]}-{text[4:6]}-{text[6:8]} {text[8:10]}:00"


def load_fields(
    ds: xr.Dataset,
    ds_coarse: xr.Dataset,
    ds_surface_grid: xr.Dataset,
    time_index: int,
    coarse_time_index: int,
    surface_index: int,
) -> dict[str, xr.DataArray]:
    u = ds["U"].isel(time=time_index, k=surface_index, drop=True).load()
    v = ds["V"].isel(time=time_index, k=surface_index, drop=True).load()
    u_center, v_center = center_surface_velocity(ds_surface_grid, u, v)

    wet = ds_surface_grid["hFacC"] > 0
    return {
        "U": u_center,
        "V": v_center,
        "W": ds["W"].isel(time=time_index, k_l=surface_index, drop=True).load().where(wet),
        "T": ds["T"].isel(time=time_index, k=surface_index, drop=True).load().where(wet),
        "S": ds["S"].isel(time=time_index, k=surface_index, drop=True).load().where(wet),
        "Eta": ds["Eta"].isel(time=time_index, drop=True).load().where(wet),
        "kpphbl": ds["kpphbl"].isel(time=time_index, drop=True).load().where(wet),
        "wT_submeso": ds_coarse["wT_submeso"]
        .isel(time=coarse_time_index, k=surface_index, drop=True)
        .load(),
    }


def plot_block(
    fields: dict[str, xr.DataArray],
    ds_surface_grid: xr.Dataset,
    ds_coarse_grid: xr.Dataset,
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

    xc = ds_surface_grid["XC"].values
    yc = ds_surface_grid["YC"].values
    xc_coarse = ds_coarse_grid["XC"].values
    yc_coarse = ds_coarse_grid["YC"].values
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), constrained_layout=True)

    for panel_label, (ax, spec) in enumerate(zip(axes.flat, PANEL_SPECS), start=1):
        name, title, units, cmap, symmetric = spec
        values = fields[name].values
        panel_xc = xc_coarse if name == "wT_submeso" else xc
        panel_yc = yc_coarse if name == "wT_submeso" else yc
        norm = robust_norm(values, percentile, symmetric)
        mesh = ax.pcolormesh(
            panel_xc,
            panel_yc,
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        colorbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
        colorbar.set_label(units)
        colorbar.ax.tick_params(length=2)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.text(
            0.01,
            0.98,
            chr(96 + panel_label),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        )

    fig.suptitle(
        f"LLC4320 12-hour mean | {format_timestamp(time_start)} to "
        f"{format_timestamp(time_end)}",
        fontsize=11,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    coarse_input_path = args.coarse_input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not 50.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (50, 100]")

    with (
        xr.open_dataset(input_path, engine="h5netcdf", decode_times=False) as ds,
        xr.open_dataset(coarse_input_path, engine="h5netcdf", decode_times=False) as ds_coarse,
    ):
        required = {"time_start", "time_end", "U", "V", "W", "T", "S", "Eta", "kpphbl"}
        missing = sorted(required - set(ds.variables))
        if missing:
            raise KeyError(f"{input_path} is missing variables: {missing}")
        if "wT_submeso" not in ds_coarse:
            raise KeyError(f"{coarse_input_path} is missing wT_submeso")

        source_dir = Path(str(ds.attrs["source_data_dir"]))
        source_file = source_dir / str(ds.attrs["source_first_file"])
        ds_grid = read_static_grid(source_file)
        ds_surface_grid = surface_grid(ds_grid, args.surface_index)
        box_j = int(ds_coarse.attrs["box_j"])
        box_i = int(ds_coarse.attrs["box_i"])
        ds_coarse_grid = build_coarse_grid.build_coarse_grid_staggered(
            ds_grid, box_j=box_j, box_i=box_i
        )
        coarse_time_lookup = {
            int(value): index
            for index, value in enumerate(ds_coarse["time_start"].values)
        }

        if args.indices is None:
            indices = list(range(ds.sizes["time"]))
        else:
            indices = args.indices
        invalid = [index for index in indices if index < 0 or index >= ds.sizes["time"]]
        if invalid:
            raise IndexError(f"Time indices outside 0..{ds.sizes['time'] - 1}: {invalid}")

        for number, index in enumerate(indices, start=1):
            time_start = int(ds["time_start"].isel(time=index).item())
            time_end = int(ds["time_end"].isel(time=index).item())
            if time_start not in coarse_time_lookup:
                raise KeyError(f"No coarse wT_submeso block with time_start={time_start}")
            coarse_time_index = coarse_time_lookup[time_start]
            output = output_dir / f"LLC4320_12hour_fields_{time_start}_{time_end}.png"
            if output.exists() and not args.overwrite:
                print(f"[{number}/{len(indices)}] skip existing {output.name}", flush=True)
                continue

            print(
                f"[{number}/{len(indices)}] plotting index={index} "
                f"{time_start}-{time_end}",
                flush=True,
            )
            fields = load_fields(
                ds, ds_coarse, ds_surface_grid, index, coarse_time_index, args.surface_index
            )
            plot_block(
                fields,
                ds_surface_grid,
                ds_coarse_grid,
                time_start,
                time_end,
                output,
                args.percentile,
                args.dpi,
            )
            print(f"    saved {output}", flush=True)


if __name__ == "__main__":
    main()
