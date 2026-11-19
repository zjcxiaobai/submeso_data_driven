#!/usr/bin/env python3
"""Plot bathymetry plus direct MITgcm dynamics panels for the paper."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

from submeso_data_driven.utils import coarsen_UVWTS, read_mitgcm

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = PROJECT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from submeso_data_driven.utils import build_coarse_grid  # noqa: E402
from submeso_data_driven.utils.wT_submeso import submeso_vertical_heat_flux  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_BI/"
    "verification/tutorial_reentrant_channel/run"
)


@dataclass(frozen=True)
class CoarseDomain:
    name: str
    j_start: int
    j_stop: int
    i_start: int
    i_stop: int
    color: str


DOMAINS_56 = (
    CoarseDomain("analysis: upper 40x20", 30, 50, 10, 50, "#D95F02"),
    CoarseDomain("evaluation: lower 40x20", 10, 30, 10, 50, "#1B9E77"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a three-column paper figure from direct MITgcm output: "
            "left-column 3D bathymetry, northern-half temperature/submesoscale "
            "heat flux, and upper/lower 40x20 analysis domains."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--iteration", type=int, default=26687088)
    parser.add_argument("--k-index", type=int, default=2, help="Vertical center index plotted.")
    parser.add_argument("--box-j", type=int, default=10, help="Fine cells per coarse cell in y.")
    parser.add_argument("--box-i", type=int, default=10, help="Fine cells per coarse cell in x.")
    parser.add_argument("--bathy-stride", type=int, default=4, help="Subsample stride for the 3D bathymetry surface.")
    parser.add_argument(
        "--ml-j-origin",
        type=int,
        default=0,
        help=(
            "Coarse-grid j index of the 56x56 ML crop. Defaults to 0 because "
            "the train_on_small_domain metadata stores only indices within that crop."
        ),
    )
    parser.add_argument(
        "--ml-i-origin",
        type=int,
        default=0,
        help="Coarse-grid i index of the 56x56 ML crop.",
    )
    parser.add_argument(
        "--flux-abs-limit",
        type=float,
        default=None,
        help="Optional symmetric heat-flux color limit in W m-2.",
    )
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=99.0,
        help="Percentile used for robust color limits when explicit limits are omitted.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "dynamics_snapshot",
        help="Output prefix; .png and .json summary are written.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def finite_symmetric_limit(values: xr.DataArray | np.ndarray, percentile: float, fallback: float) -> float:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return fallback
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit <= 0:
        return fallback
    return limit


def finite_limits(values: xr.DataArray | np.ndarray, percentile: float) -> tuple[float, float]:
    arr = np.asarray(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    tail = (100.0 - percentile) / 2.0
    vmin = float(np.nanpercentile(finite, tail))
    vmax = float(np.nanpercentile(finite, 100.0 - tail))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.05, 1.0)
        return vmin - pad, vmax + pad
    return vmin, vmax


def km_extent_from_grid(x: xr.DataArray, y: xr.DataArray) -> tuple[float, float, float, float]:
    x_km = np.asarray(x) / 1000.0
    y_km = np.asarray(y) / 1000.0
    return float(np.nanmin(x_km)), float(np.nanmax(x_km)), float(np.nanmin(y_km)), float(np.nanmax(y_km))


def read_bathy_5km(run_dir: Path, grid: xr.Dataset) -> xr.DataArray:
    bathy_path = run_dir / "bathy.5km.bin"
    if bathy_path.exists():
        bathy = np.fromfile(bathy_path, dtype=">f4").reshape(grid.sizes["j"], grid.sizes["i"])
    else:
        bathy = -np.asarray(read_mitgcm._grid_var(run_dir, "Depth", ("j", "i")), dtype=np.float64)
    return xr.DataArray(
        bathy,
        dims=("j", "i"),
        coords={dim: grid.coords[dim] for dim in ("j", "i") if dim in grid.coords},
        name="bathy",
    )


def read_snapshot(run_dir: Path, iteration: int) -> tuple[xr.Dataset, xr.Dataset, xr.DataArray]:
    grid = read_mitgcm.read_mitgrid_staggered(run_dir)
    ocean = xr.Dataset(
        {
            "W": read_mitgcm._ocean_var(run_dir, "W", iteration, ("k_l", "j", "i")),
            "T": read_mitgcm._ocean_var(run_dir, "T", iteration, ("k", "j", "i")),
        }
    )
    ocean = ocean.assign_coords({dim: grid.coords[dim] for dim in ocean.dims if dim in grid.coords})
    bathy = read_bathy_5km(run_dir, grid)
    return grid, ocean, bathy


def select_k(da: xr.DataArray, k_index: int) -> xr.DataArray:
    if "k" not in da.dims:
        return da
    if not -da.sizes["k"] <= k_index < da.sizes["k"]:
        raise IndexError(f"--k-index {k_index} is outside valid range for {da.name}: {da.sizes['k']} levels")
    return da.isel(k=k_index)


def northern_half(da: xr.DataArray) -> xr.DataArray:
    j_dim = "j" if "j" in da.dims else "j_f"
    start = da.sizes[j_dim] // 2
    return da.isel({j_dim: slice(start, None)})


def build_coarse_ocean(grid: xr.Dataset, ocean: xr.Dataset, box_j: int, box_i: int) -> tuple[xr.Dataset, xr.Dataset]:
    coarse_grid = build_coarse_grid.build_coarse_grid_staggered(grid, box_j=box_j, box_i=box_i)
    coarse_fields = {
        name: coarsen_UVWTS.coarse_field(name, ocean[name], grid, box_j, box_i)
        for name in ("W", "T")
    }
    return coarse_grid, xr.Dataset(coarse_fields)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.02) -> None:
    text_kwargs = dict(
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, label, **text_kwargs)
    else:
        ax.text(x, y, label, **text_kwargs)


def plot_bathymetry_3d(
    ax: plt.Axes,
    grid: xr.Dataset,
    bathy_field: xr.DataArray,
    stride: int,
    box_extent_km: tuple[float, float, float, float] | None = None,
) -> None:
    if stride <= 0:
        raise ValueError("--bathy-stride must be positive")
    x = np.asarray(grid["XC"], dtype=np.float64) / 1000.0
    y = np.asarray(grid["YC"], dtype=np.float64) / 1000.0
    bathy = np.asarray(bathy_field, dtype=np.float64)
    bathy = np.where(np.isfinite(bathy) & (bathy < 0), bathy, np.nan)
    bathy = np.clip(bathy, -4000.0, 0.0)

    # Bathymetry-only version of the existing 5 km domain figure: keep the
    # bathy.5km.bin surface/aspect, omit wind, temperature, and sponge layers.
    j0 = 10
    surf = ax.plot_surface(
        x[j0::stride, ::stride],
        y[j0::stride, ::stride],
        bathy[j0::stride, ::stride],
        cmap="copper",
        edgecolor="none",
        linewidth=0,
        alpha=0.92,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    surf.set_zorder(1)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 6000)
    ax.set_zlim(-4000, 0)
    if box_extent_km is not None:
        x0, x1, y0, y1 = box_extent_km
        xs = [x0, x1, x1, x0, x0]
        ys = [y0, y0, y1, y1, y0]
        zs = [0.0] * len(xs)
        ax.plot(xs, ys, zs, color="black", linewidth=1.4, zorder=10)

    ax.set_xlabel("x (km)", fontsize=9, labelpad=8)
    ax.set_ylabel("y (km)", fontsize=9, labelpad=9)
    ax.set_zlabel("z (m)", fontsize=9, labelpad=7)
    ax.set_title("")
    ax.set_box_aspect((3, 5, 2.3))
    ax.view_init(elev=24, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.70, 0.70, 0.70, 0.8)
        axis._axinfo["grid"]["linewidth"] = 0.6


def plot_field(
    ax: plt.Axes,
    field: xr.DataArray,
    *,
    x: xr.DataArray,
    y: xr.DataArray,
    cmap: str,
    norm=None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> mpl.image.AxesImage:
    extent = km_extent_from_grid(x, y)
    image = ax.imshow(
        np.asarray(field),
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return image


def plot_domain_panel(
    ax: plt.Axes,
    coarse_grid: xr.Dataset,
    *,
    ml_j_origin: int,
    ml_i_origin: int,
) -> list[dict[str, object]]:
    land = np.asarray(coarse_grid["hFacC"].isel(k=0) <= 0)
    x = coarse_grid["XC"]
    y = coarse_grid["YC"]
    extent = km_extent_from_grid(x, y)
    ax.imshow(
        np.where(land, 0.86, 1.0),
        origin="lower",
        extent=extent,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )

    nx = coarse_grid.sizes["i"]
    ny = coarse_grid.sizes["j"]
    dx = (float(x.max()) - float(x.min())) / max(nx - 1, 1) / 1000.0
    dy = (float(y.max()) - float(y.min())) / max(ny - 1, 1) / 1000.0
    x0 = float(x.min()) / 1000.0
    y0 = float(y.min()) / 1000.0

    ml_rect = Rectangle(
        (x0 + ml_i_origin * dx, y0 + ml_j_origin * dy),
        56 * dx,
        56 * dy,
        fill=False,
        edgecolor="#4D4D4D",
        linewidth=1.0,
        linestyle=":",
    )
    ax.add_patch(ml_rect)
    ax.text(
        x0 + ml_i_origin * dx + 1.5 * dx,
        y0 + (ml_j_origin + 53) * dy,
        "56x56 crop",
        color="#333333",
        fontsize=6.5,
    )

    domain_summary: list[dict[str, object]] = []
    for domain in DOMAINS_56:
        j_start = ml_j_origin + domain.j_start
        i_start = ml_i_origin + domain.i_start
        width = (domain.i_stop - domain.i_start) * dx
        height = (domain.j_stop - domain.j_start) * dy
        rect = Rectangle(
            (x0 + i_start * dx, y0 + j_start * dy),
            width,
            height,
            fill=False,
            edgecolor=domain.color,
            linewidth=1.35,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + i_start * dx + 1.0 * dx,
            y0 + (j_start + domain.j_stop - domain.j_start - 2) * dy,
            domain.name,
            color=domain.color,
            fontsize=6.5,
            weight="bold",
        )
        item = asdict(domain)
        item.update(
            coarse_j_start=j_start,
            coarse_j_stop=ml_j_origin + domain.j_stop,
            coarse_i_start=i_start,
            coarse_i_stop=ml_i_origin + domain.i_stop,
        )
        domain_summary.append(item)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return domain_summary


def plot_heat_flux_domain_crops(
    axes: tuple[plt.Axes, plt.Axes],
    heat_flux: xr.DataArray,
    *,
    ml_j_origin: int,
    ml_i_origin: int,
    flux_limit: float,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for ax, domain in zip(axes, DOMAINS_56):
        j_start = ml_j_origin + domain.j_start
        j_stop = ml_j_origin + domain.j_stop
        i_start = ml_i_origin + domain.i_start
        i_stop = ml_i_origin + domain.i_stop
        crop = heat_flux.isel(j=slice(j_start, j_stop), i=slice(i_start, i_stop))
        ax.imshow(
            np.asarray(crop),
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-flux_limit, vmax=flux_limit),
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_axis_off()
        title = "Training subdomain" if "analysis" in domain.name else "Evaluation subdomain"
        ax.set_title(title, fontsize=9, pad=8)
        item = asdict(domain)
        item.update(
            source_panel="northern-half submesoscale heat flux",
            crop_j_start=j_start,
            crop_j_stop=j_stop,
            crop_i_start=i_start,
            crop_i_stop=i_stop,
            plotted_field="submesoscale heat flux",
        )
        summaries.append(item)
    return summaries


def add_heat_flux_domain_boxes(
    ax: plt.Axes,
    x: xr.DataArray,
    y: xr.DataArray,
    *,
    ml_j_origin: int,
    ml_i_origin: int,
) -> None:
    nx = x.sizes["i"]
    ny = y.sizes["j"]
    dx = (float(x.max()) - float(x.min())) / max(nx - 1, 1) / 1000.0
    dy = (float(y.max()) - float(y.min())) / max(ny - 1, 1) / 1000.0
    x0 = float(x.min()) / 1000.0
    y0 = float(y.min()) / 1000.0
    for domain in DOMAINS_56:
        j_start = ml_j_origin + domain.j_start
        i_start = ml_i_origin + domain.i_start
        width = (domain.i_stop - domain.i_start) * dx
        height = (domain.j_stop - domain.j_start) * dy
        rect = Rectangle(
            (x0 + i_start * dx, y0 + j_start * dy),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(rect)


def make_figure(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    grid, ocean, bathy = read_snapshot(run_dir, args.iteration)
    temp = select_k(ocean["T"], args.k_index)

    coarse_grid, coarse_ocean = build_coarse_ocean(grid, ocean[["W", "T"]], args.box_j, args.box_i)
    heat_flux = submeso_vertical_heat_flux(
        grid,
        coarse_grid,
        ocean[["W", "T"]],
        coarse_ocean,
        box_j=args.box_j,
        box_i=args.box_i,
        boundary_Z="fill",
        to_wm2=True,
        output="submeso_center",
    ).isel(k=args.k_index)

    temp_vmin, temp_vmax = finite_limits(northern_half(temp), args.robust_percentile)
    flux_limit = args.flux_abs_limit or finite_symmetric_limit(northern_half(heat_flux), args.robust_percentile, fallback=50.0)

    fig = plt.figure(figsize=(8.4, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.75, 1.0, 0.72), height_ratios=(1, 1))
    ax_bathy = fig.add_subplot(gs[:, 0], projection="3d")
    ax_temp = fig.add_subplot(gs[0, 1])
    ax_flux = fig.add_subplot(gs[1, 1])
    ax_upper_flux = fig.add_subplot(gs[0, 2])
    ax_lower_flux = fig.add_subplot(gs[1, 2])

    flux_north_j0 = coarse_grid.sizes["j"] // 2
    heat_flux_extent = km_extent_from_grid(
        coarse_grid["XC"].isel(j=slice(flux_north_j0, None)),
        coarse_grid["YC"].isel(j=slice(flux_north_j0, None)),
    )
    box_extent_km = (heat_flux_extent[0], heat_flux_extent[1], heat_flux_extent[2], heat_flux_extent[3])
    plot_bathymetry_3d(ax_bathy, grid, bathy, args.bathy_stride, box_extent_km=box_extent_km)
    add_panel_label(ax_bathy, "a", x=0.02, y=0.86)
    ax_bathy.text2D(
        0.09,
        0.86,
        "SO reentrant channel",
        transform=ax_bathy.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    temp_north = northern_half(temp)
    temp_image = plot_field(
        ax_temp,
        temp_north,
        x=grid["XC"].isel(j=slice(grid.sizes["j"] // 2, None)),
        y=grid["YC"].isel(j=slice(grid.sizes["j"] // 2, None)),
        cmap="magma",
        vmin=temp_vmin,
        vmax=temp_vmax,
    )
    ax_temp.set_title("Potential temperature", fontsize=9)
    add_panel_label(ax_temp, "b", x=-0.14, y=1.02)
    cbar = fig.colorbar(temp_image, ax=ax_temp, fraction=0.045, pad=0.02)
    cbar.set_label("$^\\circ$C")

    flux_north = northern_half(heat_flux)
    flux_image = plot_field(
        ax_flux,
        flux_north,
        x=coarse_grid["XC"].isel(j=slice(coarse_grid.sizes["j"] // 2, None)),
        y=coarse_grid["YC"].isel(j=slice(coarse_grid.sizes["j"] // 2, None)),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-flux_limit, vmax=flux_limit),
    )
    ax_flux.set_title("Submesoscale heat flux", fontsize=9)
    add_heat_flux_domain_boxes(
        ax_flux,
        coarse_grid["XC"].isel(j=slice(coarse_grid.sizes["j"] // 2, None)),
        coarse_grid["YC"].isel(j=slice(coarse_grid.sizes["j"] // 2, None)),
        ml_j_origin=args.ml_j_origin,
        ml_i_origin=args.ml_i_origin,
    )
    add_panel_label(ax_flux, "c", x=-0.14, y=1.02)
    cbar = fig.colorbar(flux_image, ax=ax_flux, fraction=0.045, pad=0.02)
    cbar.set_label("W m$^{-2}$")

    domain_summary = plot_heat_flux_domain_crops(
        (ax_upper_flux, ax_lower_flux),
        flux_north,
        ml_j_origin=args.ml_j_origin,
        ml_i_origin=args.ml_i_origin,
        flux_limit=flux_limit,
    )
    add_panel_label(ax_upper_flux, "d", x=-0.16, y=1.02)
    add_panel_label(ax_lower_flux, "e", x=-0.16, y=1.02)

    png_output = output_prefix.with_suffix(".png")
    fig.savefig(png_output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    bathy_values = np.asarray(bathy)
    wet = np.isfinite(bathy_values) & (bathy_values < 0)
    summary = {
        "run_dir": str(run_dir),
        "iteration": args.iteration,
        "k_index": args.k_index,
        "box_j": args.box_j,
        "box_i": args.box_i,
        "bathy_stride": args.bathy_stride,
        "fine_grid_shape_j_i": [int(grid.sizes["j"]), int(grid.sizes["i"])],
        "coarse_grid_shape_j_i": [int(coarse_grid.sizes["j"]), int(coarse_grid.sizes["i"])],
        "physical_extent_km": {
            "x_min": km_extent_from_grid(grid["XC"], grid["YC"])[0],
            "x_max": km_extent_from_grid(grid["XC"], grid["YC"])[1],
            "y_min": km_extent_from_grid(grid["XC"], grid["YC"])[2],
            "y_max": km_extent_from_grid(grid["XC"], grid["YC"])[3],
        },
        "depth_m": {
            "min_positive": float(np.nanmin(-bathy_values[wet])),
            "max": float(np.nanmax(-bathy_values[wet])),
        },
        "color_limits": {
            "temperature": [temp_vmin, temp_vmax],
            "heat_flux_symmetric_wm2": flux_limit,
        },
        "bathymetry_box_z0_km": {
            "x_min": box_extent_km[0],
            "x_max": box_extent_km[1],
            "y_min": box_extent_km[2],
            "y_max": box_extent_km[3],
            "source": "middle-column northern-half submesoscale heat-flux panel",
        },
        "domains": domain_summary,
        "outputs": [str(png_output)],
    }
    with output_prefix.with_suffix(".json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    apply_style()
    summary = make_figure(args)
    for output in summary["outputs"]:
        print(f"Saved {output}")
    print(f"Saved {Path(summary['outputs'][0]).with_suffix('.json')}")


if __name__ == "__main__":
    main()
