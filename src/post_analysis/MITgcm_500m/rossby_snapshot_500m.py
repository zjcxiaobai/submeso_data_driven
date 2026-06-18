#!/usr/bin/env python3
"""Plot one MITgcm_500m surface Rossby-number snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_mitgcm  # noqa: E402
from src.utils.Rossby import DEFAULT_BETA, DEFAULT_F0, rossby_number  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_500m/verification/tutorial_reentrant_channel/run"
)
DEFAULT_ITERATION = 2419200
DEFAULT_OUTPUT = Path(__file__).resolve().parent / f"rossby_snapshot_500m_{DEFAULT_ITERATION:010d}.png"


def iteration_path(run_dir: Path, varname: str, iteration: int, suffix: str) -> Path:
    return run_dir / f"{varname}.{iteration:010d}.{suffix}"


def validate_iteration_files(
    run_dir: Path,
    iteration: int,
    varnames: tuple[str, ...] = ("U", "V"),
) -> None:
    missing: list[Path] = []
    for varname in varnames:
        for suffix in ("data", "meta"):
            path = iteration_path(run_dir, varname, int(iteration), suffix)
            if not path.exists():
                missing.append(path)

    if missing:
        sample = "\n".join(f"  {path}" for path in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n  ... {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing MITgcm U/V files:\n{sample}{extra}")


def one_level_grid(ds_grid: xr.Dataset, k_index: int) -> xr.Dataset:
    if k_index < 0 or k_index >= ds_grid.sizes.get("k", 0):
        raise IndexError(
            f"surface-index must be between 0 and {ds_grid.sizes.get('k', 0) - 1}; "
            f"got {k_index}"
        )

    ds_level = xr.Dataset(
        {
            "hFacW": ds_grid["hFacW"].isel(k=slice(k_index, k_index + 1)),
            "hFacS": ds_grid["hFacS"].isel(k=slice(k_index, k_index + 1)),
            "hFacC": ds_grid["hFacC"].isel(k=slice(k_index, k_index + 1)),
            "DXC": ds_grid["DXC"],
            "DYC": ds_grid["DYC"],
            "RAZ": ds_grid["RAZ"],
            "XC": ds_grid["XC"],
            "YC": ds_grid["YC"],
        }
    )

    for dim in ("k_l", "k_p1"):
        if dim in ds_grid.coords and dim not in ds_level.coords:
            ds_level = ds_level.assign_coords({dim: ds_grid.coords[dim]})
        elif dim in ds_grid and dim not in ds_level:
            ds_level = ds_level.assign_coords({dim: ds_grid[dim]})

    return ds_level


def load_surface_velocity(run_dir: Path, iteration: int, surface_index: int) -> xr.Dataset:
    u = read_mitgcm._ocean_var(str(run_dir), "U", int(iteration), ("k", "j", "i_g"))
    v = read_mitgcm._ocean_var(str(run_dir), "V", int(iteration), ("k", "j_g", "i"))
    return xr.Dataset(
        {
            "U": u.isel(k=slice(surface_index, surface_index + 1)),
            "V": v.isel(k=slice(surface_index, surface_index + 1)),
        }
    )


def load_surface_rossby(
    *,
    run_dir: Path,
    iteration: int,
    surface_index: int,
    f0: float,
    beta: float,
    y_ref: float | None,
    reference: str,
) -> xr.DataArray:
    ds_grid = read_mitgcm.read_mitgrid_staggered(str(run_dir))
    ds_surface_grid = one_level_grid(ds_grid, surface_index)
    ds_uv = load_surface_velocity(run_dir, iteration, surface_index)
    ro = rossby_number(
        ds_surface_grid,
        ds_uv,
        f0=f0,
        beta=beta,
        y_ref=y_ref,
        reference=reference,
        y_boundary="fill",
        output_name="surface_Ro",
    )
    if "k" in ro.dims:
        ro = ro.isel(k=0, drop=True)
    ro.attrs["iteration"] = int(iteration)
    return ro


def robust_symmetric_limits(values: np.ndarray, percentile: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite Rossby-number values found in this snapshot")
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    return -limit, limit


def image_extent_km(ro: xr.DataArray) -> tuple[float, float, float, float] | None:
    # The Rossby utility returns the C-grid vorticity array on zeta/corner-like
    # points, so the dimensions may not carry XC/YC directly. For plotting, use
    # array indices if physical coordinates are unavailable.
    return None


def plot_snapshot(
    *,
    run_dir: Path,
    output: Path,
    iteration: int,
    surface_index: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    percentile: float,
    f0: float,
    beta: float,
    y_ref: float | None,
    reference: str,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render the plot. Load or install matplotlib "
            "in the Python environment used to run this script."
        ) from exc

    validate_iteration_files(run_dir, iteration)
    ro = load_surface_rossby(
        run_dir=run_dir,
        iteration=iteration,
        surface_index=surface_index,
        f0=f0,
        beta=beta,
        y_ref=y_ref,
        reference=reference,
    )

    values = ro.values
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = robust_symmetric_limits(values, percentile)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(
        values,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label(ro.attrs.get("units", "1"))
    ax.set_xlabel("i index")
    ax.set_ylabel("j index")
    ax.set_title(f"MITgcm_500m surface Rossby number | iteration {iteration}")

    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    finite = values[np.isfinite(values)]
    print(f"Saved figure: {output}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Iteration: {iteration}", flush=True)
    print(f"Surface index: {surface_index}", flush=True)
    print(f"Finite samples: {finite.size}", flush=True)
    print(f"Color limits: {vmin:.6g}, {vmax:.6g}", flush=True)
    if finite.size:
        print(
            f"Ro min/mean/max: {np.nanmin(finite):.6g}, "
            f"{np.nanmean(finite):.6g}, {np.nanmax(finite):.6g}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a surface Rossby-number snapshot for MITgcm_500m."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--f0", type=float, default=DEFAULT_F0)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--y-ref", type=float, default=None)
    parser.add_argument("--reference", choices=("south", "north", "center"), default="south")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_snapshot(
        run_dir=args.run_dir.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        iteration=args.iteration,
        surface_index=args.surface_index,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        percentile=args.percentile,
        f0=args.f0,
        beta=args.beta,
        y_ref=args.y_ref,
        reference=args.reference,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
