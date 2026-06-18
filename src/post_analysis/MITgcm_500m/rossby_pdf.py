#!/usr/bin/env python3
"""Plot the probability density function of MITgcm Rossby number."""

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
    "/scratch/jczhang/MITgcm_1km/verification/tutorial_reentrant_channel/run"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "rossby_pdf.png"
DEFAULT_ITERATION = 1555380


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
            f"surface_index must be between 0 and {ds_grid.sizes.get('k', 0) - 1}; "
            f"got {k_index}. This is a vertical level index, not a model iteration."
        )

    ds_level = xr.Dataset(
        {
            "hFacW": ds_grid["hFacW"].isel(k=slice(k_index, k_index + 1)),
            "hFacS": ds_grid["hFacS"].isel(k=slice(k_index, k_index + 1)),
            "hFacC": ds_grid["hFacC"].isel(k=slice(k_index, k_index + 1)),
            "DXC": ds_grid["DXC"],
            "DYC": ds_grid["DYC"],
            "RAZ": ds_grid["RAZ"],
            "YC": ds_grid["YC"],
        }
    )

    for dim in ("k_l", "k_p1"):
        if dim in ds_grid.coords and dim not in ds_level.coords:
            ds_level = ds_level.assign_coords({dim: ds_grid.coords[dim]})
        elif dim in ds_grid and dim not in ds_level:
            ds_level = ds_level.assign_coords({dim: ds_grid[dim]})

    return ds_level


def load_velocity(
    run_dir: Path,
    iteration: int,
    k_index: int | None,
) -> xr.Dataset:
    u = read_mitgcm._ocean_var(str(run_dir), "U", int(iteration), ("k", "j", "i_g"))
    v = read_mitgcm._ocean_var(str(run_dir), "V", int(iteration), ("k", "j_g", "i"))

    if k_index is not None:
        u = u.isel(k=slice(k_index, k_index + 1))
        v = v.isel(k=slice(k_index, k_index + 1))

    return xr.Dataset({"U": u, "V": v})


def load_rossby_snapshot(
    *,
    run_dir: Path,
    iteration: int,
    surface_index: int,
    all_depths: bool,
    f0: float,
    beta: float,
    y_ref: float | None,
    reference: str,
) -> xr.DataArray:
    ds_grid = read_mitgcm.read_mitgrid_staggered(str(run_dir))
    if all_depths:
        ds_ro_grid = ds_grid
        k_index = None
    else:
        ds_ro_grid = one_level_grid(ds_grid, surface_index)
        k_index = surface_index

    ds_uv = load_velocity(run_dir, iteration, k_index)
    ro = rossby_number(
        ds_ro_grid,
        ds_uv,
        f0=f0,
        beta=beta,
        y_ref=y_ref,
        reference=reference,
        y_boundary="fill",
        output_name="Ro",
    )

    if not all_depths and "k" in ro.dims:
        ro = ro.isel(k=0, drop=True)
    ro.attrs["iteration"] = int(iteration)
    return ro


def finite_rossby_values(ro: xr.DataArray, clip: float | None) -> np.ndarray:
    values = ro.values.ravel()
    values = values[np.isfinite(values)]
    if clip is not None:
        values = values[np.abs(values) <= clip]
    if values.size == 0:
        iteration = ro.attrs.get("iteration", "unknown")
        raise ValueError(
            "No finite Rossby-number values found for iteration "
            f"{iteration}. Check that the corresponding U/V snapshots are finite; "
            "in the current run, outputs after 1555200 appear to contain NaNs."
        )
    return values


def robust_range(values: np.ndarray, percentile: float) -> tuple[float, float]:
    if percentile >= 100.0:
        return float(np.nanmin(values)), float(np.nanmax(values))

    tail = (100.0 - percentile) / 2.0
    lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def plot_rossby_pdf(
    *,
    run_dir: Path,
    output: Path,
    iteration: int,
    surface_index: int,
    all_depths: bool,
    bins: int,
    range_percentile: float,
    clip: float | None,
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
    ro = load_rossby_snapshot(
        run_dir=run_dir,
        iteration=iteration,
        surface_index=surface_index,
        all_depths=all_depths,
        f0=f0,
        beta=beta,
        y_ref=y_ref,
        reference=reference,
    )
    values = finite_rossby_values(ro, clip)
    hist_range = robust_range(values, range_percentile)

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.hist(values, bins=bins, range=hist_range, density=True, histtype="stepfilled", alpha=0.75)
    ax.axvline(0.0, color="k", linewidth=1.0, alpha=0.55)
    ax.set_xlabel("Rossby number")
    ax.set_ylabel("PDF")
    depth_label = "all depths" if all_depths else f"surface k={surface_index}"
    ax.set_title(f"Rossby number PDF | iteration {iteration} | {depth_label}")
    ax.grid(True, alpha=0.25)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved figure: {output}", flush=True)
    print(f"Samples used: {values.size}", flush=True)
    print(f"Histogram range: {hist_range[0]:.6g}, {hist_range[1]:.6g}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the PDF of Rossby number for one MITgcm U/V snapshot."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--all-depths", action="store_true")
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--range-percentile", type=float, default=99.5)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--f0", type=float, default=DEFAULT_F0)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--y-ref", type=float, default=None)
    parser.add_argument("--reference", choices=("south", "north", "center"), default="south")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_rossby_pdf(
        run_dir=args.run_dir.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        iteration=args.iteration,
        surface_index=args.surface_index,
        all_depths=args.all_depths,
        bins=args.bins,
        range_percentile=args.range_percentile,
        clip=args.clip,
        f0=args.f0,
        beta=args.beta,
        y_ref=args.y_ref,
        reference=args.reference,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
