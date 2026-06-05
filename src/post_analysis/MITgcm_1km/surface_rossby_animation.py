#!/usr/bin/env python3
"""Animate 1 km MITgcm surface Rossby number from U/V snapshots."""

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
DEFAULT_OUT = Path(__file__).resolve().parent / "surface_rossby_1490400_1555200.mp4"


def iteration_path(run_dir: Path, varname: str, iteration: int, suffix: str) -> Path:
    return run_dir / f"{varname}.{iteration:010d}.{suffix}"


def validate_iteration_files(
    run_dir: Path,
    iterations: np.ndarray,
    varnames: tuple[str, ...] = ("U", "V"),
) -> None:
    missing: list[Path] = []
    for iteration in iterations:
        for varname in varnames:
            data_path = iteration_path(run_dir, varname, int(iteration), "data")
            meta_path = iteration_path(run_dir, varname, int(iteration), "meta")
            if not data_path.exists():
                missing.append(data_path)
            if not meta_path.exists():
                missing.append(meta_path)

    if missing:
        sample = "\n".join(f"  {path}" for path in missing[:20])
        extra = "" if len(missing) <= 20 else f"\n  ... {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing MITgcm U/V files:\n{sample}{extra}")


def surface_grid(ds_grid: xr.Dataset, surface_index: int) -> xr.Dataset:
    """Select the one-level grid variables needed by Rossby.rossby_number."""
    ds_surface = xr.Dataset(
        {
            "hFacW": ds_grid["hFacW"].isel(k=slice(surface_index, surface_index + 1)),
            "hFacS": ds_grid["hFacS"].isel(k=slice(surface_index, surface_index + 1)),
            "hFacC": ds_grid["hFacC"].isel(k=slice(surface_index, surface_index + 1)),
            "DXC": ds_grid["DXC"],
            "DYC": ds_grid["DYC"],
            "RAZ": ds_grid["RAZ"],
            "XC": ds_grid["XC"],
            "YC": ds_grid["YC"],
        }
    )

    for dim in ("k_l", "k_p1"):
        if dim in ds_grid.coords and dim not in ds_surface.coords:
            ds_surface = ds_surface.assign_coords({dim: ds_grid.coords[dim]})
        elif dim in ds_grid and dim not in ds_surface:
            ds_surface = ds_surface.assign_coords({dim: ds_grid[dim]})

    return ds_surface


def load_surface_velocity(run_dir: Path, iteration: int, surface_index: int) -> xr.Dataset:
    """Load U and V at one MITgcm iteration and keep one surface k level."""
    u = read_mitgcm._ocean_var(str(run_dir), "U", int(iteration), ("k", "j", "i_g"))
    v = read_mitgcm._ocean_var(str(run_dir), "V", int(iteration), ("k", "j_g", "i"))
    return xr.Dataset(
        {
            "U": u.isel(k=slice(surface_index, surface_index + 1)),
            "V": v.isel(k=slice(surface_index, surface_index + 1)),
        }
    )


def load_surface_rossby(
    run_dir: Path,
    ds_surface_grid: xr.Dataset,
    iteration: int,
    surface_index: int,
    f0: float,
    beta: float,
    y_ref: float | None,
    reference: str,
) -> xr.DataArray:
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
    ro = ro.isel(k=0, drop=True) if "k" in ro.dims else ro
    ro.attrs["iteration"] = int(iteration)
    return ro


def image_extent(ds_surface_grid: xr.Dataset) -> tuple[float, float, float, float] | None:
    if "XC" not in ds_surface_grid or "YC" not in ds_surface_grid:
        return None

    xc = ds_surface_grid["XC"].values
    yc = ds_surface_grid["YC"].values
    if not np.isfinite(xc).any() or not np.isfinite(yc).any():
        return None

    return (
        float(np.nanmin(xc)),
        float(np.nanmax(xc)),
        float(np.nanmin(yc)),
        float(np.nanmax(yc)),
    )


def choose_writer(output: Path, fps: int):
    from matplotlib.animation import FFMpegWriter, PillowWriter

    suffix = output.suffix.lower()
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    if suffix in {".mp4", ".m4v", ".mov"}:
        return FFMpegWriter(fps=fps, bitrate=2400)
    raise ValueError("Output must end with .mp4, .m4v, .mov, or .gif")


def make_animation(
    *,
    run_dir: Path,
    output: Path,
    start: int,
    end: int,
    interval: int,
    surface_index: int,
    fps: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
    f0: float,
    beta: float,
    y_ref: float | None,
    reference: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render the animation. Load or install "
            "matplotlib in the Python environment used to run this script."
        ) from exc

    iterations = np.arange(start, end + interval, interval, dtype=int)
    iterations = iterations[iterations <= end]
    if iterations.size == 0:
        raise ValueError("No iterations selected; check start/end/interval")

    validate_iteration_files(run_dir, iterations)

    ds_grid = read_mitgcm.read_mitgrid_staggered(str(run_dir))
    ds_surface_grid = surface_grid(ds_grid, surface_index)

    first_ro = load_surface_rossby(
        run_dir,
        ds_surface_grid,
        int(iterations[0]),
        surface_index,
        f0,
        beta,
        y_ref,
        reference,
    )
    first_values = first_ro.values
    if vmin is None or vmax is None:
        limit = float(np.nanpercentile(np.abs(first_values), 99.0))
        if not np.isfinite(limit) or limit == 0.0:
            limit = 1.0
        if vmin is None:
            vmin = -limit
        if vmax is None:
            vmax = limit

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    extent = image_extent(ds_surface_grid)
    image = ax.imshow(
        first_values,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(first_ro.attrs.get("units", "1"))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title = ax.set_title(f"Surface Rossby number | iteration {int(iterations[0])}")

    def update(frame_index: int):
        iteration = int(iterations[frame_index])
        ro = load_surface_rossby(
            run_dir,
            ds_surface_grid,
            iteration,
            surface_index,
            f0,
            beta,
            y_ref,
            reference,
        )
        image.set_data(ro.values)
        title.set_text(f"Surface Rossby number | iteration {iteration}")
        print(f"[{frame_index + 1}/{len(iterations)}] iteration {iteration}", flush=True)
        return image, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(iterations),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    animation.save(output, writer=choose_writer(output, fps), dpi=dpi)
    plt.close(fig)
    print(f"Saved animation: {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a surface-Rossby-number animation from MITgcm U/V snapshots."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", type=int, default=1490400)
    parser.add_argument("--end", type=int, default=1555200)
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--f0", type=float, default=DEFAULT_F0)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--y-ref", type=float, default=None)
    parser.add_argument("--reference", choices=("south", "north", "center"), default="south")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_animation(
        run_dir=args.run_dir.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        start=args.start,
        end=args.end,
        interval=args.interval,
        surface_index=args.surface_index,
        fps=args.fps,
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
        f0=args.f0,
        beta=args.beta,
        y_ref=args.y_ref,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
