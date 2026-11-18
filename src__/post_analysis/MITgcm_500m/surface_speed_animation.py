#!/usr/bin/env python3
"""Animate 1 km MITgcm surface speed from U/V snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from submeso_data_driven.utils import read_mitgcm  # noqa: E402
from submeso_data_driven.utils.speed import horizontal_speed  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_1km/verification/tutorial_reentrant_channel/run"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "surface_speed_1490400_1555200.mp4"


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
    """Select the horizontal grid variables needed by speed.horizontal_speed."""
    return xr.Dataset(
        {
            "hFacW": ds_grid["hFacW"].isel(k=surface_index, drop=True),
            "hFacS": ds_grid["hFacS"].isel(k=surface_index, drop=True),
            "hFacC": ds_grid["hFacC"].isel(k=surface_index, drop=True),
            "DYG": ds_grid["DYG"],
            "DXG": ds_grid["DXG"],
            "XC": ds_grid["XC"],
            "YC": ds_grid["YC"],
        }
    )


def load_surface_velocity(run_dir: Path, iteration: int, surface_index: int) -> xr.Dataset:
    """Load U and V at one MITgcm iteration and select the surface level."""
    u = read_mitgcm._ocean_var(str(run_dir), "U", int(iteration), ("k", "j", "i_g"))
    v = read_mitgcm._ocean_var(str(run_dir), "V", int(iteration), ("k", "j_g", "i"))
    return xr.Dataset(
        {
            "U": u.isel(k=surface_index, drop=True),
            "V": v.isel(k=surface_index, drop=True),
        }
    )


def load_surface_speed(
    run_dir: Path,
    ds_surface_grid: xr.Dataset,
    iteration: int,
    surface_index: int,
) -> xr.DataArray:
    ds_uv = load_surface_velocity(run_dir, iteration, surface_index)
    speed = horizontal_speed(
        ds_surface_grid,
        ds_uv,
        x_boundary="periodic",
        y_boundary="fill",
        output_name="surface_speed",
    )
    speed.attrs["iteration"] = int(iteration)
    return speed


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

    first_speed = load_surface_speed(run_dir, ds_surface_grid, int(iterations[0]), surface_index)
    first_values = first_speed.values
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(first_values, 99.0))

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
    cbar.set_label(first_speed.attrs.get("units", "m s^-1"))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    title = ax.set_title(f"Surface speed | iteration {int(iterations[0])}")

    def update(frame_index: int):
        iteration = int(iterations[frame_index])
        speed = load_surface_speed(run_dir, ds_surface_grid, iteration, surface_index)
        image.set_data(speed.values)
        title.set_text(f"Surface speed | iteration {iteration}")
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
        description="Create a surface-speed animation from MITgcm U/V snapshots."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", type=int, default=1490400)
    parser.add_argument("--end", type=int, default=1555200)
    parser.add_argument("--interval", type=int, default=90)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=140)
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
    )


if __name__ == "__main__":
    main()
