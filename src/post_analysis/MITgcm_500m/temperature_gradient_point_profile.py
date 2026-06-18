#!/usr/bin/env python3
"""Plot averaged vertical profile of horizontal temperature gradient around one MITgcm point."""

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


DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_1km/verification/tutorial_reentrant_channel/run"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "temperature_gradient_profile_circle_x400_y1500_1555200.png"


def iteration_path(run_dir: Path, varname: str, iteration: int, suffix: str) -> Path:
    return run_dir / f"{varname}.{iteration:010d}.{suffix}"


def validate_iteration_files(run_dir: Path, iteration: int, varname: str = "T") -> None:
    missing: list[Path] = []
    for suffix in ("data", "meta"):
        path = iteration_path(run_dir, varname, int(iteration), suffix)
        if not path.exists():
            missing.append(path)

    if missing:
        sample = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing MITgcm {varname} files:\n{sample}")


def nearest_point(ds_grid: xr.Dataset, x_km: float, y_km: float) -> tuple[int, int, float, float]:
    x_m = x_km * 1000.0
    y_m = y_km * 1000.0
    dist2 = (ds_grid["XC"] - x_m) ** 2 + (ds_grid["YC"] - y_m) ** 2
    flat_index = int(np.nanargmin(dist2.values))
    j_index, i_index = np.unravel_index(flat_index, dist2.shape)
    actual_x_km = float(ds_grid["XC"].isel(j=j_index, i=i_index)) / 1000.0
    actual_y_km = float(ds_grid["YC"].isel(j=j_index, i=i_index)) / 1000.0
    return j_index, i_index, actual_x_km, actual_y_km


def circle_targets(x_km: float, y_km: float, radius_km: float, n_points: int) -> list[tuple[float, float]]:
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    if radius_km < 0:
        raise ValueError("radius_km must be non-negative")

    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    return [(x_km + radius_km * np.cos(angle), y_km + radius_km * np.sin(angle)) for angle in angles]


def nearest_circle_points(
    ds_grid: xr.Dataset,
    x_km: float,
    y_km: float,
    radius_km: float,
    n_points: int,
) -> list[tuple[int, int, float, float, float, float]]:
    points: list[tuple[int, int, float, float, float, float]] = []
    seen: set[tuple[int, int]] = set()

    for target_x_km, target_y_km in circle_targets(x_km, y_km, radius_km, n_points):
        j_index, i_index, actual_x_km, actual_y_km = nearest_point(ds_grid, target_x_km, target_y_km)
        key = (j_index, i_index)
        if key in seen:
            continue
        seen.add(key)
        points.append((j_index, i_index, actual_x_km, actual_y_km, target_x_km, target_y_km))

    if not points:
        raise ValueError("No grid points selected on the requested circle")
    return points


def averaged_temperature_gradient_profile(
    temp: xr.DataArray,
    ds_grid: xr.Dataset,
    points: list[tuple[int, int, float, float, float, float]],
) -> xr.DataArray:
    profiles = []
    for point_index, (j_index, i_index, *_coords) in enumerate(points):
        profile = centered_temperature_gradient_profile(temp, ds_grid, j_index, i_index)
        profiles.append(profile.rename("temperature_gradient").expand_dims(sample=[point_index]))

    stacked = xr.concat(profiles, dim="sample")
    mean_profile = stacked.mean("sample", skipna=True).rename("temperature_gradient_mean")
    mean_profile.attrs.update(
        long_name="mean horizontal temperature-gradient magnitude around circle",
        units="degC m^-1",
        sample_count=len(points),
    )
    return mean_profile


def centered_temperature_gradient_profile(
    temp: xr.DataArray,
    ds_grid: xr.Dataset,
    j_index: int,
    i_index: int,
) -> xr.DataArray:
    ni = temp.sizes["i"]
    nj = temp.sizes["j"]
    if j_index <= 0 or j_index >= nj - 1:
        raise ValueError("Selected y point is too close to a meridional boundary for centered dT/dy")

    i_w = (i_index - 1) % ni
    i_e = (i_index + 1) % ni
    j_s = j_index - 1
    j_n = j_index + 1

    t_w = temp.isel(j=j_index, i=i_w)
    t_e = temp.isel(j=j_index, i=i_e)
    t_s = temp.isel(j=j_s, i=i_index)
    t_n = temp.isel(j=j_n, i=i_index)

    x_w = float(ds_grid["XC"].isel(j=j_index, i=i_w))
    x_e = float(ds_grid["XC"].isel(j=j_index, i=i_e))
    y_s = float(ds_grid["YC"].isel(j=j_s, i=i_index))
    y_n = float(ds_grid["YC"].isel(j=j_n, i=i_index))

    domain_width = float(ds_grid["XC"].max() - ds_grid["XC"].min())
    dx = x_e - x_w
    if dx <= 0:
        dx += domain_width
    dy = y_n - y_s
    if dy == 0:
        raise ValueError("Cannot compute dT/dy because neighboring YC values are identical")

    dtdx = (t_e - t_w) / dx
    dtdy = (t_n - t_s) / dy
    grad = xr.apply_ufunc(np.hypot, dtdx, dtdy)

    wet = ds_grid["hFacC"].isel(j=j_index, i=i_index) > 0
    grad = grad.where(wet).rename("temperature_gradient")
    grad.attrs.update(
        long_name="horizontal temperature-gradient magnitude",
        units="degC m^-1",
        dx_m=dx,
        dy_m=dy,
    )
    return grad


def plot_temperature_gradient_profile(
    *,
    run_dir: Path,
    output: Path,
    iteration: int,
    x_km: float,
    y_km: float,
    radius_km: float,
    n_points: int,
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

    validate_iteration_files(run_dir, iteration, "T")
    ds_grid = read_mitgcm.read_mitgrid_staggered(str(run_dir))
    temp = read_mitgcm._ocean_var(str(run_dir), "T", int(iteration), ("k", "j", "i"))

    points = nearest_circle_points(ds_grid, x_km, y_km, radius_km, n_points)
    grad = averaged_temperature_gradient_profile(temp, ds_grid, points)
    depth_m = ds_grid["RC"].values

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 6.2), constrained_layout=True)
    ax.plot(grad.values, depth_m, marker="o", markersize=3, linewidth=1.4)
    ax.set_xlabel(r"$|\nabla_h T|$ (degC m$^{-1}$)")
    ax.set_ylabel("depth (m)")
    ax.set_title(
        "Mean temperature gradient profile on circle\n"
        f"iteration {iteration}, center ({x_km:g}, {y_km:g}) km\n"
        f"radius {radius_km:g} km, samples {len(points)}"
    )
    ax.grid(True, alpha=0.25)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved figure: {output}", flush=True)
    print(f"Requested samples: {n_points}; unique grid samples: {len(points)}", flush=True)
    for point_index, (j_index, i_index, actual_x_km, actual_y_km, target_x_km, target_y_km) in enumerate(points, start=1):
        print(
            f"sample {point_index:02d}: target=({target_x_km:.3f}, {target_y_km:.3f}) km, "
            f"nearest i={i_index}, j={j_index}, x={actual_x_km:.3f} km, y={actual_y_km:.3f} km",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged vertical profile of horizontal temperature gradient on a circle."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=1555200)
    parser.add_argument("--x-km", type=float, default=400.0)
    parser.add_argument("--y-km", type=float, default=1500.0)
    parser.add_argument("--radius-km", type=float, default=10.0)
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_temperature_gradient_profile(
        run_dir=args.run_dir.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        iteration=args.iteration,
        x_km=args.x_km,
        y_km=args.y_km,
        radius_km=args.radius_km,
        n_points=args.n_points,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
