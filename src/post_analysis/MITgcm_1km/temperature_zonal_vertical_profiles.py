#!/usr/bin/env python3
"""Plot MITgcm temperature vertical sections along x at selected y positions."""

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
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "temperature_zonal_vertical_profiles_1555200.png"
DEFAULT_Y_KM = (1000.0, 1250.0, 1500.0, 1750.0)


def iteration_path(run_dir: Path, varname: str, iteration: int, suffix: str) -> Path:
    return run_dir / f"{varname}.{iteration:010d}.{suffix}"


def validate_iteration_files(run_dir: Path, iteration: int, varname: str = "T") -> None:
    missing = []
    for suffix in ("data", "meta"):
        path = iteration_path(run_dir, varname, int(iteration), suffix)
        if not path.exists():
            missing.append(path)

    if missing:
        sample = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing MITgcm {varname} files:\n{sample}")


def load_temperature(run_dir: Path, iteration: int) -> xr.DataArray:
    temp = read_mitgcm._ocean_var(str(run_dir), "T", int(iteration), ("k", "j", "i"))
    temp.attrs.update(long_name="Potential temperature", units="degC")
    return temp


def nearest_y_index(ds_grid: xr.Dataset, y_km: float) -> tuple[int, float]:
    yc = ds_grid["YC"]
    y_m = y_km * 1000.0

    if "i" in yc.dims:
        y_line = yc.mean("i")
    else:
        y_line = yc

    idx = int(np.abs(y_line - y_m).argmin())
    return idx, float(y_line.isel(j=idx)) / 1000.0


def section_at_y(
    temp: xr.DataArray,
    ds_grid: xr.Dataset,
    y_km: float,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray, float]:
    j_index, actual_y_km = nearest_y_index(ds_grid, y_km)
    section = temp.isel(j=j_index).where(ds_grid["hFacC"].isel(j=j_index) > 0)

    x_km = ds_grid["XC"].isel(j=j_index).values / 1000.0
    depth_m = ds_grid["RC"].values
    return section, x_km, depth_m, actual_y_km


def finite_color_limits(sections: list[xr.DataArray]) -> tuple[float, float]:
    values = np.concatenate([section.values.ravel() for section in sections])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite temperature values found in selected sections")
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def plot_temperature_sections(
    *,
    run_dir: Path,
    output: Path,
    iteration: int,
    y_km: tuple[float, ...],
    cmap: str,
    vmin: float | None,
    vmax: float | None,
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
    temp = load_temperature(run_dir, iteration)

    section_data = [section_at_y(temp, ds_grid, y) for y in y_km]
    sections = [item[0] for item in section_data]
    if vmin is None or vmax is None:
        data_vmin, data_vmax = finite_color_limits(sections)
        if vmin is None:
            vmin = data_vmin
        if vmax is None:
            vmax = data_vmax

    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    mesh = None

    for ax, (section, x_km, depth_m, actual_y_km), target_y_km in zip(axes_flat, section_data, y_km):
        mesh = ax.pcolormesh(
            x_km,
            depth_m,
            section.values,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"target y = {target_y_km:g} km | nearest y = {actual_y_km:.1f} km")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("depth (m)")

    cbar = fig.colorbar(mesh, ax=axes_flat, orientation="vertical", shrink=0.92)
    cbar.set_label(temp.attrs.get("units", "degC"))
    fig.suptitle(f"Temperature zonal vertical sections | iteration {iteration}")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved figure: {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot temperature x-depth sections at selected y positions."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iteration", type=int, default=1555200)
    parser.add_argument("--y-km", type=float, nargs="+", default=list(DEFAULT_Y_KM))
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_temperature_sections(
        run_dir=args.run_dir.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        iteration=args.iteration,
        y_km=tuple(args.y_km),
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
