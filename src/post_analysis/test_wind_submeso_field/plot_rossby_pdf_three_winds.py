#!/usr/bin/env python3
"""Compare Rossby-number PDFs for constant, no-wind, and temporal-wind runs."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_mitgcm  # noqa: E402
from src.utils.Rossby import DEFAULT_BETA, DEFAULT_F0, rossby_number  # noqa: E402


START_ITERATION = 26701056
SECONDS_PER_STEP = 25
DAYS = (0, 7, 30, 60, 90)

RUNS = {
    "constant wind": Path(
        "/scratch/jczhang/MITgcm_baroclinic_instability_old/MITgcm/"
        "verification/tutorial_reentrant_channel/run"
    ),
    "wind off": Path("/scratch/jczhang/MITgcm_wind_off/verification/tutorial_reentrant_channel/run"),
    "temporal wind": Path(
        "/scratch/jczhang/MITgcm_wind_temporal/verification/tutorial_reentrant_channel/run"
    ),
}
GRID_RUN_DIR = RUNS["constant wind"]

COLORS = {
    "constant wind": "#2F6B9A",
    "wind off": "#C44E52",
    "temporal wind": "#4C956C",
}


@dataclass(frozen=True)
class Snapshot:
    label: str
    run_dir: Path
    day: int
    iteration: int
    values: np.ndarray


def iteration_for_day(day: int, start_iteration: int, seconds_per_step: int) -> int:
    steps_per_day = 86400 // seconds_per_step
    if 86400 % seconds_per_step != 0:
        raise ValueError("seconds_per_step must divide one day exactly")
    return int(start_iteration + day * steps_per_day)


def iteration_file(run_dir: Path, varname: str, iteration: int, suffix: str) -> Path:
    return run_dir / f"{varname}.{iteration:010d}.{suffix}"


def validate_snapshot_files(run_dir: Path, iteration: int) -> None:
    missing = [
        iteration_file(run_dir, varname, iteration, suffix)
        for varname in ("U", "V")
        for suffix in ("data", "meta")
        if not iteration_file(run_dir, varname, iteration, suffix).exists()
    ]
    if missing:
        sample = "\n".join(f"  {path}" for path in missing[:16])
        extra = "" if len(missing) <= 16 else f"\n  ... {len(missing) - 16} more"
        raise FileNotFoundError(f"Missing MITgcm U/V files:\n{sample}{extra}")


def one_level_grid(ds_grid: xr.Dataset, k_index: int) -> xr.Dataset:
    if k_index < 0 or k_index >= ds_grid.sizes.get("k", 0):
        raise IndexError(
            f"surface_index must be between 0 and {ds_grid.sizes.get('k', 0) - 1}; "
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
            "YC": ds_grid["YC"],
        }
    )
    for dim in ("k_l", "k_p1"):
        if dim in ds_grid.coords:
            ds_level = ds_level.assign_coords({dim: ds_grid.coords[dim]})
        elif dim in ds_grid:
            ds_level = ds_level.assign_coords({dim: ds_grid[dim]})
    return ds_level


def load_velocity(run_dir: Path, iteration: int, k_index: int | None) -> xr.Dataset:
    u = read_mitgcm._ocean_var(run_dir, "U", iteration, ("k", "j", "i_g"))
    v = read_mitgcm._ocean_var(run_dir, "V", iteration, ("k", "j_g", "i"))
    if k_index is not None:
        u = u.isel(k=slice(k_index, k_index + 1))
        v = v.isel(k=slice(k_index, k_index + 1))
    return xr.Dataset({"U": u, "V": v})


def finite_values(ro: xr.DataArray, clip: float | None) -> np.ndarray:
    values = np.asarray(ro.values).ravel()
    values = values[np.isfinite(values)]
    if clip is not None:
        values = values[np.abs(values) <= clip]
    if values.size == 0:
        raise ValueError("No finite Rossby-number values found after masking")
    return values


def load_snapshot(
    *,
    label: str,
    run_dir: Path,
    ds_grid: xr.Dataset,
    day: int,
    iteration: int,
    surface_index: int,
    all_depths: bool,
    clip: float | None,
    f0: float,
    beta: float,
    y_ref: float | None,
    reference: str,
) -> Snapshot:
    validate_snapshot_files(run_dir, iteration)
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
    return Snapshot(label, run_dir, day, iteration, finite_values(ro, clip))


def robust_range(values: np.ndarray, percentile: float) -> tuple[float, float]:
    if percentile >= 100.0:
        lo, hi = np.nanmin(values), np.nanmax(values)
    else:
        tail = (100.0 - percentile) / 2.0
        lo, hi = np.nanpercentile(values, [tail, 100.0 - tail])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        raise ValueError(f"Invalid histogram range: {lo}, {hi}")
    return float(lo), float(hi)


def write_stats_csv(snapshots: list[Snapshot], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case",
                "day",
                "iteration",
                "n",
                "mean",
                "std",
                "median",
                "p01",
                "p05",
                "p95",
                "p99",
            ]
        )
        for snapshot in snapshots:
            values = snapshot.values
            writer.writerow(
                [
                    snapshot.label,
                    snapshot.day,
                    snapshot.iteration,
                    values.size,
                    np.mean(values),
                    np.std(values),
                    np.median(values),
                    *np.percentile(values, [1, 5, 95, 99]),
                ]
            )


def plot_pdf_grid(
    *,
    snapshots: list[Snapshot],
    days: tuple[int, ...],
    output_stem: Path,
    bins: int,
    range_percentile: float,
    dpi: int,
    all_depths: bool,
    surface_index: int,
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    by_day = {day: [item for item in snapshots if item.day == day] for day in days}
    fig, axes = plt.subplots(len(days), 1, figsize=(6.8, 8.6), sharey=False, constrained_layout=True)
    if len(days) == 1:
        axes = [axes]

    for ax, day in zip(axes, days):
        day_snapshots = by_day[day]
        combined = np.concatenate([snapshot.values for snapshot in day_snapshots])
        hist_range = robust_range(combined, range_percentile)

        for snapshot in day_snapshots:
            ax.hist(
                snapshot.values,
                bins=bins,
                range=hist_range,
                density=True,
                histtype="step",
                linewidth=1.4,
                color=COLORS[snapshot.label],
                label=snapshot.label,
            )
        ax.axvline(0.0, color="0.2", linewidth=0.8, alpha=0.7)
        ax.set_title(f"t = {day} days  |  iteration {day_snapshots[0].iteration}", loc="left")
        ax.set_ylabel("PDF")
        ax.grid(True, color="0.88", linewidth=0.7)

    depth_label = "all depths" if all_depths else f"surface k={surface_index}"
    axes[-1].set_xlabel(f"Rossby number, Ro ({depth_label})")
    axes[0].legend(ncols=3, loc="upper right")
    fig.suptitle("Rossby-number PDFs under constant, off, and temporal wind forcing", y=1.002)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": dpi}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Rossby-number PDFs for constant-wind, wind-off, and temporal-wind "
            "MITgcm runs at 0, 7, 30, 60, and 90 days from the start iteration."
        )
    )
    parser.add_argument("--start-iteration", type=int, default=START_ITERATION)
    parser.add_argument("--seconds-per-step", type=int, default=SECONDS_PER_STEP)
    parser.add_argument("--days", type=int, nargs="+", default=list(DAYS))
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--all-depths", action="store_true")
    parser.add_argument("--bins", type=int, default=140)
    parser.add_argument("--range-percentile", type=float, default=99.5)
    parser.add_argument("--clip", type=float, default=None)
    parser.add_argument("--f0", type=float, default=DEFAULT_F0)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--y-ref", type=float, default=None)
    parser.add_argument("--reference", choices=("south", "north", "center"), default="south")
    parser.add_argument(
        "--grid-run-dir",
        type=Path,
        default=GRID_RUN_DIR,
        help="Run directory containing static MITgcm grid files. Defaults to the constant-wind run.",
    )
    parser.add_argument("--output-stem", type=Path, default=Path(__file__).resolve().parent / "rossby_pdf_three_winds")
    parser.add_argument("--stats-csv", type=Path, default=Path(__file__).resolve().parent / "rossby_pdf_three_winds_stats.csv")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    days = tuple(args.days)
    snapshots: list[Snapshot] = []
    ds_grid = read_mitgcm.read_mitgrid_staggered(args.grid_run_dir.expanduser().resolve())

    for day in days:
        iteration = iteration_for_day(day, args.start_iteration, args.seconds_per_step)
        for label, run_dir in RUNS.items():
            print(f"Loading {label}, day {day}, iteration {iteration}", flush=True)
            snapshots.append(
                load_snapshot(
                    label=label,
                    run_dir=run_dir,
                    ds_grid=ds_grid,
                    day=day,
                    iteration=iteration,
                    surface_index=args.surface_index,
                    all_depths=args.all_depths,
                    clip=args.clip,
                    f0=args.f0,
                    beta=args.beta,
                    y_ref=args.y_ref,
                    reference=args.reference,
                )
            )

    plot_pdf_grid(
        snapshots=snapshots,
        days=days,
        output_stem=args.output_stem.expanduser().resolve(),
        bins=args.bins,
        range_percentile=args.range_percentile,
        dpi=args.dpi,
        all_depths=args.all_depths,
        surface_index=args.surface_index,
    )
    write_stats_csv(snapshots, args.stats_csv.expanduser().resolve())
    print(f"Saved figure stem: {args.output_stem.expanduser().resolve()}", flush=True)
    print(f"Saved stats: {args.stats_csv.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
