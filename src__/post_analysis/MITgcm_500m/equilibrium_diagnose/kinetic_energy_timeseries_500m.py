#!/usr/bin/env python3
"""Incrementally diagnose MITgcm_500m kinetic energy from U/V snapshots."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from submeso_data_driven.utils import read_mitgcm  # noqa: E402
from submeso_data_driven.utils.speed import horizontal_speed  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "/scratch/jczhang/MITgcm_500m/verification/tutorial_reentrant_channel/run"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "kinetic_energy_timeseries_500m.csv"
ITERATION_RE = re.compile(r"^U\.(\d{10})\.meta$")
FIELDNAMES = (
    "iteration",
    "model_seconds",
    "model_days",
    "ke_mean_m2_s2",
    "ke_integral_m5_s2",
    "speed_mean_m_s",
    "speed_max_m_s",
    "finite_fraction",
    "u_mtime",
    "v_mtime",
    "status",
)


@dataclass(frozen=True)
class IterationFiles:
    iteration: int
    u_data: Path
    u_meta: Path
    v_data: Path
    v_meta: Path

    @property
    def u_mtime(self) -> float:
        return self.u_data.stat().st_mtime

    @property
    def v_mtime(self) -> float:
        return self.v_data.stat().st_mtime


def iteration_path(run_dir: Path, varname: str, iteration: int, suffix: str) -> Path:
    return run_dir / f"{varname}.{iteration:010d}.{suffix}"


def discover_iterations(run_dir: Path) -> list[IterationFiles]:
    found: list[IterationFiles] = []
    for u_meta in sorted(run_dir.glob("U.*.meta")):
        match = ITERATION_RE.match(u_meta.name)
        if not match:
            continue
        iteration = int(match.group(1))
        files = IterationFiles(
            iteration=iteration,
            u_data=iteration_path(run_dir, "U", iteration, "data"),
            u_meta=u_meta,
            v_data=iteration_path(run_dir, "V", iteration, "data"),
            v_meta=iteration_path(run_dir, "V", iteration, "meta"),
        )
        if all(path.exists() for path in (files.u_data, files.u_meta, files.v_data, files.v_meta)):
            found.append(files)
    return found


def read_existing(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as handle:
        return {int(row["iteration"]): row for row in csv.DictReader(handle)}


def needs_update(files: IterationFiles, existing: dict[int, dict[str, str]], force: bool) -> bool:
    if force or files.iteration not in existing:
        return True

    row = existing[files.iteration]
    try:
        old_u_mtime = float(row.get("u_mtime", "nan"))
        old_v_mtime = float(row.get("v_mtime", "nan"))
    except ValueError:
        return True

    # Recompute if old blown-up files were overwritten by a clean rerun.
    return not (
        math.isclose(old_u_mtime, files.u_mtime, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(old_v_mtime, files.v_mtime, rel_tol=0.0, abs_tol=1.0e-6)
    )


def load_velocity(run_dir: Path, iteration: int) -> xr.Dataset:
    u = read_mitgcm._ocean_var(str(run_dir), "U", iteration, ("k", "j", "i_g"))
    v = read_mitgcm._ocean_var(str(run_dir), "V", iteration, ("k", "j_g", "i"))
    return xr.Dataset({"U": u, "V": v})


def volume_weights(ds_grid: xr.Dataset) -> xr.DataArray:
    weights = ds_grid["hFacC"] * ds_grid["RAC"] * ds_grid["DRF"]
    weights = weights.rename("cell_volume")
    weights.attrs.update(long_name="wet cell volume", units="m3")
    return weights


def compute_ke_row(
    *,
    run_dir: Path,
    ds_grid: xr.Dataset,
    weights: xr.DataArray,
    files: IterationFiles,
    delta_t: float,
) -> dict[str, str]:
    ds_uv = load_velocity(run_dir, files.iteration)
    speed = horizontal_speed(
        ds_grid,
        ds_uv,
        x_boundary="periodic",
        y_boundary="fill",
        output_name="speed",
    )

    speed_values = speed.values
    finite_fraction = float(np.isfinite(speed_values).sum() / speed_values.size)
    speed_mean = float(speed.mean(skipna=True).item())
    speed_max = float(speed.max(skipna=True).item())

    ke = 0.5 * speed**2
    ke_integral = float((ke * weights).sum(skipna=True).item())
    wet_volume = float(weights.where(np.isfinite(speed)).sum(skipna=True).item())
    ke_mean = ke_integral / wet_volume if wet_volume > 0.0 else np.nan

    status = "ok" if np.isfinite(ke_mean) and finite_fraction > 0.0 else "nan"
    model_seconds = files.iteration * delta_t

    return {
        "iteration": str(files.iteration),
        "model_seconds": f"{model_seconds:.6g}",
        "model_days": f"{model_seconds / 86400.0:.8g}",
        "ke_mean_m2_s2": f"{ke_mean:.12g}",
        "ke_integral_m5_s2": f"{ke_integral:.12g}",
        "speed_mean_m_s": f"{speed_mean:.12g}",
        "speed_max_m_s": f"{speed_max:.12g}",
        "finite_fraction": f"{finite_fraction:.12g}",
        "u_mtime": f"{files.u_mtime:.9f}",
        "v_mtime": f"{files.v_mtime:.9f}",
        "status": status,
    }


def write_rows(path: Path, rows: dict[int, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for iteration in sorted(rows):
            writer.writerow({name: rows[iteration].get(name, "") for name in FIELDNAMES})
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally compute volume-mean horizontal KE from MITgcm_500m U/V files. "
            "Existing CSV rows are reused unless the U/V data file mtimes changed."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delta-t", type=float, default=30.0)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute all selected iterations.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List iterations that would be computed without reading velocity fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    iterations = discover_iterations(run_dir)
    if args.start is not None:
        iterations = [item for item in iterations if item.iteration >= args.start]
    if args.end is not None:
        iterations = [item for item in iterations if item.iteration <= args.end]

    existing = read_existing(output)
    todo = [item for item in iterations if needs_update(item, existing, args.force)]

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Output CSV: {output}", flush=True)
    print(f"Discovered complete U/V iterations: {len(iterations)}", flush=True)
    print(f"Rows already in CSV: {len(existing)}", flush=True)
    print(f"Iterations to compute/update: {len(todo)}", flush=True)

    if args.dry_run:
        for item in todo:
            print(item.iteration, flush=True)
        return

    if not todo:
        print("Nothing new to compute.", flush=True)
        return

    ds_grid = read_mitgcm.read_mitgrid_staggered(str(run_dir))
    weights = volume_weights(ds_grid)

    rows = dict(existing)
    for index, files in enumerate(todo, start=1):
        print(f"[{index}/{len(todo)}] iteration {files.iteration}", flush=True)
        try:
            row = compute_ke_row(
                run_dir=run_dir,
                ds_grid=ds_grid,
                weights=weights,
                files=files,
                delta_t=args.delta_t,
            )
        except Exception as exc:
            row = {
                "iteration": str(files.iteration),
                "model_seconds": f"{files.iteration * args.delta_t:.6g}",
                "model_days": f"{files.iteration * args.delta_t / 86400.0:.8g}",
                "ke_mean_m2_s2": "nan",
                "ke_integral_m5_s2": "nan",
                "speed_mean_m_s": "nan",
                "speed_max_m_s": "nan",
                "finite_fraction": "0",
                "u_mtime": f"{files.u_mtime:.9f}",
                "v_mtime": f"{files.v_mtime:.9f}",
                "status": f"error: {exc}",
            }
        rows[files.iteration] = row
        write_rows(output, rows)
        print(
            f"  status={row['status']} KE_mean={row['ke_mean_m2_s2']} "
            f"speed_max={row['speed_max_m_s']}",
            flush=True,
        )

    print(f"Updated: {output}", flush=True)


if __name__ == "__main__":
    main()
