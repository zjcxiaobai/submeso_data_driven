"""Create the resumable 12-hour submesoscale heat-flux field."""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from coarse_fields import (
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_STEP,
    GRID_DIR,
    OUTPUT_DIR,
    load_field,
    write_time_blocks,
)
from utils import build_coarse_grid, make_grid, read_mitgcm
from utils.coarsen_UVWTS import coarse_field


RHO0 = 1035.0
CP = 3994.0


def _ordinal(field: xr.DataArray) -> xr.DataArray:
    field = field.reset_coords(drop=True)
    return field.assign_coords(
        {
            dim: np.arange(field.sizes[dim], dtype=np.int64)
            for dim in field.dims
        }
    )


def _interface_area(grid: xr.Dataset, w: xr.DataArray) -> xr.DataArray:
    upper = grid["hFacC"]
    lower = upper.shift(k=-1, fill_value=0.0)
    open_fraction = xr.apply_ufunc(np.minimum, upper, lower).rename(k="k_l")
    if "k_l" in w.coords:
        open_fraction = open_fraction.assign_coords(k_l=w.coords["k_l"])
    return open_fraction * grid["RAC"]


def _area_mean(
    values: xr.DataArray,
    area: xr.DataArray,
    box_j: int,
    box_i: int,
) -> xr.DataArray:
    numerator = (
        values.where(area > 0, 0.0).fillna(0.0) * area
    ).coarsen(j=box_j, i=box_i, boundary="trim").sum()
    denominator = area.coarsen(
        j=box_j,
        i=box_i,
        boundary="trim",
    ).sum()
    return xr.where(denominator > 0, numerator / denominator, np.nan)


def process_wt(
    *,
    grid_dir: Path = GRID_DIR,
    output_path: Path = OUTPUT_DIR / "wT_sub_12h_blocks_true.nc",
    start: int = DEFAULT_START,
    end: int = DEFAULT_END,
    step: int = DEFAULT_STEP,
    block_size: int = 12,
    box_j: int = 10,
    box_i: int = 10,
) -> None:
    """Compute the submesoscale flux from 12-hour mean W and T fields."""
    fine_grid = read_mitgcm.read_mitgrid_staggered(grid_dir)
    coarse_grid = build_coarse_grid.build_coarse_grid_staggered(
        fine_grid,
        box_j,
        box_i,
    )
    fine_xgcm = make_grid.make_grid_periodic_x(fine_grid)
    coarse_xgcm = make_grid.make_grid_periodic_x(coarse_grid)

    def compute_block(iterations: list[int]) -> xr.DataArray:
        first_w = load_field(grid_dir, "W", iterations[0])
        first_t = load_field(grid_dir, "T", iterations[0])
        sum_w = first_w.values.astype(np.float32, copy=True)
        sum_t = first_t.values.astype(np.float32, copy=True)

        for iteration in iterations[1:]:
            sum_w += load_field(grid_dir, "W", iteration).values.astype(
                np.float32,
                copy=False,
            )
            sum_t += load_field(grid_dir, "T", iteration).values.astype(
                np.float32,
                copy=False,
            )

        mean_w = first_w.copy(data=sum_w / len(iterations)).rename("W")
        mean_t = first_t.copy(data=sum_t / len(iterations)).rename("T")

        temperature_at_w = fine_xgcm.interp(
            mean_t,
            "Z",
            to="left",
            boundary="fill",
        )
        total_flux = mean_w * temperature_at_w
        area = _interface_area(fine_grid, mean_w)
        coarse_total = _ordinal(
            _area_mean(total_flux, area, box_j, box_i)
        )

        coarse_w = coarse_field("W", mean_w, fine_grid, box_j, box_i)
        coarse_t = coarse_field("T", mean_t, fine_grid, box_j, box_i)
        coarse_t_at_w = coarse_xgcm.interp(
            coarse_t,
            "Z",
            to="left",
            boundary="fill",
        )

        resolved_flux = coarse_w * coarse_t_at_w
        submesoscale_flux = coarse_total - resolved_flux
        submesoscale_flux = coarse_xgcm.interp(
            submesoscale_flux,
            "Z",
            to="center",
            boundary="fill",
        )
        submesoscale_flux = (RHO0 * CP * submesoscale_flux).rename("wT_sub_kC")
        submesoscale_flux.attrs.update(
            long_name="submesoscale vertical heat flux at tracer centers",
            units="W m^-2",
        )
        return submesoscale_flux

    write_time_blocks(
        compute_block,
        output_path,
        "wT_sub_kC",
        start,
        end,
        step=step,
        block_size=block_size,
        box_j=box_j,
        box_i=box_i,
        attributes={
            "rho0": RHO0,
            "cp": CP,
            "total_flux_definition": (
                "mean(W,12h) * interp_Zleft(mean(T,12h))"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=GRID_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "wT_sub_12h_blocks_true.nc",
    )
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--block-size", type=int, default=12)
    parser.add_argument("--box-j", type=int, default=10)
    parser.add_argument("--box-i", type=int, default=10)
    args = parser.parse_args()

    process_wt(
        grid_dir=args.grid_dir,
        output_path=args.output,
        start=args.start,
        end=args.end,
        step=args.step,
        block_size=args.block_size,
        box_j=args.box_j,
        box_i=args.box_i,
    )


if __name__ == "__main__":
    main()
