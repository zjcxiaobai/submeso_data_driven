"""xgcm Grid constructors for this repo's MITgcm C-grid convention."""

from __future__ import annotations

import xarray as xr
from xgcm import Grid


MITGCM_C_GRID_COORDS: dict[str, dict[str, str]] = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
    "Z": {"center": "k", "left": "k_l", "right": "k_p1"},
}

def make_grid(ds: xr.Dataset) -> Grid:
    """Construct the xgcm grid used by this re-entrant-channel simulation."""
    return Grid(
        ds,
        coords=MITGCM_C_GRID_COORDS,
        periodic=["X"],
        autoparse_metadata=False,
    )


def make_grid_periodic_x(ds: xr.Dataset) -> Grid:
    """Backward-compatible constructor for re-entrant-channel grids periodic in X."""
    return make_grid(ds)
