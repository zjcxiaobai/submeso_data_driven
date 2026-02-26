import xarray as xr
from xgcm import Grid

def make_grid_periodic_x(ds: xr.Dataset) -> Grid:
    coords = {
        "X": {"center": "i", "left": "i_g"},
        "Y": {"center": "j", "left": "j_g"},
        "Z": {"center": "k", "left": "k_l", "right": "k_p1"},
    }

    grid = Grid(
        ds,
        coords=coords,
        periodic=["X"],          # re-entrant channel periodic in X
        autoparse_metadata=False # avoid conflict
    )
    return grid