"""xgcm Grid constructors for this repo's MITgcm C-grid convention."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import xarray as xr
from xgcm import Grid


MITGCM_C_GRID_COORDS: dict[str, dict[str, str]] = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
    "Z": {"center": "k", "left": "k_l", "right": "k_p1"},
}


def _available_coords(ds: xr.Dataset, coords: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, str]]:
    """Keep declared axis positions whose dimensions are present in the dataset."""
    available: dict[str, dict[str, str]] = {}
    for axis, positions in coords.items():
        present = {position: dim for position, dim in positions.items() if dim in ds.dims}
        if present:
            available[axis] = present
    return available


def _normalize_periodic(
    periodic: bool | Iterable[str],
    coords: Mapping[str, Mapping[str, str]],
) -> bool | list[str]:
    if isinstance(periodic, bool):
        return periodic
    periodic_axes = list(periodic)
    unknown = sorted(set(periodic_axes) - set(coords))
    if unknown:
        raise ValueError(f"Periodic axes {unknown} are not present in grid coords {sorted(coords)}")
    return periodic_axes


def make_grid(
    ds: xr.Dataset,
    *,
    periodic: bool | Iterable[str] = ("X",),
    coords: Mapping[str, Mapping[str, str]] = MITGCM_C_GRID_COORDS,
    boundary: str | Mapping[str, str] | None = None,
    fill_value: float | Mapping[str, float] | None = None,
    metrics: Mapping[tuple[str, ...], list[str]] | None = None,
) -> Grid:
    """
    Build an ``xgcm.Grid`` from a dataset using the local MITgcm convention.

    The expected dimensions are ``i/i_g`` for X, ``j/j_g`` for Y, and
    ``k/k_l/k_p1`` for Z. Axes are included when at least one declared
    position is present, so this works for full C-grid datasets and smaller
    slices used for plotting. Metadata autoparsing is disabled because these
    datasets use explicit dimension names.
    """
    grid_coords = _available_coords(ds, coords)
    if not grid_coords:
        raise ValueError("Dataset does not contain any recognized MITgcm C-grid dimensions")

    return Grid(
        ds,
        coords=grid_coords,
        periodic=_normalize_periodic(periodic, grid_coords),
        boundary=boundary,
        fill_value=fill_value,
        metrics=metrics,
        autoparse_metadata=False,
    )


def make_grid_periodic_x(ds: xr.Dataset) -> Grid:
    """Backward-compatible constructor for re-entrant-channel grids periodic in X."""
    return make_grid(ds, periodic=("X",))
