"""Rossby number utilities for MITgcm C-grid output."""

import numpy as np
import xarray as xr

from src.utils.vorticity import vorticity


DEFAULT_F0 = -1.363e-4  # s^-1
DEFAULT_BETA = 1.313e-11  # s^-1 m^-1


def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variable(s): {missing}")


def _center_wet_mask(ds_grid: xr.Dataset, hfac_c_name: str = "hFacC") -> xr.DataArray:
    wet = ds_grid[hfac_c_name] > 0
    if set(wet.dims) - {"j", "i"}:
        wet = wet.isel({dim: 0 for dim in wet.dims if dim not in {"j", "i"}})
    return wet


def infer_beta_plane_y_ref(
    ds_grid: xr.Dataset,
    yc_name: str = "YC",
    hfac_c_name: str = "hFacC",
    reference: str = "south",
) -> float:
    """
    Infer a beta-plane reference y-coordinate from wet tracer cells.

    Parameters
    ----------
    ds_grid
        Grid dataset containing ``YC`` and ``hFacC``.
    reference
        ``"south"`` uses the minimum wet YC, ``"north"`` uses the maximum wet
        YC, and ``"center"`` uses the wet-domain mean YC.
    """
    _require_variables(ds_grid, (yc_name, hfac_c_name))
    wet2d = _center_wet_mask(ds_grid, hfac_c_name)
    yc = ds_grid[yc_name]

    if reference == "south":
        y_ref = yc.where(wet2d).min()
    elif reference == "north":
        y_ref = yc.where(wet2d).max()
    elif reference == "center":
        y_ref = yc.where(wet2d).mean()
    else:
        raise ValueError("reference must be 'south', 'north', or 'center'")

    return float(y_ref)


def beta_plane_f(
    ds_grid: xr.Dataset,
    f0: float = DEFAULT_F0,
    beta: float = DEFAULT_BETA,
    y_ref: float | None = None,
    reference: str = "south",
    yc_name: str = "YC",
    hfac_c_name: str = "hFacC",
    output_name: str = "f",
) -> xr.DataArray:
    """
    Coriolis parameter on tracer centers for a beta plane.

    ``f = f0 + beta * (YC - y_ref)``. If ``y_ref`` is omitted, it is inferred
    from the southernmost wet tracer cell by default, appropriate for a
    Southern Ocean reentrant channel with y increasing northward.
    """
    _require_variables(ds_grid, (yc_name, hfac_c_name))
    wet2d = _center_wet_mask(ds_grid, hfac_c_name)
    y_ref_value = infer_beta_plane_y_ref(ds_grid, yc_name, hfac_c_name, reference) if y_ref is None else y_ref

    f = (f0 + beta * (ds_grid[yc_name] - y_ref_value)).where(wet2d).rename(output_name)
    f.attrs.update(
        long_name="Coriolis parameter on beta plane",
        units="s^-1",
        definition="f = f0 + beta * (YC - y_ref)",
        f0=f0,
        beta=beta,
        y_ref=y_ref_value,
        reference=reference if y_ref is None else "user_supplied",
    )
    return f


def rossby_number(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    f0: float = DEFAULT_F0,
    beta: float = DEFAULT_BETA,
    y_ref: float | None = None,
    reference: str = "south",
    yc_name: str = "YC",
    hfac_c_name: str = "hFacC",
    zeta_name: str = "zeta",
    output_name: str = "Ro",
    min_abs_f: float = 1e-12,
    return_components: bool = False,
    **vorticity_kwargs,
) -> xr.DataArray | xr.Dataset:
    """
    Rossby number from C-grid velocity: ``Ro = zeta / f``.

    This function first calls ``vorticity(ds_grid, ds_ocean, ...)`` and then
    divides by the beta-plane Coriolis parameter at tracer-cell centers.
    Extra keyword arguments are passed through to ``vorticity``.
    """
    zeta = vorticity(ds_grid, ds_ocean, output_name=zeta_name, **vorticity_kwargs)
    f = beta_plane_f(
        ds_grid,
        f0=f0,
        beta=beta,
        y_ref=y_ref,
        reference=reference,
        yc_name=yc_name,
        hfac_c_name=hfac_c_name,
        output_name="f",
    )
    _, f_on_zeta = xr.broadcast(zeta, f)

    ro = (zeta / f_on_zeta).where(np.abs(f_on_zeta) > min_abs_f).rename(output_name)
    ro.attrs.update(
        long_name="Rossby number",
        units="1",
        definition="Ro = zeta / f, f = f0 + beta * (YC - y_ref)",
        f0=f0,
        beta=beta,
        y_ref=f.attrs["y_ref"],
        reference=f.attrs["reference"],
        min_abs_f=min_abs_f,
    )

    if return_components:
        return xr.Dataset({ro.name: ro, zeta.name: zeta, f.name: f})
    return ro


# Backward-compatible short alias, matching older notebook naming habits.
def Ro(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    **kwargs,
) -> xr.DataArray | xr.Dataset:
    return rossby_number(ds_grid, ds_ocean, **kwargs)
