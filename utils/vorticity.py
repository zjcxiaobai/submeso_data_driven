# codes for calculate the vorticity - match the vorticity.py
# --------------------- modules ----------------------------------
import sys
from pathlib import Path

import xarray as xr

sys.path.append(str(Path.cwd().parent))
from submeso_data_driven.utils import make_grid
# --------------------- vorticity function ----------------------------------

def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variable(s): {missing}")


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dimension(s): {missing}; got dims {da.dims}")


def vorticity(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    u_name: str = "U",
    v_name: str = "V",
    hfac_u_name: str = "hFacW",
    hfac_v_name: str = "hFacS",
    hfac_c_name: str = "hFacC",
    dx_u_name: str = "DXC",
    dy_v_name: str = "DYC",
    raz_name: str = "RAZ",
    u_face_dim: str = "i_g",
    v_face_dim: str = "j_g",
    x_center_dim: str = "i",
    y_center_dim: str = "j",
    y_boundary: str = "fill",
    output_name: str = "zeta",
    return_corner: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Vertical relative vorticity at MITgcm tracer-cell centers.

    The C-grid curl is first computed at zeta/corner points and then
    interpolated to tracer centers:

      zeta_G = (delta_x[V * DYC * hFacS] - delta_y[U * DXC * hFacW]) / RAZ
      zeta   = interp_y(interp_x(zeta_G))

    Notes
    -----
    U is expected on (j, i_g), V on (j_g, i), hFacW/hFacS on the same
    horizontal face grids, and RAZ on (j_g, i_g). X is periodic through the
    local xgcm grid; Y uses ``y_boundary`` and is usually ``"fill"`` for walls.
    """
    ds = xr.merge([ds_grid, ds_ocean])
    _require_variables(
        ds,
        (
            u_name,
            v_name,
            hfac_u_name,
            hfac_v_name,
            hfac_c_name,
            dx_u_name,
            dy_v_name,
            raz_name,
        ),
    )
    _require_dims(ds[u_name], (y_center_dim, u_face_dim), u_name)
    _require_dims(ds[v_name], (v_face_dim, x_center_dim), v_name)
    _require_dims(ds[hfac_u_name], (y_center_dim, u_face_dim), hfac_u_name)
    _require_dims(ds[hfac_v_name], (v_face_dim, x_center_dim), hfac_v_name)
    _require_dims(ds[hfac_c_name], (y_center_dim, x_center_dim), hfac_c_name)
    _require_dims(ds[dx_u_name], (y_center_dim, u_face_dim), dx_u_name)
    _require_dims(ds[dy_v_name], (v_face_dim, x_center_dim), dy_v_name)
    _require_dims(ds[raz_name], (v_face_dim, u_face_dim), raz_name)

    grid = make_grid.make_grid_periodic_x(ds)

    wet_u = ds[hfac_u_name] > 0
    wet_v = ds[hfac_v_name] > 0
    wet_c = ds[hfac_c_name] > 0

    u_line = ds[u_name].where(wet_u, 0.0) * ds[dx_u_name] * ds[hfac_u_name]
    v_line = ds[v_name].where(wet_v, 0.0) * ds[dy_v_name] * ds[hfac_v_name]

    dvdx_line = grid.diff(v_line, "X")
    dudy_line = grid.diff(u_line, "Y", boundary=y_boundary)

    zeta_corner = ((dvdx_line - dudy_line) / ds[raz_name]).rename(f"{output_name}_corner")
    zeta_corner.attrs.update(
        long_name="vertical relative vorticity at zeta points",
        units="s^-1",
        y_boundary=y_boundary,
    )

    zeta = grid.interp(grid.interp(zeta_corner, "X", to="center"), "Y", to="center")
    zeta = zeta.where(wet_c).rename(output_name)
    zeta.attrs.update(
        long_name="vertical relative vorticity interpolated to tracer-cell centers",
        units="s^-1",
        y_boundary=y_boundary,
    )

    if return_corner:
        return xr.Dataset({zeta.name: zeta, zeta_corner.name: zeta_corner})
    return zeta


def _vorticity(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill",
) -> xr.DataArray:
    """Backward-compatible wrapper for older notebooks/scripts."""
    return vorticity(ds_grid, ds_ocean, y_boundary=boundary_Y, output_name="zeta")


def vorticity_coarse(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill",
) -> xr.DataArray:
    """Backward-compatible coarse-grid name used by older scripts."""
    return _vorticity(ds_grid, ds_ocean, boundary_Y=boundary_Y)
