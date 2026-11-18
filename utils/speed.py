import xarray as xr
import numpy as np

def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in ds]
    if missing:
        raise KeyError(f"Dataset is missing required variable(s): {missing}")


def _require_dims(da: xr.DataArray, dims: tuple[str, ...], name: str) -> None:
    missing = [dim for dim in dims if dim not in da.dims]
    if missing:
        raise ValueError(f"{name} is missing required dimension(s): {missing}; got dims {da.dims}")


# --------------------- horizontal velocity ----------------------------------
def _weighted_face_to_center(
    face: xr.DataArray,
    weight: xr.DataArray,
    face_dim: str,
    center_dim: str,
    boundary: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Center adjacent C-grid faces with wet face-length weights."""
    if boundary == "periodic":
        face_next = face.roll({face_dim: -1}, roll_coords=False)
        weight_next = weight.roll({face_dim: -1}, roll_coords=False)
    elif boundary in {"fill", None}:
        face_next = face.shift({face_dim: -1}, fill_value=0.0)
        weight_next = weight.shift({face_dim: -1}, fill_value=0.0)
    else:
        raise ValueError("boundary must be 'periodic' or 'fill'")

    numerator = face * weight + face_next * weight_next
    denominator = weight + weight_next
    centered = xr.where(denominator > 0, numerator / denominator, np.nan)
    return centered.rename({face_dim: center_dim}), denominator.rename({face_dim: center_dim})


def horizontal_speed(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    u_name: str = "U",
    v_name: str = "V",
    hfac_u_name: str = "hFacW",
    hfac_v_name: str = "hFacS",
    hfac_c_name: str = "hFacC",
    dy_u_name: str = "DYG",
    dx_v_name: str = "DXG",
    u_face_dim: str = "i_g",
    v_face_dim: str = "j_g",
    x_center_dim: str = "i",
    y_center_dim: str = "j",
    x_boundary: str = "periodic",
    y_boundary: str = "fill",
    output_name: str = "speed_center",
    return_components: bool = False,
) -> xr.DataArray | xr.Dataset:
    """
    Horizontal speed at MITgcm tracer-cell centers from C-grid face velocities.

    The centered components are computed as wet-face-length weighted averages:
    Uc = (U_w * hFacW_w * DYG_w + U_e * hFacW_e * DYG_e) / sum(hFacW * DYG),
    Vc = (V_s * hFacS_s * DXG_s + V_n * hFacS_n * DXG_n) / sum(hFacS * DXG),
    speed = sqrt(Uc**2 + Vc**2), masked by hFacC. DRF cancels for a
    single vertical level, so it is not needed in the horizontal interpolation.

    Parameters
    ----------
    ds_grid, ds_ocean
        Grid metrics/masks and ocean variables. The default names match MITgcm:
        U(k,j,i_g), V(k,j_g,i), hFacW, hFacS, hFacC, DYG, and DXG.
    *_name
        Override these when a dataset uses different variable names.
    u_face_dim, v_face_dim, x_center_dim, y_center_dim
        Staggered face dimensions and the center dimensions they should be renamed to.
    x_boundary, y_boundary
        How to find the second face adjacent to each center. Re-entrant channel
        grids usually use ``x_boundary="periodic"`` and ``y_boundary="fill"``.
    return_components
        If True, return a Dataset containing speed plus centered U and V.
    """
    ds = xr.merge([ds_grid, ds_ocean])
    _require_variables(
        ds,
        (u_name, v_name, hfac_u_name, hfac_v_name, hfac_c_name, dy_u_name, dx_v_name),
    )
    _require_dims(ds[u_name], (y_center_dim, u_face_dim), u_name)
    _require_dims(ds[v_name], (v_face_dim, x_center_dim), v_name)
    _require_dims(ds[hfac_u_name], (y_center_dim, u_face_dim), hfac_u_name)
    _require_dims(ds[hfac_v_name], (v_face_dim, x_center_dim), hfac_v_name)
    _require_dims(ds[hfac_c_name], (y_center_dim, x_center_dim), hfac_c_name)
    _require_dims(ds[dy_u_name], (y_center_dim, u_face_dim), dy_u_name)
    _require_dims(ds[dx_v_name], (v_face_dim, x_center_dim), dx_v_name)

    wet_u = ds[hfac_u_name] > 0
    wet_v = ds[hfac_v_name] > 0
    U = ds[u_name].where(wet_u, 0.0)
    V = ds[v_name].where(wet_v, 0.0)
    weight_u = (ds[hfac_u_name] * ds[dy_u_name]).where(wet_u, 0.0)
    weight_v = (ds[hfac_v_name] * ds[dx_v_name]).where(wet_v, 0.0)

    Uc, Uc_den = _weighted_face_to_center(U, weight_u, u_face_dim, x_center_dim, x_boundary)
    Vc, Vc_den = _weighted_face_to_center(V, weight_v, v_face_dim, y_center_dim, y_boundary)

    has_face = (Uc_den > 0) | (Vc_den > 0)
    wet_center = ds[hfac_c_name] > 0
    speed = xr.apply_ufunc(np.hypot, Uc, Vc).where(wet_center).where(has_face)
    speed = speed.rename(output_name)
    speed.attrs.update(
        long_name="horizontal speed from face velocities centered with wet face-length weights",
        units=ds[u_name].attrs.get("units", "m s^-1"),
        x_boundary=x_boundary,
        y_boundary=y_boundary,
    )

    if not return_components:
        return speed

    Uc = Uc.rename(f"{u_name}_center")
    Vc = Vc.rename(f"{v_name}_center")
    Uc.attrs.update(long_name=f"{u_name} centered with wet face-length weights", units=speed.attrs["units"])
    Vc.attrs.update(long_name=f"{v_name} centered with wet face-length weights", units=speed.attrs["units"])
    return xr.Dataset({speed.name: speed, Uc.name: Uc, Vc.name: Vc})

def _horizontal_speed(
    ds_grid: xr.Dataset,
    ds_ocean: xr.Dataset,
    boundary_Y: str = "fill",
) -> xr.DataArray:
    """Backward-compatible wrapper for older notebooks/scripts."""
    return horizontal_speed(ds_grid, ds_ocean, y_boundary=boundary_Y, output_name="speed_coarse_center")

