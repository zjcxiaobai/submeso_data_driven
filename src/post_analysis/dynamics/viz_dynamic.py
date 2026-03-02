# plot the dynamics of the model
import xarray as xr
import numpy as np
from src.utils import read_mitgcm
from src.utils import vorticity
from src.utils import Ro
from src.utils import build_coarse_grid
from src.utils import wT_submeso
from src.utils import coarsen_UVWTS
from src.utils import temperature_gradient
# --------------------- MITgcm folder path ----------------------------------
path_scratch = '/scratch/jczhang/'
path_MITgcm = 'MITgcm_baroclinic_instability_old/MITgcm/verification/tutorial_reentrant_channel/run/'
grid_dir = path_scratch + path_MITgcm
# --------------------- grid variable ----------------------------------
ds_grid = read_mitgcm.read_mitgrid_staggered(grid_dir)
ds_grid_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_grid,box_j=10, box_i=10)
t = 17857152
# calculate the zeta
U = read_mitgcm._ocean_var(grid_dir, "U", t, ("k","j","i_g"))
V = read_mitgcm._ocean_var(grid_dir, "V", t, ("k","j_g","i"))
W = read_mitgcm._ocean_var(grid_dir, "W", t, ("k_l","j","i"))
W_coarse = coarsen_UVWTS.coarse_field("W", W, ds_grid, 10, 10)
T = read_mitgcm._ocean_var(grid_dir, "T", t, ("k","j","i"))
T_coarse = coarsen_UVWTS.coarse_field("T", T, ds_grid, 10, 10)

ds_var = xr.Dataset(dict(U=U, V=V))
zeta = vorticity._vorticity(ds_grid, ds_var, "fill")
ds_var = xr.Dataset(dict(U=U, V=V, zeta = zeta, W=W, T=T))
ds_var_coarse = xr.Dataset(dict(W=W_coarse, T=T_coarse))


# calulate the rossby number
f0=-1.363e-4
beta=1.313e-11
Ro_number = Ro.rossby_number(ds_var, ds_grid, f0, beta, None)
# calculate the submesoscale heat flux
wT_sub_kC, wT_bar_kC, meso_wT_KC =  wT_submeso.submeso_vertical_heat_flux(ds_grid, ds_grid_coarse, ds_var, ds_var_coarse, box_j = 10, box_i = 10,boundary_Z = "fill", to_wm2 = True)

# calculate the temperature gradient
T_gradient = temperature_gradient.T_grad(ds_grid, ds_var)

##########################################################
# plot
#########################################################
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean.cm as cmo

def _surface_2d(da: xr.DataArray) -> xr.DataArray:
    """Return a 2D (j,i) slice from da by taking the top index in k or k_l if present."""
    if "k" in da.dims:
        return da.isel(k=0)
    if "k_l" in da.dims:
        return da.isel(k_l=0)
    return da  # already 2D

def _robust_vlims(da: xr.DataArray, q=(2, 98), symmetric=False):
    vmin, vmax = np.nanpercentile(da.values, q)
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return vmin, vmax

def _xy_for_field(da2d: xr.DataArray, ds_fine, ds_coarse):
    """
    Choose XC/YC from fine or coarse grid by matching (j,i) shape.
    """
    nj, ni = da2d.sizes.get("j", None), da2d.sizes.get("i", None)
    if nj is None or ni is None:
        raise ValueError(f"Field must have dims ('j','i'). Got dims={da2d.dims}")

    # fine grid match?
    if "XC" in ds_fine and ds_fine["XC"].sizes.get("j", -1) == nj and ds_fine["XC"].sizes.get("i", -1) == ni:
        return ds_fine["XC"], ds_fine["YC"]

    # coarse grid match?
    if "XC" in ds_coarse and ds_coarse["XC"].sizes.get("j", -1) == nj and ds_coarse["XC"].sizes.get("i", -1) == ni:
        return ds_coarse["XC"], ds_coarse["YC"]

    # fallback: index coordinates
    x = xr.DataArray(np.arange(ni), dims=("i",))
    y = xr.DataArray(np.arange(nj), dims=("j",))
    XC, YC = xr.broadcast(x, y)
    return XC.transpose("j","i"), YC.transpose("j","i")

def plot_three_fields_row(
    fields, titles, cmaps, symmetric_flags,
    ds_fine, ds_coarse,
    out_png="dyn_3panel.png",
    q=(2, 98)
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    for ax, da, ttl, cmap, sym in zip(axes, fields, titles, cmaps, symmetric_flags):
        da2d = _surface_2d(da)
        XC, YC = _xy_for_field(da2d, ds_fine, ds_coarse)

        vmin, vmax = _robust_vlims(da2d, q=q, symmetric=sym)
        pm = ax.pcolormesh(XC, YC, da2d, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(ttl)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        cb = fig.colorbar(pm, ax=ax, pad=0.02, shrink=0.92)
        cb.set_label(getattr(da2d, "name", ""))

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")

# -------------------------
# YOUR THREE FIELDS (edit titles if you like)
# -------------------------
Ro2d = Ro_number if isinstance(Ro_number, xr.DataArray) else Ro_number["Ro"]
wT2d = wT_sub_kC
G2d  = T_gradient

plot_three_fields_row(
    fields=[Ro_number, wT_sub_kC, T_gradient],
    titles=["Rossby number (surface)", "Submeso vertical heat flux w'T' (surface)", "Temperature gradient |∇T| (surface)"],
    cmaps=[cmo.balance, cmo.balance, cmo.thermal],
    symmetric_flags=[True, True, False],   # Ro and w'T' are signed; |∇T| should be positive
    ds_fine=ds_grid,
    ds_coarse=ds_grid_coarse,
    out_png=f"/scratch/jczhang/submeso_data_driven/src/post_analysis/MITgcm_domain/dynamics_{t}.png",
    q=(2, 98),
)