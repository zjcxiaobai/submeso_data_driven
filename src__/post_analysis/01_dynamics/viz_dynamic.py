# plot the dynamics of the model
import xarray as xr
import numpy as np
from submeso_data_driven.utils import read_mitgcm
from submeso_data_driven.utils import vorticity
from src.utils import Ro
from submeso_data_driven.utils import build_coarse_grid
from submeso_data_driven.utils import wT_submeso
from submeso_data_driven.utils import coarsen_UVWTS
from src.utils import temperature_gradient
# --------------------- MITgcm folder path ----------------------------------
path_scratch = '/scratch/jczhang/'
path_MITgcm = 'MITgcm_baroclinic_instability_old/MITgcm/verification/tutorial_reentrant_channel/run'
grid_dir = path_scratch + path_MITgcm
# --------------------- grid variable ----------------------------------
ds_grid = read_mitgcm.read_mitgrid_staggered(grid_dir)
ds_grid_coarse = build_coarse_grid.build_coarse_grid_staggered(ds_grid,box_j=10, box_i=10)
t = 18593280
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
##########################################################
# plot: 2x2 panel
# (1) surface |∇T|      (2) vertical section |∇T|(j=1040)
# (3) surface Ro        (4) surface submeso w'T'
#########################################################
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean.cm as cmo

# ---------- helpers ----------
def _surface_2d(da: xr.DataArray) -> xr.DataArray:
    if "k" in da.dims:
        return da.isel(k=2)
    if "k_l" in da.dims:
        return da.isel(k_l=2)
    return da

def _robust_vlims(da: xr.DataArray, q=(2, 98), symmetric=False):
    vmin, vmax = np.nanpercentile(da.values, q)
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return vmin, vmax

def _xy_for_field(da2d: xr.DataArray, ds_fine, ds_coarse):
    nj, ni = da2d.sizes.get("j", None), da2d.sizes.get("i", None)
    if nj is None or ni is None:
        raise ValueError(f"Field must have dims ('j','i'). Got dims={da2d.dims}")

    if "XC" in ds_fine and ds_fine["XC"].sizes["j"] == nj and ds_fine["XC"].sizes["i"] == ni:
        return ds_fine["XC"], ds_fine["YC"]

    if "XC" in ds_coarse and ds_coarse["XC"].sizes["j"] == nj and ds_coarse["XC"].sizes["i"] == ni:
        return ds_coarse["XC"], ds_coarse["YC"]

    x = xr.DataArray(np.arange(ni), dims=("i",))
    y = xr.DataArray(np.arange(nj), dims=("j",))
    XC, YC = xr.broadcast(x, y)
    return XC.transpose("j", "i"), YC.transpose("j", "i")

def _z_for_k(da3d: xr.DataArray, ds_grid):
    """Return depth array for k (positive down) if RC exists, else k index."""
    if "k" in da3d.dims and "RC" in ds_grid:
        return np.abs(ds_grid["RC"].values)
    if "k" in da3d.dims:
        return np.arange(da3d.sizes["k"])
    return None

# ---------- choose section index (THIS is what you meant) ----------
j_sec = 1040

# ---------- 2D surface fields ----------
G_sfc  = _surface_2d(T_gradient)  # |∇T| surface
Ro_sfc = _surface_2d(Ro_number if isinstance(Ro_number, xr.DataArray) else Ro_number["Ro"])
wT_sfc = _surface_2d(wT_sub_kC)

XC, YC = _xy_for_field(G_sfc, ds_grid, ds_grid_coarse)

# ---------- vertical section of |∇T| at j=1040 ----------
if "k" not in T_gradient.dims:
    raise ValueError("T_gradient must have dim 'k' to plot a vertical section.")
G_sec = T_gradient.isel(j=j_sec)  # (k,i)

# x coordinate along section
if "XC" in ds_grid:
    x_sec = ds_grid["XC"].isel(j=j_sec).values
else:
    x_sec = np.arange(G_sec.sizes["i"])

# z coordinate
z_sec = _z_for_k(G_sec, ds_grid)

# ---------- plot 2x2 ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, lab in zip(axes.ravel(), ["(a)", "(b)", "(c)", "(d)"]):
    ax.set_title(ax.get_title(), pad=10)  # keep your existing title
    ax.text(0.0, 1.02, lab, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=12, fontweight="bold")
    
XC = XC[620:1180, 20:580]
YC = YC[620:1180, 20:580]
# (1) surface |∇T|
ax = axes[0, 0]
vmin, vmax = _robust_vlims(G_sfc, q=(2, 98), symmetric=False)
pm = ax.pcolormesh(XC/1000, YC/1000, G_sfc[620:1180, 20:580], shading="auto", cmap=cmo.thermal, vmin=vmin, vmax=vmax)
ax.set_title(r" temperature gradient $|\nabla_h T|$ (k=2)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
cb = fig.colorbar(pm, ax=ax, pad=0.02, shrink=0.9)
cb.set_label(r"$|\nabla_h T|$")

# (2) vertical section |∇T| at j=1040
ax = axes[0, 1]
vmin, vmax = _robust_vlims(G_sec, q=(2, 98), symmetric=False)
pm = ax.pcolormesh(x_sec[20:580]/1000, z_sec, G_sec.values[:,20:580], shading="auto", cmap=cmo.thermal, vmin=vmin, vmax=vmax)
ax.set_title(rf"vertical profile $|\nabla_h T|$ at y={j_sec * 5} km")
ax.set_xlabel("x (km)")
ax.set_ylabel("depth (m)" if "RC" in ds_grid else "k index")
if "RC" in ds_grid:
    ax.set_ylim(z_sec.max(), 0)  # surface on top
cb = fig.colorbar(pm, ax=ax, pad=0.02, shrink=0.9)
cb.set_label(r"$|\nabla_h T|$")

# (3) surface Ro
ax = axes[1, 0]
vmin, vmax = _robust_vlims(Ro_sfc, q=(2, 98), symmetric=True)
pm = ax.pcolormesh(XC/1000, YC/1000, Ro_sfc[620:1180, 20:580], shading="auto", cmap=cmo.balance, vmin=vmin, vmax=vmax)
ax.set_title(" Rossby number Ro (k=2)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
cb = fig.colorbar(pm, ax=ax, pad=0.02, shrink=0.9)
cb.set_label("Ro")

# (4) surface submeso w'T'
ax = axes[1, 1]
vmin, vmax = _robust_vlims(wT_sfc, q=(2, 98), symmetric=True)
pm = ax.pcolormesh(ds_grid_coarse["XC"][62:118, 2:58]/1000, ds_grid_coarse["YC"][62:118, 2:58]/1000, wT_sfc[62:118, 2:58], shading="auto", cmap=cmo.balance, vmin=vmin, vmax=vmax)
ax.set_title(r" submesoscale heat flux $w'T'$ (k=2)")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
cb = fig.colorbar(pm, ax=ax, pad=0.02, shrink=0.9)
cb.set_label(r"$w'T'$")


out_png = f"/scratch/jczhang/submeso_data_driven/src/post_analysis/dynamics/dynamics.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)
# plt.show()
print("Saved:", out_png)