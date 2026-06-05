import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from MITgcmutils import mds
import cmocean.cm as cmo

# ============================================================
# Helper functions
# ============================================================

def smooth_meridional_profile(val_south, val_north, j_norm, sharpness=6.0):
    xi = (j_norm - 0.5) * sharpness
    w = 0.5 * (1.0 + np.tanh(xi))
    return val_south * (1.0 - w) + val_north * w

def add_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, color='red', alpha=0.08, zorder=1):
    v = np.array([
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ])

    faces = [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # south
        [v[3], v[2], v[6], v[7]],  # north
        [v[0], v[3], v[7], v[4]],  # west
        [v[1], v[2], v[6], v[5]],  # east
    ]

    box = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors='none',
        alpha=alpha
    )
    box.set_zorder(zorder)
    ax.add_collection3d(box)

# ============================================================
# 1. Read MITgcm grid and bathymetry
# ============================================================

path_scratch = '/scratch/jczhang/'
path_MITgcm = 'MITgcm_baroclinic_instability_old/MITgcm/verification/tutorial_reentrant_channel/run/'
grid_dir = path_scratch + path_MITgcm

XC = mds.rdmds(grid_dir + 'XC')
YC = mds.rdmds(grid_dir + 'YC')
RC = mds.rdmds(grid_dir + 'RC')

Ny, Nx = XC.shape
Nr = RC.size

bathy = np.fromfile(grid_dir + 'bathy.5km.bin', dtype='>f4')
bathy = bathy.reshape(Ny, Nx)

X_km = XC / 1000.0
Y_km = YC / 1000.0

x_min = float(X_km.min())
x_max = float(X_km.max())
y_min = float(Y_km.min())
y_max = float(Y_km.max())

# ============================================================
# 2. Build initial temperature field
# ============================================================

dr = np.array([
    5.48716549,   6.19462098,   6.99291201,   7.89353689,
    8.90937723,  10.05483267,  11.34595414,  12.80056778,
    14.43837763,  16.28102917,  18.35210877,  20.67704362,
    23.28285446,  26.19769810,  29.45012046,  33.06792588,
    37.07656002,  41.49691200,  46.34247864,  51.61592052,
    57.30518684,  63.37960847,  69.78661289,  76.44996107,
    83.27047568,  90.13003112,  96.89898027, 103.44631852,
   109.65099217, 115.41222750, 120.65692923, 125.34295968,
   129.45821977, 133.01641219, 136.05088105, 138.60793752,
   140.74074276, 142.50436556, 143.95220912, 145.13372400,
   146.09317287, 146.86917206, 147.49475454, 147.99774783,
   148.40131516, 148.72455653, 148.98310489, 149.18968055,
   149.35458582
])

nr = len(dr)

z_w = np.concatenate(([0.0], -np.cumsum(dr)))
z_c = 0.5 * (z_w[:-1] + z_w[1:])
depth = -z_c
z_plot = -depth

j = np.arange(Ny)
j_norm = j / (Ny - 1.0)

T_south = -1.0
T_north = 6.0
T_surf = smooth_meridional_profile(T_south, T_north, j_norm, sharpness=5.0)

MLD_south = 250.0
MLD_north = 60.0
MLD = smooth_meridional_profile(MLD_south, MLD_north, j_norm, sharpness=4.0)

T_deep = 0.5
h_T = 700.0

depth_3d = depth[:, None, None]
MLD_3d = MLD[None, :, None]
T_surf_3d = T_surf[None, :, None]

above_mld = depth_3d <= MLD_3d
dz_below = np.maximum(depth_3d - MLD_3d, 0.0)

T_below = T_deep + (T_surf_3d - T_deep) * np.exp(-dz_below / h_T)
T_kj1 = np.where(above_mld, T_surf_3d, T_below)

T_3D = np.repeat(T_kj1, Nx, axis=2)

# ============================================================
# 3. Sponge layer info
# ============================================================

j_strong = 10
j_weak = 20

y_weak_start = float(Y_km[-j_weak, 0])
y_strong_start = float(Y_km[-j_strong, 0])

# ============================================================
# 4. Create figure
# ============================================================

fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection='3d')

# IMPORTANT: disable automatic z-order sorting
# so artists drawn later / with larger zorder stay on top
try:
    ax.computed_zorder = False
except Exception:
    pass

# ============================================================
# 5. Plot bathymetry
# ============================================================

bathy_skip_x = 6
bathy_skip_y = 6

bathy_surf = ax.plot_surface(
    X_km[10::bathy_skip_y, ::bathy_skip_x],
    Y_km[10::bathy_skip_y, ::bathy_skip_x],
    bathy[10::bathy_skip_y, ::bathy_skip_x],
    cmap='copper',
    edgecolor='none',
    linewidth=0,
    alpha=0.92
)
bathy_surf.set_zorder(1)

# ============================================================
# 6. Plot meridional temperature section at x = 0 FIRST
# ============================================================

x_sec_km = 0.0
i_sec = np.argmin(np.abs(X_km[0, :] - x_sec_km))

y_sec = Y_km[:, i_sec]
T_sec = T_3D[:, :, i_sec]

skip_temp_y = 4
skip_temp_z = 1

Ycur, Zcur = np.meshgrid(
    y_sec[::skip_temp_y],
    z_plot[::skip_temp_z]
)

Xcur = np.full_like(Ycur, X_km[0, i_sec])
Tcur = T_sec[::skip_temp_z, ::skip_temp_y]

normT = colors.Normalize(vmin=-1.0, vmax=6.0)
facecolors = cmo.thermal(normT(Tcur))

temp_surf = ax.plot_surface(
    Xcur,
    Ycur,
    Zcur,
    facecolors=facecolors,
    rstride=1,
    cstride=1,
    linewidth=0,
    antialiased=False,
    shade=False,
    alpha=1.0
)
temp_surf.set_zorder(2)


# ============================================================
# 7. Plot sponge layer (not in legend)
# ============================================================

zmin = -4000.0
zmax = 0.0

add_box(
    ax,
    x_min, x_max,
    y_weak_start, y_strong_start,
    zmin, zmax,
    color='orange',
    alpha=0.06,
    zorder=3
)

add_box(
    ax,
    x_min, x_max,
    y_strong_start, y_max,
    zmin, zmax,
    color='red',
    alpha=0.06,
    zorder=3
)

# move sponge layer text a bit higher
sponge_txt = ax.text(
    2200.0,
    y_weak_start + 100.0,
    -120.0,
    "sponge layer",
    color='red',
    fontsize=12
)
sponge_txt.set_zorder(9)

# ============================================================
# 8. Plot wind profile LAST so it stays visible on top
# ============================================================

y_line = np.linspace(0.0, 6000.0, 1000)
tauX = 0.2 * np.sin((y_line / 6000.0) * np.pi)

# slight x offset so the line is not buried in the x=0 plane
x0 = 20.0
wind_scale = 6000.0
x_line = x0 + wind_scale * tauX

# put the whole wind line above the temperature plane
z_wind = 450.0
z_line = np.full_like(y_line, z_wind)

wind_line, = ax.plot(
    x_line,
    y_line,
    z_line,
    color='k',
    linewidth=3.5
)
wind_line.set_zorder(20)

# red wind arrow
y_arrow = 3000.0
tau_arrow = 0.2 * np.sin((y_arrow / 6000.0) * np.pi)
arrow_len = wind_scale * tau_arrow

wind_arrow = ax.quiver(
    120.0, y_arrow, z_wind,
    arrow_len, 0.0, 0.0,
    color='red',
    linewidth=2.8,
    arrow_length_ratio=0.18,
    normalize=False
)
wind_arrow.set_zorder(21)

# wind text as 2D overlay so it is never hidden
ax.text2D(
    0.10, 0.75,
    r"wind stress $\tau_{\max}=0.2$ N m$^{-2}$",
    transform=ax.transAxes,
    color='red',
    fontsize=12
)

# ============================================================
# 9. Colorbar
# ============================================================

mappable = cm.ScalarMappable(norm=normT, cmap=cmo.thermal)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.60, pad=0.08)
cbar.set_label("Initial temperature (°C)", fontsize=12)

# ============================================================
# 10. Legend: only wind
# ============================================================

legend_handles = [
    Line2D([0], [0], color='k', lw=3, label='Wind profile'),
]

ax.legend(
    handles=legend_handles,
    loc='upper left',
    bbox_to_anchor=(0.8, 0.18),
    fontsize=10
)

# ============================================================
# 11. Axes settings
# ============================================================

ax.set_xlim(0, 3000)
ax.set_ylim(0, 6000)
ax.set_zlim(-4000, 500)

ax.set_xlabel('x (km)', fontsize=13, labelpad=10)
ax.set_ylabel('y (km)', fontsize=13, labelpad=10)
ax.set_zlabel('z (m)', fontsize=13, labelpad=10)

# no title
ax.set_box_aspect((3, 5, 2.3))
ax.view_init(elev=24, azim=-58)

# ============================================================
# 12. Save
# ============================================================

out_dir = "/scratch/jczhang/submeso_data_driven/src/post_analysis/MITgcm_5km/02_MITgcm_domain"
os.makedirs(out_dir, exist_ok=True)

out = os.path.join(out_dir, "bathy_wind_temp_x0_final_visible_wind.png")

plt.savefig(out, dpi=400, bbox_inches="tight")
plt.show()

print("Saved to:", out)