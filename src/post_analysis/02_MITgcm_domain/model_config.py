import numpy as np
import cmocean.cm as cmo
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
# coordinate parameters
dr = np.array([5.48716549,   6.19462098,   6.99291201,   7.89353689, \
               8.90937723,  10.05483267,  11.34595414,  12.80056778, \
              14.43837763,  16.28102917,  18.35210877,  20.67704362, \
              23.28285446,  26.1976981 ,  29.45012046,  33.06792588, \
              37.07656002,  41.496912  ,  46.34247864,  51.61592052, \
              57.30518684,  63.37960847,  69.78661289,  76.44996107, \
              83.27047568,  90.13003112,  96.89898027, 103.44631852, \
             109.65099217, 115.4122275 , 120.65692923, 125.34295968, \
             129.45821977, 133.01641219, 136.05088105, 138.60793752, \
             140.74074276, 142.50436556, 143.95220912, 145.133724  , \
             146.09317287, 146.86917206, 147.49475454, 147.99774783, \
             148.40131516, 148.72455653, 148.98310489, 149.18968055, \
             149.35458582])
nx = 600
ny = 1200
# dr: (nr,) layer thicknesses, already defined above
nr = len(dr)
#############################################################################
# create realistic temperature and salinity initial condition
#############################################################################
# Vertical interfaces (z_w) and centers (z_c), z<0 downward
z_w = np.concatenate(([0.0], -np.cumsum(dr)))        # interfaces: 0, -5.4, ...
z_c = 0.5 * (z_w[:-1] + z_w[1:])                       # cell centers
depth = -z_c                                           # positive downward
H = depth[-1]                                          # total depth (m)

# ----------------------------
# Meridional coordinate (y) for shaping profiles
# ----------------------------
# y index 0 = "south", ny-1 = "north"
j = np.arange(ny)
j_norm = j / (ny - 1.0)    # 0 (south) → 1 (north)

# You can think of j_norm ~ latitude-like coordinate.

# ----------------------------
# Surface T, S profiles (SST, SSS)
# ----------------------------
# Rough Southern-Ocean-flavored values:
# South: cold (-1.5 C), fresh-ish (33.8)
# North: warmer (8 C), saltier (35.3)

T_south = -1.0
T_north =  6.0
S_south = 34.0
S_north = 35.0

# Use a smooth tanh-like transition instead of perfectly linear
def smooth_meridional_profile(val_south, val_north, j_norm, sharpness=6.0):
    """Sigmoidal transition from south to north."""
    # center the transition around mid-channel
    xi = (j_norm - 0.5) * sharpness
    w = 0.5 * (1.0 + np.tanh(xi))   # goes from 0 to 1
    return val_south * (1 - w) + val_north * w

T_surf = smooth_meridional_profile(T_south, T_north, j_norm, sharpness=5.0)  # (ny,)
S_surf = smooth_meridional_profile(S_south, S_north, j_norm, sharpness=5.0)  # (ny,)

# ----------------------------
# Mixed layer depth profile MLD(y)
# ----------------------------
# Southern Ocean-ish:
# Deep mixed layers in south (200-300 m), shallower north (50-80 m)
MLD_south = 250.0  # m
MLD_north =  60.0  # m
MLD = smooth_meridional_profile(MLD_south, MLD_north, j_norm, sharpness=4.0)  # (ny,)

# ----------------------------
# Deep water T, S and vertical decay scales
# ----------------------------
# Deep ACC waters: cold ~0.5 C, salty-ish ~34.7
T_deep = 0.5
S_deep = 34.7

h_T = 700.0   # e-folding scale for T below MLD (m)
h_S = 1000.0  # e-folding scale for S below MLD (m)

# ----------------------------
# Build 3D T and S fields
# ----------------------------
depth_3d   = depth[:, None, None]      # (nr,1,1)
MLD_3d     = MLD[None, :, None]        # (1,ny,1)
T_surf_3d  = T_surf[None, :, None]     # (1,ny,1)
S_surf_3d  = S_surf[None, :, None]     # (1,ny,1)

above_mld = depth_3d <= MLD_3d         # (nr,ny,1)
dz_below  = np.maximum(depth_3d - MLD_3d, 0.0)

# Below-MLD profiles toward deep values
T_below = T_deep + (T_surf_3d - T_deep) * np.exp(-dz_below / h_T)
S_below = S_deep + (S_surf_3d - S_deep) * np.exp(-dz_below / h_S)

# Combine: if above MLD -> surface value, else -> exponential profile
T_kj1 = np.where(above_mld, T_surf_3d, T_below)  # (nr,ny,1)
S_kj1 = np.where(above_mld, S_surf_3d, S_below)  # (nr,ny,1)

# Now replicate in x (zonal) – copy across all longitudes
T_3D = np.repeat(T_kj1, nx, axis=2)              # (nr,ny,nx)
S_3D = np.repeat(S_kj1, nx, axis=2)              # (nr,ny,nx)

# Now replicate in x (zonal) – just copy across all longitudes
T_3D = np.repeat(T_3D[:, :, :1], nx, axis=2)
S_3D = np.repeat(S_3D[:, :, :1], nx, axis=2)

# ----------------------------
# RBCS masks (sponge layer at the north)
# ----------------------------
rbcs_mask_T = np.zeros((nr, ny, nx), dtype="float64")
rbcs_mask_S = np.zeros((nr, ny, nx), dtype="float64")

# no restoring in top cell (k=0), core model handles SST/SSS restoring
j_strong = 10   # last 10 grid rows: strong sponge
j_weak   = 20   # last 20 grid rows: weak sponge transition

# Strong sponge in northernmost 10 grid rows, below surface:
rbcs_mask_T[1:, -j_strong:, :] = 1.0
rbcs_mask_S[1:, -j_strong:, :] = 1.0

# Weaker sponge in next 10 grid rows:
rbcs_mask_T[1:, -j_weak:-j_strong, :] = 0.25
rbcs_mask_S[1:, -j_weak:-j_strong, :] = 0.25

########################################################################
# plot the configs
########################################################################
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo

# ---- assumptions: these exist from your script ----
# T_3D: (nr, ny, nx)
# S_3D: (nr, ny, nx)
# T_surf: (ny,)
# S_surf: (ny,)
# MLD:    (ny,)
# depth:  (nr,) positive downward
# rbcs_mask_T: (nr, ny, nx)

# Use physical y if you know dy (meters). If not, leave as index.
# dy_m = 5000.0
# y = np.arange(ny) * dy_m / 1000.0   # km
# y_label = "y (km)"
y = np.arange(ny)
y_label = "y (km)"

z = depth  # m, positive downward

# Pick an x-slice (since fields are zonally uniform initially, x=0 is fine)
T_sec = T_3D[:, :, 0]           # (nr, ny)
S_sec = S_3D[:, :, 0]
mask_sec = rbcs_mask_T[:, :, 0] # (nr, ny)

# Create mesh for pcolormesh
Y, Z = np.meshgrid(y, z)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# (a) Surface restoring targets
ax = axs[0, 0]
ax.plot(y, T_surf, label="SST target")
ax.set_xlabel(y_label)
ax.set_ylabel("Temperature (°C)")
ax.set_title("(a) Surface relaxation: SST")
ax.grid(True, alpha=0.3)

# (b) Mixed layer depth profile
ax = axs[0, 1]
ax.plot(y, MLD)
ax.set_xlabel(y_label)
ax.set_ylabel("MLD (m)")
ax.set_title("(b) Initial mixed layer depth MLD(y)")
ax.grid(True, alpha=0.3)

# (c) Initial temperature section + MLD overlay
ax = axs[1, 0]
pcm = ax.pcolormesh(Y, Z, T_sec, shading="auto",
                    cmap=cmo.thermal, vmin=-1.0, vmax=6.0)
ax.plot(y, MLD, linewidth=2)  # overlay MLD line
ax.invert_yaxis()
ax.set_xlabel(y_label)
ax.set_ylabel("Depth (m)")
ax.set_title("(c) Initial condition: SST(z,y), MLD(y)")
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label("Temperature (°C)")

# (d) Sponge mask (where subsurface restoring is applied)
mask_plot = np.ma.masked_where(mask_sec == 0.0, mask_sec)

cmap = cmo.amp.copy()         # any perceptual cmap is fine
cmap.set_bad("white")         # make the masked (zero) area white

# since your mask has discrete values (0, 0.25, 1.0), use discrete bins
bounds = [0.0, 0.125, 0.625, 1.01]   # splits: 0–0.125, 0.125–0.625, 0.625–1.01
norm = colors.BoundaryNorm(bounds, cmap.N)
ax = axs[1, 1]
pcm2 = ax.pcolormesh(Y, Z, mask_plot, shading="auto", cmap=cmap, norm=norm)
ax.set_xlim(y[-50], y[-1])
ax.set_title("(d) Northern sponge mask (RBCS): restoring strength")
ax.invert_yaxis()
ax.set_xlabel(y_label)
cbar2 = fig.colorbar(pcm2, ax=ax)
cbar2.set_ticks([0.25, 1.0])
cbar2.set_ticklabels(["0.25 (weak)", "1.0 (strong)"])
out = "/scratch/jczhang/submeso_data_driven/src/post_analysis/MITgcm_domain/model_config.png"
fig.savefig(out, dpi=300, bbox_inches="tight")  # save
plt.show()






