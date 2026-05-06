#plot bathymetry as 3D surface plot

import numpy as np
from matplotlib import pyplot as plt
from MITgcmutils import mds

path_scratch = '/scratch/jczhang/'
path_MITgcm = 'MITgcm_baroclinic_instability_old/MITgcm/verification/tutorial_reentrant_channel/run/'
grid_dir = path_scratch + path_MITgcm
XC = mds.rdmds(grid_dir + 'XC')
YC = mds.rdmds(grid_dir + 'YC')
RC = mds.rdmds(grid_dir + 'RC')
# extract overall MITgcm domain dimensions
Ny = XC.shape[0]
Nx = XC.shape[1]
Nr = RC.size

bathy = np.fromfile(grid_dir + 'bathy.5km.bin', dtype='>f4')
bathy.shape = (Ny,Nx)
plt.figure(figsize=(10,8)) 
ax = plt.axes(projection='3d')
ax.plot_surface(XC[10:,:]/1000, YC[10:,:]/1000, bathy[10:,:], cmap='copper', edgecolor='none');

y_line = np.linspace(0, 6000, 2000)  # km
tauX = 0.2 * np.sin((y_line/6000) * np.pi)

x0 = 0.0     # place it around mid-domain in km
scale = 6000.0  # km per (tauX unit); adjust until it looks nice

x_line = x0 + scale * tauX

ax.plot(x_line, y_line, np.zeros_like(y_line), color="k", linewidth=3)

# ---- arrow from (0, y_mid, 0) to (tauX(y_mid), y_mid, 0) ----
y_mid = 3000.0  # km
tau_mid = 1000 * np.sin((y_mid/6000) * np.pi)

# 3D arrow via quiver: start + direction vector
ax.quiver(
    0.0, y_mid, 0.0,          # start (x,y,z)
    tau_mid - 0.0, 0.0, 0.0,   # direction (dx,dy,dz) -> purely in x
    color="red",
    linewidth=2.5,
    arrow_length_ratio=0.2,
    normalize=False
)

# place text a bit above the arrow in y, on z=0
ax.text(
    0.0, y_mid + 250.0, 0.0,   # (x, y, z) in your plot units
    r"wind stress $\tau_{\max}=0.2$",
    color="red",
    fontsize=11
)

ax.view_init(40, -50)
ax.set_zlim(-4000,0)
ax.set_zlabel('Z (m)', fontsize=16)
ax.set_title('Tutorial SO Reentrant Channel Bathymetry', fontsize=18)
plt.xlabel('x (km)',fontsize=16); plt.ylabel('y (km)', fontsize=16)
plt.xlim(0,3000)
plt.ylim(0,6000);
plt.savefig("/scratch/jczhang/submeso_data_driven/src/post_analysis/MITgcm_domain/bathy_wind.png", dpi=300, bbox_inches="tight")
plt.show()