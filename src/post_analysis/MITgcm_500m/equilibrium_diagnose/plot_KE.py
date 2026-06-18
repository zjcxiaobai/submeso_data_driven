import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/scratch/jczhang/submeso_data_driven/src/post_analysis/MITgcm_1km/equilibrium_diagnose/kinetic_energy_timeseries_500m.csv"

df = pd.read_csv(csv_path)
df = df[df["status"] == "ok"]

plt.figure(figsize=(8, 4))
plt.plot(df["model_days"], df["ke_mean_m2_s2"], marker="o", lw=1.5)
plt.xlabel("Model time [days]")
plt.ylabel("Domain-mean KE [m$^2$ s$^{-2}$]")
plt.title("MITgcm 500 m kinetic energy spinup")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/scratch/jczhang/submeso_data_driven/src/post_analysis/MITgcm_1km/equilibrium_diagnose/kinetic_energy_timeseries_500m.png", dpi=200)
plt.show()