# Temporal Average Correlation

This workflow tests how temporal smoothing **across consecutive test samples** changes the spatial correlation between selected physics diagnostics and bottleneck structure.

It uses:

```text
analysis/model_analysis/outputs_final/test_predictions_denorm.npz
analysis/model_analysis/data/X.npy
```

For each test sample, the selected physics variables are calculated from the raw fields at that same sample. Then both the physics-variable maps and the 4 bottleneck feature maps are smoothed using trailing rolling means across the ordered test samples.

Default rolling windows:

```text
1, 3, 7, 15, 30, 60, 90, 180, 365 test samples
```

For each variable and each rolling window, the script computes spatial correlations with:

```text
Feature 1, Feature 2, Feature 3, Feature 4
PC1, PC2, PC3, PC4
```

One image is saved per physics variable. Each image plots correlation versus rolling-window length, with one line for each bottleneck feature and PCA component.

Default diagnostics:

```text
T
rho
gradT
strain
strain_times_gradT
frontogenesis_FT
zeta
Wprime_times_Tprime
KPPhbl
```

## Run From `_src`

```bash
cd /scratch/jczhang/submeso_data_driven/_src
source ../setup
python analysis/explanability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py
```

Outputs:

```text
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_T.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_rho.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_gradT.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_strain.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_strain_times_gradT.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_frontogenesis_FT.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_zeta.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_Wprime_times_Tprime.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_KPPhbl.png
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_summary.csv
analysis/explanability/bottle_feature/temporal_average_correlation/outputs/temporal_average_correlation_results.npz
```

The first run also writes a reusable cache:

```text
analysis/explanability/bottle_feature/temporal_average_correlation/physics_variable_test_series.npz
```

Run a smaller variable list:

```bash
python analysis/explanability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py --variables T rho gradT
```

Use custom rolling windows:

```bash
python analysis/explanability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py --rolling-windows 1 5 10 30 90 365
```
