# Bottleneck Feature Maps

This folder plots the 4 learned bottleneck feature maps saved by the final `model_analysis` evaluation file:

```text
results/final_model/test_predictions_denorm.npz
```

The script reads the saved `latent_features` array with shape:

```text
(test_sample, 4, 20, 40)
```

## Run From `_src`

```bash
cd /scratch/jczhang/submeso_data_driven/_src
source ../setup
python analysis/explanability/bottle_feature/plot_bottleneck_features.py
```

Default outputs:

```text
bottleneck_4_features_best.png
bottleneck_4_features_selected_samples.csv
```

Plot one specific test sample:

```bash
python analysis/explanability/bottle_feature/plot_bottleneck_features.py --sample-index 10384
```

Plot best, middle, worst, and test mean:

```bash
python analysis/explanability/bottle_feature/plot_bottleneck_features.py --sample-mode best_middle_worst
```

Plot every test sample:

```bash
python analysis/explanability/bottle_feature/plot_bottleneck_features.py --sample-mode all
```

## Best-Sample Physics, PCA, And Correlations

Calculate the same physics diagnostics used by `model_analysis_last_decoder_pca`, but for the best-performing final `model_analysis` test sample:

```bash
python analysis/explanability/bottle_feature/prepare_physics_variables.py --overwrite
```

Then run PCA on the 4 bottleneck feature maps for that same best sample and make the correlation figures. The first correlation image compares physics variables with the original 4 bottleneck feature maps; the second compares physics variables with the PCA components:

```bash
python analysis/explanability/bottle_feature/pca_physics_correlation.py
```

This creates:

```text
best_sample_pca_physics/bottleneck_4_features_pca_best_sample.png
best_sample_pca_physics/physics_vs_bottleneck_features_correlation.png
best_sample_pca_physics/physics_vs_bottleneck_pca_correlation.png
best_sample_pca_physics/bottleneck_pca_physics_correlations.npz
best_sample_pca_physics/physics_vs_bottleneck_features_correlation.csv
best_sample_pca_physics/physics_vs_bottleneck_pca_correlation.csv
```
