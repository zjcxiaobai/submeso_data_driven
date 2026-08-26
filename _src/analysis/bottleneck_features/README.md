# End-to-End Spatial Bottleneck Feature Experiment

This folder is an isolated supervised bottleneck test. It does not modify the existing `model_analysis` training code.

Architecture:

```text
input temporal window -> convolutional encoder -> 6 learned spatial feature maps -> convolutional decoder -> heat-flux map
```

The learned features have the same spatial size as the input/target domain: `6 x H x W` by default, currently `6 x 20 x 40`. They are trained end-to-end by minimizing the prediction error against the heat-flux target.

Run from `_src`:

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python analysis/bottleneck_features/run_bottleneck.py train --config analysis/bottleneck_features/config.yaml
python analysis/bottleneck_features/run_bottleneck.py evaluate --config analysis/bottleneck_features/config.yaml
python analysis/bottleneck_features/plot_learned_features.py
python analysis/bottleneck_features/pca_best_sample_features.py
```

Outputs for the current regularized config are written to `analysis/bottleneck_features/outputs_regularized/`. The evaluation output includes both validation R2 and test R2 in `evaluation_summary.txt`.

Feature plots are written to `analysis/bottleneck_features/feature_plots/` by default. The plotting script saves per-sample 6-feature grids, test-set mean/std feature maps, and a feature-correlation matrix. It also saves same-time latent-input correlation heatmaps/CSVs for the selected best, worst, and middle test samples under `analysis/bottleneck_features/feature_plots/input_correlations/`.

To correlate against every temporal offset in the input window instead of only the same-time snapshot, run:

```bash
python analysis/bottleneck_features/plot_learned_features.py --input-mode all-offsets
```

Notes:
- The split is raw chronological time blocks first, then temporal windows are built inside each block.
- With offsets `[-8..0]`, no train/validation/test window crosses a split boundary.
- Change `model.latent_channels` to `5` or `6` to test how many spatial features are needed.

## Best-Sample PCA

`pca_best_sample_features.py` performs PCA on the best test sample's learned latent maps. Pixels are observations and the 6 learned feature maps are variables. Outputs are written to `analysis/bottleneck_features/feature_plots/pca_best_sample/` and include PC score maps, feature loadings, explained variance, and a compressed NPZ with the PCA arrays.

To analyze a specific test position instead of the best-R2 sample:

```bash
python analysis/bottleneck_features/pca_best_sample_features.py --position 123
```

## Regularized Test

The current config uses a smaller encoder/decoder, Huber loss, stronger weight decay, latent smoothness, and latent decorrelation. It writes to `analysis/bottleneck_features/outputs_regularized/` so it does not mix with earlier bottleneck runs.
