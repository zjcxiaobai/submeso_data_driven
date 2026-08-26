# Temporal Bottleneck Experiment

This isolated experiment keeps time as an ordered dimension instead of concatenating time snapshots as independent channels.

Architecture:

```text
input sequence (B, T=9, C=8, H=20, W=40)
  -> shared spatial CNN applied to each time frame
  -> temporal Conv3D over the ordered T dimension
  -> 6 learned spatial bottleneck maps (B, 6, H, W)
  -> convolutional decoder
  -> final-time heat flux prediction (B, 1, H, W)
```

This is designed to test whether ordered temporal history helps compared with treating all time frames as 72 unordered channels.

Run from `_src`:

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python analysis/temporal_bottleneck/run_temporal_bottleneck.py train --config analysis/temporal_bottleneck/config.yaml
python analysis/temporal_bottleneck/run_temporal_bottleneck.py evaluate --config analysis/temporal_bottleneck/config.yaml
```

Outputs are written to `analysis/temporal_bottleneck/outputs/`.

The evaluation NPZ stores `latent_features` with shape `(N_test, 6, 20, 40)`. The summary text reports validation and test R2.


## Feature Analysis

After running evaluation, plot the 6 learned temporal-bottleneck feature maps and their correlations with same-time input variables:

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python analysis/temporal_bottleneck/plot_temporal_features.py
```

Outputs are written to:

```text
analysis/temporal_bottleneck/feature_plots/
analysis/temporal_bottleneck/feature_plots/input_correlations/
```

The default correlation mode compares each of the 6 latent maps with same-time input features at the selected best, worst, and middle test samples. To correlate against all temporal offsets in the input window:

```bash
python analysis/temporal_bottleneck/plot_temporal_features.py --input-mode all-offsets
```

Run PCA on the best test sample's 6 learned feature maps:

```bash
python analysis/temporal_bottleneck/pca_temporal_best_sample_features.py
```

PCA outputs are written to:

```text
analysis/temporal_bottleneck/feature_plots/pca_best_sample/
```

To run PCA for a specific test position instead of the best-R2 sample:

```bash
python analysis/temporal_bottleneck/pca_temporal_best_sample_features.py --position 123
```
