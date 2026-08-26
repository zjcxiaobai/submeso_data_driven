# Bottleneck Feature Count Sweep

This isolated experiment tests the number of learned spatial bottleneck feature maps from 1 to 8. It reuses the supervised bottleneck implementation from `analysis/bottleneck_features/run_bottleneck.py`, but each feature count has its own config and output directory.

Architecture for each run:

```text
input window (72,20,40)
  -> convolutional encoder
  -> K learned spatial feature maps (K,20,40)
  -> convolutional decoder
  -> heat flux prediction (1,20,40)
```

where `K = 1, 2, ..., 8`.

The sweep uses a smaller regularized setup:

```yaml
encoder_channels: [32, 64]
decoder_channels: [32, 16]
dropout: 0.20
loss: huber
weight_decay: 3.0e-4
latent_smooth_weight: 1.0e-4
latent_corr_weight: 1.0e-4
```

The weaker `1e-4` latent penalties are intentional: the previous `1e-3` setting reduced overfitting but also reduced train/validation skill.

## Run

From `_src`:

```bash
cd /scratch/jczhang/submeso_data_driven/_src
```

Train and evaluate all feature counts:

```bash
python analysis/bottleneck_feature_sweep/run_sweep.py train-evaluate --latents all
```

Train/evaluate one feature count, for example K=6:

```bash
python analysis/bottleneck_feature_sweep/run_sweep.py train-evaluate --latents 6
```

Train/evaluate selected feature counts:

```bash
python analysis/bottleneck_feature_sweep/run_sweep.py train-evaluate --latents 1,2,4,6,8
```

If training and evaluation are done separately:

```bash
python analysis/bottleneck_feature_sweep/run_sweep.py train --latents all
python analysis/bottleneck_feature_sweep/run_sweep.py evaluate --latents all
```

Create a CSV summary after evaluation:

```bash
python analysis/bottleneck_feature_sweep/run_sweep.py summarize --latents all
```

Summary output:

```text
analysis/bottleneck_feature_sweep/outputs/sweep_summary.csv
```

Each run writes to:

```text
analysis/bottleneck_feature_sweep/outputs/latent_1/
analysis/bottleneck_feature_sweep/outputs/latent_2/
...
analysis/bottleneck_feature_sweep/outputs/latent_8/
```

## Interpret

A useful result would look like this:

```text
K increases -> train R2 rises
validation/test R2 rise then plateau
```

Choose the smallest K near the validation/test plateau. If train R2 keeps rising but validation/test do not, larger K is adding non-general capacity.
