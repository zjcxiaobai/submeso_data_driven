# Temporal bottleneck feature sweep

This folder tests the temporal bottleneck model with `latent_channels` from 1 to 8.

The configs were copied from `analysis/temporal_bottleneck/config.yaml`, with only the bottleneck feature number and output paths changed.

## Run all feature counts

From `_src`:

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python analysis/temporal_bottleneck_feature_sweep/run_sweep.py train-evaluate --latents all
```

## Run selected feature counts

```bash
python analysis/temporal_bottleneck_feature_sweep/run_sweep.py train-evaluate --latents 1,2,4,6,8
```

## Separate train/evaluate steps

```bash
python analysis/temporal_bottleneck_feature_sweep/run_sweep.py train --latents all
python analysis/temporal_bottleneck_feature_sweep/run_sweep.py evaluate --latents all
```

## Summarize existing outputs

```bash
python analysis/temporal_bottleneck_feature_sweep/run_sweep.py summarize --latents all
```

The summary table is written to:

```text
analysis/temporal_bottleneck_feature_sweep/outputs/sweep_summary.csv
```

Each feature count writes its own checkpoint, history, predictions, and evaluation output under:

```text
analysis/temporal_bottleneck_feature_sweep/outputs/latent_1
...
analysis/temporal_bottleneck_feature_sweep/outputs/latent_8
```
