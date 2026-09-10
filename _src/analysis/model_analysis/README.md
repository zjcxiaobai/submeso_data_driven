# Final Model Analysis Configuration

This folder contains the model promoted from
`lr_clip_tuning/outputs/lr_1p0em03__clip_none__sched_plateau`.
The checkpoint was selected at epoch 89 by minimum validation loss (0.05609848).
Validation R2 is 0.72841083, global test R2 is 0.70489770, and mean snapshot
test R2 is 0.69511562 across 1143 snapshots.
The previous model and settings are preserved in
`backups/pre_lr_clip_promotion_20260908/`.

Important: this is **not** the old `UNet_3Layers_EncoderConvLSTM` stack. The model keeps time as a separate sequence axis:

```text
input: (batch, time, feature, y, x)
spatial encoder: applied to each time frame
3D temporal encoder: learns across the time axis
bottleneck: 4 spatial feature maps
spatial decoder: maps the 4 feature maps to heat flux
```

Final settings:

```text
features = U, V, W, zeta, strain, T, rho, kpphbl
domain = upper_40x20, y=30:50, x=10:50
temporal_offsets = [-5, -4, -3, -2, -1, 0]
sequence_length = 6
input_channels per frame = 8
latent_channels = 4
spatial_channels = [64, 128, 192]
temporal_channels = 192
temporal_layers = 5
decoder_channels = [64, 32]
temporal_reduce = final
dropout = 0.15
lr = 1e-3 (initial; 5e-4 at the selected checkpoint)
lr_scheduler = ReduceLROnPlateau, monitoring validation loss
scheduler_factor = 0.5
scheduler_patience = 10
scheduler_min_lr = 1e-5
gradient_clip_norm = 0 (disabled)
weight_decay = 1e-4
loss = huber (PyTorch SmoothL1Loss, beta=1)
optimizer = AdamW
epochs = 250 (maximum)
early_stopping_patience = 40 (validation loss)
latent_smooth_weight = 0
latent_corr_weight = 0
seed = 42
batch_size = 64
split = raw time blocks, 80/10/10
```

The raw per-time arrays are saved in:

```text
analysis/model_analysis/data/X.npy
analysis/model_analysis/data/y.npy
analysis/model_analysis/data/time_index.npy
```

Temporal windows are built by `run_model_analysis.py` from `config.yaml`; the builder does not pre-stack the sequence dimension.

`temporal_reduce = final` means the model sees the full sequence `[-5, -4, -3, -2, -1, 0]`, then uses the final temporal hidden representation to create the 4 bottleneck feature maps for predicting heat flux at `t=0`.

## Rebuild Data

Run this when you want to replace the old `analysis/model_analysis/data` arrays:

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python analysis/model_analysis/build_upper40x20_dataset.py
```

## Train

```bash
python analysis/model_analysis/run_model_analysis.py train --config analysis/model_analysis/config.yaml
```

## Evaluate

```bash
python analysis/model_analysis/run_model_analysis.py evaluate --config analysis/model_analysis/config.yaml
```

Outputs are written to:

```text
analysis/model_analysis/outputs_final
```
