# Hyperparameter tunning

This folder is a hyperparameter test for the temporal bottleneck model.

The recommended runner is `run_one_parameter_tuning.py`. It fixes a baseline setting, then changes only one parameter at a time.

Baseline:

```text
latent_channels = 4
lr = 1e-3
channel_width = base_deep
history_length = 4
dropout = 0.15
batch_size = 64
weight_decay = 1e-4
loss = huber
```

It explores:

- learning rate from `1e-5` to `1e-2` on a log-spaced scale
- channel width from small to intentionally overfitting-size models
- input history length from `0` to `9`
- dropout values `0, 0.05, 0.10, ..., 0.40`

History length means:

```text
0 -> [0]
1 -> [-1, 0]
2 -> [-2, -1, 0]
...
9 -> [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0]
```

The intentionally overfitting channel settings use wide channels. To isolate one variable at a time, dropout and weight decay still remain fixed at the baseline values during the channel-width sweep.

## Run From `_src`

```bash
cd /scratch/jczhang/submeso_data_driven/_src
```

## One-Parameter-At-A-Time Sweeps

Create all controlled configs:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py materialize --sweep all
```

Run all sweeps:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py train-evaluate --sweep all
```

Resume after interruption:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py resume --sweep all
```

Summarize:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py summarize --sweep all
```

## Run One Sweep

Learning rate only:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py train-evaluate --sweep learning_rate
```

Channel width only:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py train-evaluate --sweep channel_width
```

History length only:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py train-evaluate --sweep history_length
```

Dropout only:

```bash
python analysis/hyperparameter_tunning/run_one_parameter_tuning.py train-evaluate --sweep dropout
```

## Dropout From Best Basis

After training `best_from_first_three.yaml`, tune dropout using that stronger config as the fixed basis:

```bash
python analysis/hyperparameter_tunning/run_dropout_from_best_basis.py materialize
python analysis/hyperparameter_tunning/run_dropout_from_best_basis.py train-evaluate
python analysis/hyperparameter_tunning/run_dropout_from_best_basis.py summarize
```

Resume if interrupted:

```bash
python analysis/hyperparameter_tunning/run_dropout_from_best_basis.py resume
```

This tests:

```text
dropout = 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
```

Fixed basis:

```text
lr = 1.8e-3
history_length = 5
spatial_channels = [128, 256, 384]
temporal_channels = 384
temporal_layers = 4
decoder_channels = [128, 64]
latent_channels = 4
weight_decay = 1e-4
loss = huber
```

Outputs:

```text
analysis/hyperparameter_tunning/configs_dropout_from_best_basis
analysis/hyperparameter_tunning/outputs_dropout_from_best_basis
```

## Outputs

Configs:

```text
analysis/hyperparameter_tunning/configs_one_at_time/learning_rate
analysis/hyperparameter_tunning/configs_one_at_time/channel_width
analysis/hyperparameter_tunning/configs_one_at_time/history_length
analysis/hyperparameter_tunning/configs_one_at_time/dropout
```

Trial outputs:

```text
analysis/hyperparameter_tunning/outputs_one_at_time/<sweep>/<trial>
```

Summary:

```text
analysis/hyperparameter_tunning/outputs_one_at_time/<sweep>/summary.csv
```

Important columns in the summary:

```text
best_train_r2
best_val_r2
test_r2
train_test_gap
history_length
dropout
lr
spatial_channels
temporal_channels
temporal_layers
```

For overfitting diagnosis, sort by `train_test_gap` and inspect configs with high `best_train_r2` but lower `test_r2`.

The older `run_hyperparameter_tunning.py` random/grid runner is kept for reference, but the controlled one-parameter sweeps above should be used for clean interpretation.
