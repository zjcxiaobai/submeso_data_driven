# Temporal bottleneck tuning

This folder tunes the temporal bottleneck model after fixing the bottleneck to 4 feature maps.

Every generated config keeps:

```yaml
model:
  latent_channels: 4

train:
  latent_smooth_weight: 0
  latent_corr_weight: 0
```

The tuning stages are:

1. Stage 1: regularization, learning rate, batch size, loss, and patience.
2. Stage 2: light architecture tuning from the best Stage 1 configs.
3. Stage 3: time-window tuning from the best Stage 2 configs.
4. Stage 4: rerun the best Stage 3 config with three seeds.
5. Final: choose using validation R2, then report test once.

## Run From `_src`

```bash
cd /scratch/jczhang/submeso_data_driven/_src
```

## Stage 1

Generate 16 fixed random configs:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py materialize --stage stage1
```

Train and evaluate them:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage1
```

Resume Stage 1 after an interrupted run. This skips trials that already have a checkpoint and evaluation summary:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py resume --stage stage1
```

You can also skip completed trials with the normal command:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage1 --skip-completed
```

Summarize:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py summarize --stage stage1
```

## Stage 2

Generate architecture configs from the best 3 Stage 1 trials:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py materialize --stage stage2 --top-k 3
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage2
python analysis/temporal_bottleneck_tuning/run_tuning.py summarize --stage stage2
```

## Stage 3

Generate time-window configs from the best 2 Stage 2 trials:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py materialize --stage stage3 --top-k 2
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage3
python analysis/temporal_bottleneck_tuning/run_tuning.py summarize --stage stage3
```

The extra repeated-final control uses `[0, 0, 0, 0, 0]` to compare against the real recent-history window `[-4, -3, -2, -1, 0]` with the same sequence length. Run only those controls with:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage3 --trials trial_13_arch_07_t0x5,trial_06_arch_07_t0x5
python analysis/temporal_bottleneck_tuning/run_tuning.py summarize --stage stage3
```

## Stage 4

Rerun the best Stage 3 config with three seeds:

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py materialize --stage stage4 --top-k 1
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage4
python analysis/temporal_bottleneck_tuning/run_tuning.py summarize --stage stage4
```

## Run Selected Trials

```bash
python analysis/temporal_bottleneck_tuning/run_tuning.py train-evaluate --stage stage1 --trials trial_01,trial_02
```

## Outputs

Configs:

```text
analysis/temporal_bottleneck_tuning/configs/stage1
analysis/temporal_bottleneck_tuning/configs/stage2
analysis/temporal_bottleneck_tuning/configs/stage3
analysis/temporal_bottleneck_tuning/configs/stage4
```

Model outputs:

```text
analysis/temporal_bottleneck_tuning/outputs/stage1/<trial>
analysis/temporal_bottleneck_tuning/outputs/stage2/<trial>
analysis/temporal_bottleneck_tuning/outputs/stage3/<trial>
analysis/temporal_bottleneck_tuning/outputs/stage4/<trial>
```

Summary CSV files:

```text
analysis/temporal_bottleneck_tuning/outputs/stage1/summary.csv
analysis/temporal_bottleneck_tuning/outputs/stage2/summary.csv
analysis/temporal_bottleneck_tuning/outputs/stage3/summary.csv
analysis/temporal_bottleneck_tuning/outputs/stage4/summary.csv
```
