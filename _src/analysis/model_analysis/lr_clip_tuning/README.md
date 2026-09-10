# Learning-rate and gradient-clipping test

This folder tests the current `analysis/model_analysis/config.yaml` model without modifying the final model or `outputs_final`.

The runner keeps the current architecture fixed and varies:

- initial learning rate: `5.6e-4`, `1.0e-3`, `1.8e-3`, `3.2e-3`
- gradient clipping norm: `0`, `1`, `5`
- learning-rate scheduler: default `ReduceLROnPlateau`

`gradient_clip_norm: 0` means no gradient clipping. The scheduler uses validation loss with factor `0.5`, patience `10`, and minimum learning rate `1e-5`.

Run from `_src`:

```bash
source ../setup
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py train-evaluate
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py summarize
```


Resume an interrupted run:

```bash
source ../setup
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py resume-evaluate
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py summarize
```

Use the same `--lrs`, `--clip-norms`, and `--scheduler` arguments that you used for the interrupted run. The resume command skips configs that already have `evaluation_summary.txt`, continues unfinished configs from `training_state.pt`, and starts from epoch 1 only for unfinished configs that do not yet have a resume state. Evaluation still uses the best-validation checkpoint `temporal_bottleneck.pt`. Add `--rerun-completed` only when you intentionally want to recompute completed configs.

Run a smaller test:

```bash
source ../setup
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py train-evaluate --lrs 0.0018 --clip-norms 0 1 5
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py summarize --lrs 0.0018 --clip-norms 0 1 5
```

Compare against no scheduler:

```bash
source ../setup
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py train-evaluate --scheduler none --lrs 0.0018 --clip-norms 0 1 5
python analysis/model_analysis/lr_clip_tuning/run_lr_clip_tuning.py summarize --scheduler none --lrs 0.0018 --clip-norms 0 1 5
```

Outputs are written under:

```text
analysis/model_analysis/lr_clip_tuning/outputs/
analysis/model_analysis/lr_clip_tuning/lr_clip_summary.csv
```
