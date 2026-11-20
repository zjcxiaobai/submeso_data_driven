# SubmesoFluxNet

SubmesoFluxNet is a PyTorch model for predicting submesoscale vertical heat flux from coarse-resolution ocean-state variables. The repository contains the MITgcm preprocessing workflow, the spatiotemporal neural network, evaluation utilities, and the analyses of the results.

## Model overview

Each sample contains six consecutive snapshots at temporal offsets `[-5, -4, -3, -2, -1, 0]`. Every snapshot contains eight fields on a `20 x 40` grid:

```text
U, V, W, relative vorticity, strain, potential temperature, density,
and KPP boundary-layer depth
```

The input sequence therefore has shape `(batch, 6, 8, 20, 40)`. A shared two-dimensional encoder processes each snapshot, five temporal three-dimensional convolutional layers combine information across the sequence, and a four-channel spatial bottleneck is decoded into the final heat-flux field.

The raw chronology is divided into contiguous training, validation, and test blocks using an `80/10/10` split. Temporal windows are constructed only after this split, preventing a sequence from crossing a split boundary.

## Repository structure

```text
config/                 Final model and training configuration
data/                   Local raw, intermediate, and processed datasets
scripts/                MITgcm and machine-learning data preparation
src/submesofluxnet/     Reusable model, data, training, and evaluation code
workflows/              Training and evaluation entry points
analysis/               Paper analyses and plotting scripts
results/model/          Preserved final-paper model artifacts
results/model_test/     Default destination for new reproduction runs
results/analysis_outputs/  Generated paper figures
utils/                  MITgcm grid and physical-diagnostic utilities
```

Large datasets, checkpoints, and derived numerical files are excluded from Git. Empty data directories are retained with `.gitkeep` files.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The core package installs NumPy, PyYAML, and PyTorch. Data preprocessing and paper plotting additionally require the scientific packages used by the relevant scripts, including Matplotlib, Xarray, xgcm, h5netcdf, and MITgcmutils.

On the original computing system, the local environment can instead be activated with:

```bash
source setup
```

The `setup` file contains a machine-specific virtual-environment path and should be adjusted on another system.

## Data preparation

The model expects the following processed files:

```text
data/processed/upper/X.npy
data/processed/upper/y.npy
data/processed/upper/time_index.npy
```

`X.npy` has shape `(time, 8, 20, 40)`, while `y.npy` has shape `(time, 1, 20, 40)`. The temporal sequence dimension is assembled by the dataset loader during training; it is not merged into the eight input channels.

The main preparation sequence is:

```bash
python scripts/MITgcm_processing/temporal_spatial_avg/coarse_fields.py
python scripts/MITgcm_processing/build_feature_target_4D_coarse/build_all_npz.py
python scripts/ML_processing/build_full_domain.py
python scripts/ML_processing/build_upper.py
python scripts/ML_processing/build_lower.py
```

The first step requires access to the original MITgcm output and must be configured for its local location. The remaining scripts assemble aligned feature and target arrays and crop the upper and lower `20 x 40` analysis domains.

## Configuration

The default experiment is defined in [`config/config.yaml`](config/config.yaml). It specifies:

- input data and temporal offsets;
- chronological split fractions;
- model channel widths and bottleneck size;
- dropout and optimization settings;
- checkpoint, history, prediction, and evaluation paths.

The committed configuration writes new runs to `results/model_test/`. This protects the preserved paper artifacts under `results/model/` from accidental replacement.

## Training

Start a new training run from the repository root:

```bash
python workflows/train_model.py --config config/config.yaml
```

Resume an interrupted run from `latest.pt`:

```bash
python workflows/train_model.py --config config/config.yaml --resume
```

Training uses a fixed seed, deterministic PyTorch operations, Huber loss, AdamW, validation-loss learning-rate reduction, and early stopping. The lowest-validation-loss checkpoint is retained as `temporal_bottleneck.pt`; `latest.pt` stores the optimizer, scheduler, and random-number states required for resuming.

## Evaluation

Evaluate the retained checkpoint configured in `config/config.yaml`:

```bash
python workflows/evaluate_model.py --config config/config.yaml
```

Evaluation saves denormalized predictions, targets, sample-level metrics, four bottleneck feature maps, and a text summary. Two test-skill quantities are reported:

- `test_r2`: one aggregate coefficient of determination over all test samples and grid cells;
- `mean_snapshot_r2`: the mean of the separately calculated snapshot-level coefficients of determination.

These quantities answer different questions and are not expected to be identical.

## Paper analyses

Analysis scripts are grouped by purpose:

```text
analysis/model_skill/          Prediction-skill comparisons
analysis/generalization/      Non-local generalization tests
analysis/fk_mle/              MLE-parameterization tuning and evaluation
analysis/interpretability/    Ablation, Integrated Gradients, and bottleneck analyses
analysis/ocean_simulation/    Simulation configuration and diagnostics
analysis/architecture/        Model-input and architecture assets
```

Run analysis scripts from the repository root so their project-relative paths resolve correctly. Final figures are written beneath:

```text
results/analysis_outputs/
```

Most plotting scripts reuse retained prediction or analysis archives and do not retrain the model. Calculation scripts should be rerun only when the model, dataset, or sample split changes.

## Reproducing the paper model

For an exact comparison with the paper results, preserve all of the following together:

```text
config/config.yaml
results/model/temporal_bottleneck.pt
results/model/norm_stats.npz
results/model/test_predictions_denorm.npz
results/model/history.csv
results/model/evaluation_summary.txt
```

Because checkpoints and large NumPy archives are ignored by Git, they must be stored or distributed separately. Do not use `git add -f` for the full local data tree.

## Recommended workflow

1. Prepare or obtain the processed upper-domain dataset.
2. Review `config/config.yaml` and confirm all paths.
3. Train into `results/model_test/`.
4. Evaluate the best checkpoint.
5. Compare the new summary with the preserved paper result.
6. Replace the paper artifacts only after the reproduction run has been verified.
7. Regenerate only the analyses affected by a changed model or dataset.
