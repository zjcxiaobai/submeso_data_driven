# SubmesoFluxNet

SubmesoFluxNet is a spatiotemporal convolutional neural network for predicting
submesoscale vertical heat flux from coarse-resolution ocean variables. The repository contains the MITgcm preprocessing workflow, the spatiotemporal neural network, evaluation utilities, and the analyses of the results.

## Scientific workflow

```text
MITgcm output
    |
    v
12-hour temporal averages and 50 km spatial coarsening
    |
    v
Mixed-layer features and vertical heat flux target
    |
    v
Chronological train/validation/test split
    |
    v
Six-snapshot temporal windows
    |
    v
SubmesoFluxNet training and evaluation
    |
    v
Skill, generalization, and interpretability analyses
```

Temporal blocks are split before sequence construction. Consequently, no input
window crosses a training, validation, or test boundary.

## Model

[View the SubmesoFluxNet model schematic (PDF)](analysis/architecture/SubmesoFluxNet.pdf)

Each model sample contains six consecutive snapshots at offsets
`[-5, -4, -3, -2, -1, 0]`. A snapshot consists of eight mixed-layer fields on a
`20 x 40` grid:

1. zonal velocity (`U`);
2. meridional velocity (`V`);
3. vertical velocity (`W`);
4. relative vorticity (`zeta`);
5. total strain (`strain`);
6. potential temperature (`T`);
7. density (`rho`); and
8. KPP boundary-layer depth (`kpphbl`).

Convolutional blocks use batch normalization, SiLU activation, and a dropout
rate of 0.15. The complete experiment definition is stored in
[`config/config.yaml`](config/config.yaml).

## Repository layout

```text
submeso_data_driven/
├── config/                     ML configuration
├── data/
│   ├── interim/                Intermediate MITgcm-derived fields
│   └── processed/              Model-ready data
├── scripts/
│   ├── MITgcm_processing/      MITgcm-to-feature processing
│   └── ML_processing/          Dataset assembly and subdomain extraction
├── src/submesofluxnet/         Model, data loading, training, and evaluation
├── workflows/                  User-facing training and evaluation commands
├── analysis/
│   ├── architecture/           Architecture schematic
│   ├── fk_mle/                 MLE tuning and evaluation
│   ├── generalization/         Non-local generalization analyses
│   ├── interpretability/       Sensitivity, IG, and bottleneck diagnostics
│   ├── model_skill/            Prediction-skill analyses
│   └── ocean_simulation/       MITgcm configuration schematic
├── results/
│   ├── model/                  Preserved final-model artifacts
│   └── analysis_outputs/       Generated figures
├── utils/                      MITgcm grid and physical-diagnostic utilities
└── pyproject.toml              Package metadata and core dependencies
```

Large numerical datasets and model checkpoints are intentionally excluded from
Git.


## Data preparation

### Model-ready data contract

The final upper-domain dataset is expected at:

```text
data/processed/upper/X.npy
data/processed/upper/y.npy
data/processed/upper/time_index.npy
```

| File | Shape | Description |
| --- | --- | --- |
| `X.npy` | `(time, 8, 20, 40)` | Eight input fields at each saved time |
| `y.npy` | `(time, 1, 20, 40)` | Submesoscale vertical heat-flux target |
| `time_index.npy` | `(time,)` | MITgcm iteration associated with each sample |

## Training

The committed configuration uses an 80/10/10 chronological split, Huber loss,
AdamW optimization, a validation-loss learning-rate scheduler, and early
stopping. Normalization statistics are calculated from the training block only.

Start a new run:

```bash
python workflows/train_model.py --config config/config.yaml
```

Resume an interrupted run from `results/model_test/latest.pt`:

```bash
python workflows/train_model.py --config config/config.yaml --resume
```

## Analysis and figures

Analysis scripts are grouped under `analysis/`. Final figures are written beneath:

```text
results/analysis_outputs/
```

Most plotting scripts consume saved prediction or diagnostic files and do not
retrain the model. Recompute upstream diagnostics only when the model,
normalization, dataset, or chronological split changes.
