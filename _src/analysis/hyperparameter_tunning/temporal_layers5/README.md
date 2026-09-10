# Temporal Layers 5 Test

This test copies the current final `analysis/model_analysis/config.yaml` and changes only the model depth:

```text
model.temporal_layers: 3 -> 5
```

Output paths are redirected into this folder so the base `model_analysis` results are not overwritten.

## Run From `_src`

```bash
cd /scratch/jczhang/submeso_data_driven/_src
source ../setup
python analysis/hyperparameter_tunning/temporal_layers5/run_temporal_layers5.py train-evaluate
```

Resume:

```bash
python analysis/hyperparameter_tunning/temporal_layers5/run_temporal_layers5.py resume
```

Summarize:

```bash
python analysis/hyperparameter_tunning/temporal_layers5/run_temporal_layers5.py summarize
```
