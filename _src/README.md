# PyTorch U-Net Regression

This folder is a self-contained training pipeline for `data/X.npy` and `data/y.npy`.

Current data shape:

```text
X: (N, 8, 56, 56)
y: (N, 1, 56, 56)
```

## Train

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python train.py --config configs/unet_regression.yaml
```

To continue from `train.last_checkpoint` after an interrupted run:

```bash
python train.py --config configs/unet_regression.yaml --resume
```

To resume from a specific checkpoint:

```bash
python train.py --config configs/unet_regression.yaml --resume outputs/checkpoints/last.pt
```

For one node with multiple GPUs, use PyTorch DistributedDataParallel via torchrun. For example, with 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/unet_regression.yaml
```

For Slurm, request one task per GPU and launch the same script with the cluster-supported `torchrun`/`srun` pattern. Only rank 0 writes checkpoints and history.

## Evaluate

```bash
python evaluate.py --config configs/unet_regression.yaml
```

## Predict Selected Samples

```bash
python predict.py --indices 0 1 2
```

## Plot Training History

```bash
python plot_history.py
```

## Adding Another Model

Create a new file in `models/`, register the model:

```python
from .registry import register_model

@register_model("my_model")
class MyModel(nn.Module):
    ...
```

Then change only the config:

```yaml
model:
  name: my_model
  ...
```
