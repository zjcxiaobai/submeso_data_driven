# ResUNet Regression

This folder mirrors `_src` but registers a residual U-Net model for testing map
learning on the same `data/X.npy` and `data/y.npy` arrays. The `data` directory
is a symlink to `_src/data` to avoid duplicating the large files.

## Train

```bash
cd /scratch/jczhang/submeso_data_driven/ResUNet
python train.py --config configs/resunet_no_shuffle.yaml
```

For a same-distribution sanity check:

```bash
python train.py --config configs/resunet_shuffle_train_val.yaml
```

## Evaluate

```bash
python evaluate.py --config configs/resunet_no_shuffle.yaml
```
