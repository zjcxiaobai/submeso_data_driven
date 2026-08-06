# Shuffle-All Rerun With Expanded MLD Data

This reruns the original `unet_shuffle_all` test with the same model/settings, but builds `X.npy` and `y.npy` from the current `data/mld/*.npz` files.

Feature order is unchanged:

```text
U, V, W, zeta, strain, T, speed, kpphbl
```

Target:

```text
wT
```

## Build Dataset

```bash
cd /scratch/jczhang/submeso_data_driven
source ~/.virtualenvs/JC_Ocean/bin/activate
python _src/analysis/shuffle_all_mld_kpphbl_12231/build_dataset.py
```

Expected output shape after the current data update:

```text
X: (12231, 8, 56, 56)
y: (12231, 1, 56, 56)
```

## Train Same Shuffle-All Model

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python train.py --config analysis/shuffle_all_mld_kpphbl_12231/config.yaml
```

## Evaluate

```bash
cd /scratch/jczhang/submeso_data_driven/_src
python evaluate.py --config analysis/shuffle_all_mld_kpphbl_12231/config.yaml
```
