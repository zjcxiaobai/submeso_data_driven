# Expanded Upper 40x20 Shuffle-All Test

This repeats the previous best upper_40x20 shuffle-all model with the expanded `data/mld` dataset.

It uses the saved best config settings from:

```text
_src/configs/unet_encoder_convlstm_shuffle_all_upper_40x20.yaml
```

Key settings:

```text
model: UNet_3Layers_EncoderConvLSTM
temporal_offsets: [-8, -7, -6, -5, -4, -3, -2, -1, 0]
in_channels: 72
base_channels: 128
dropout: 0.25
loss: huber
lr: 0.0008306357642686378
split.shuffle: true
```

Feature order:

```text
U, V, W, zeta, strain, T, speed, kpphbl
```

Target:

```text
wT
```

Crop:

```text
y=30:50, x=10:50  ->  upper_40x20, shape (20, 40)
```

## Build Dataset

```bash
cd /scratch/jczhang/submeso_data_driven
source ~/.virtualenvs/JC_Ocean/bin/activate
python _src/analysis/shuffle_all_upper40x20_mld_kpphbl_12231/build_dataset.py
```

Expected raw dataset shape:

```text
X: (12231, 8, 20, 40)
y: (12231, 1, 20, 40)
```

With the 9-step temporal window, valid center samples are `12223`.

## Train

```bash
cd /scratch/jczhang/submeso_data_driven/_src
source ~/.virtualenvs/JC_Ocean/bin/activate
python train.py --config analysis/shuffle_all_upper40x20_mld_kpphbl_12231/config.yaml
```

## Evaluate

```bash
python evaluate.py --config analysis/shuffle_all_upper40x20_mld_kpphbl_12231/config.yaml
```
