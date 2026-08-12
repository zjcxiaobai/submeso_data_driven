# Shuffle-All Upper 40x20, Density Replaces Speed

This test uses the same shuffle-all upper_40x20 model/config pattern as:

```text
_src/analysis/shuffle_all_upper40x20_mld_kpphbl_12231/config.yaml
```

The input data comes from:

```text
src/data/mld_rho_no_speed
```

Feature order:

```text
U, V, W, zeta, strain, T, rho, kpphbl
```

So this is an 8-feature test where `speed` is replaced by `rho`.
The ConvLSTM uses 9 time offsets, so the model input channel count is:

```text
8 features * 9 times = 72 in_channels
```

Dataset crop:

```text
y=30:50, x=10:50
```

Train from `_src` with:

```bash
python train.py --config analysis/shuffle_upper40x20_mld_rho_9features/config.yaml
```
