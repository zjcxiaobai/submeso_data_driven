# Transformer Regression

This folder mirrors `_src` but registers a patch-transformer model for testing
map regression on the same `data/X.npy` and `data/y.npy` arrays. The `data`
directory is a symlink to `_src/data` to avoid duplicating the large files.

The default transformer uses `4x4` patches, so each `56x56` map becomes a
`14x14` token grid before attention, followed by a convolutional decoder back to
`1x56x56`.

## Train

```bash
cd /scratch/jczhang/submeso_data_driven/Transformer
python train.py --config configs/transformer_no_shuffle.yaml
```

For a same-distribution sanity check:

```bash
python train.py --config configs/transformer_shuffle_train_val.yaml
```

## Evaluate

```bash
python evaluate.py --config configs/transformer_no_shuffle.yaml
```

## Conv-Transformer

This variant keeps convolutional encoder/decoder skip connections and applies
transformer attention only at the `14x14` bottleneck.

```bash
python train.py --config configs/conv_transformer_no_shuffle.yaml
```

Same-distribution sanity check:

```bash
python train.py --config configs/conv_transformer_shuffle_train_val.yaml
```
