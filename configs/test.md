# Comparison Test Log

A structured record of comparison experiments and key results.

---

## Table 1 — UNet vs UNet-LSTM

| config ID | data              | model             | pixel r2 | climate r2 | comments                                  |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 5844     | UNet              | 0.5047   | 0.9514     | 7 features;UVWT&vorticity&strain&speed     |
| cfg_2 | one layer, 5844     | UNet + LSTM (T=14)| 0.5132   | 0.9530     | 7 features; train higher than UNet        |

---

## Table 2 — 5844 data vs 6454 data

| config ID | data              | model | pixel r2 | climate r2 | comments    |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 5844     | UNet  | 0.5047   | 0.9514     | 7 features  |
| cfg_1 | one layer, 6454     | UNet  | 0.4565   | 0.9367     | 7 features  |

---

## Table 3 — 128 filters vs 256 filters

| config ID | data              | model      | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 6454     | UNet (128) | 0.4565   | 0.9367     | 7 features |
| cfg_4 | one layer, 6454     | UNet (256) | 0.4771   | 0.9286     | 7 features |

---

## Table 4 — Transpose decoder vs Bilinear decoder vs Pixel Shuffle decoder

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_4 | one layer, 6454     | UNet (256, bilinear)     | 0.4771   | 0.9286     | 7 features |
| cfg_7 | one layer, 6454     | UNet (256, pixel shuffle)      | 0.4867   | 0.9386     | 7 features |
| cfg_5 | one layer, 6454     | UNet (256, transpose) | 0.4664   | 0.3770     | 7 features |

## Table 5 — Pixel shuffle decoder: UNet VS UNet + LSTM (T = 14)

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_8 | one layer, 6454     | UNet (256, pixel shuffle) + LSTM    | 0.4884   | 0.9324     | 7 features |
| cfg_7 | one layer, 6454     | UNet (256, pixel shuffle)      | 0.4867   | 0.9286     | 7 features |

## Table 6 — conf_9 is same as conf_2 but more test data, show how imageR2 varies with time

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_2 | one layer, 5844     | UNet + LSTM (T=14)| 0.5132   | 0.9530     | 7 features       |
| cfg_9 | one layer, 6454, 30% test    | UNet + LSTM (T=14)    | 0.5041   | 0.9704     | 7 features |