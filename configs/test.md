# Comparison Test Log

A structured record of comparison experiments and key results.

---

## Table 1 — UNet vs UNet-LSTM

| config ID | data              | model             | pixel r2 | climate r2 | comments                                  |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 5844     | UNet              | 0.5047   | 0.9514     | 7 features                                |
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
| cfg_4 | one layer, 6454     | UNet (256, transpose)     | 0.4771   | 0.9286     | 7 features |
| cfg_7 | one layer, 6454     | UNet (256, bilinear)      | 0.4867   | 0.9286     | 7 features |
| cfg_5 | one layer, 6454     | UNet (256, pixel shuffle) | 0.4664   | 0.3770     | 7 features |TBD | TBD | TBD | TBD | TBD |