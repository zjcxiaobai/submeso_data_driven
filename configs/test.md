# Comparison Test Log

A structured record of comparison experiments and key results.

## One Layer (k = 12) - 7 features
---

### Table 1 — UNet-3 vs UNet-LSTM vs UNet-4

| config ID | data              | model             | pixel r2 | climate r2 | comments                                  |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 5844     | UNet 3En3De             | 0.5047   | 0.9514     | 7 features;UVWT&vorticity&strain&speed     |
| cfg_2 | one layer, 5844     | UNet + LSTM (T=14)| 0.5132   | 0.9530     | 7 features; train higher than UNet        |
| cfg_19 | mld avg, 9014     | UNet 4En3De| 0.5562   | 0.9234     | 8 features with kpphbl, val r2 can reach 0.62       |
---

### Table 2 — 5844 data vs 6454 data

| config ID | data              | model | pixel r2 | climate r2 | comments    |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 5844     | UNet  | 0.5047   | 0.9514     | 7 features  |
| cfg_1 | one layer, 6454     | UNet  | 0.4565   | 0.9367     | 7 features  |

---

### Table 3 — 128 filters vs 256 filters

| config ID | data              | model      | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_1 | one layer, 6454     | UNet (128) | 0.4565   | 0.9367     | 7 features |
| cfg_4 | one layer, 6454     | UNet (256) | 0.4771   | 0.9286     | 7 features |

---

### Table 4 — Transpose decoder vs Bilinear decoder vs Pixel Shuffle decoder

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_4 | one layer, 6454     | UNet (256, bilinear)     | 0.4771   | 0.9286     | 7 features |
| cfg_7 | one layer, 6454     | UNet (256, pixel shuffle)      | 0.4867   | 0.9386     | 7 features |
| cfg_5 | one layer, 6454     | UNet (256, transpose) | 0.4664   | 0.3770     | 7 features |

### Table 5 — Pixel shuffle decoder: UNet VS UNet + LSTM (T = 14)

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_8 | one layer, 6454     | UNet (256, pixel shuffle) + LSTM    | 0.4884   | 0.9324     | 7 features |
| cfg_7 | one layer, 6454     | UNet (256, pixel shuffle)      | 0.4867   | 0.9286     | 7 features |

### Table 6 — conf_9 is same as conf_2 but more test data, show how imageR2 varies with time

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_2 | one layer, 5844     | UNet + LSTM (T=14)| 0.5132   | 0.9530     | 7 features       |
| cfg_9 | one layer, 6454, 30% test    | UNet + LSTM (T=14)    | 0.5041   | 0.9704     | 7 features |

---

## Mixed Layer Averaged - 7 Features
---
### Table 7 — MLD average VS One Layer

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_4 | one layer, 6454     | UNet (256) | 0.4771   | 0.9286     | 7 features |
| cfg_10 | mld, 7087    | UNet (256) | 0.5318   | 0.9083     | 7 features |

---

## Mixed Layer Averaged - More Features (kpphbl, Eta, b, buoyancy_gradient)
---

### Table 8 — feature sensitivity

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_10 | mld, 7087    | UNet (256) | 0.5318   | 0.9083     | 7 features |
| cfg_11 | mld, 7087     | UNet (256) | 0.5663   | 0.9400     | 8 features, add kpphbl|
| cfg_13 | mld, 7087     | UNet (256) + pixel shuffel | 0.5722   | 0.9115 | 8 features, add kpphbl|
| cfg_12 | mld, 6635     | UNet (256) | 0.5534   | 0.9225     | 9 features, add kpphbl, Eta|
| cfg_14 | mld, 6635     | UNet (256) | 0.5412   | 0.9181     | 9 features, add kpphbl, buoyancy gradient|

### Table 9 — T sensitivity

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_13 | mld, 7087    | UNet (256) + pixel shuffel | 0.5722   | 0.9115 | 8 features, add kpphbl |
| cfg_15 | mld, 7087    | UNet LSTM (256) + pixel shuffel + T = 1 | 0.5644   | 0.8941 | 8 features, add kpphbl|
| cfg_16 | mld, 7087    | UNet LSTM (256) + pixel shuffel + T = 5 | 0.5703   | 0.9361 | 8 features, add kpphbl|
| cfg_17 | mld, 7087    | UNet LSTM (256) + pixel shuffel + T = 9 | 0.5819   | 0.9375 | 8 features, add kpphbl|
| cfg_18 | mld, 7087    | UNet LSTM (256) + pixel shuffel + T = 20 | 0.   | 0. | out of memory|

### Table 10 — Skewness process of target

| config ID | data              | model                     | pixel r2 | climate r2 | comments   |
|---|---|---|---:|---:|---|
| cfg_17 | mld, 7087 , no asinh  | UNet LSTM (256) + pixel shuffel + T = 9 | 0.5819   | 0.9375 | 8 features, add kpphbl|
| cfg_18 | mld, 7087 , with asinh  | UNet LSTM (256) + pixel shuffel + T = 9 | 0.5504  | 0.9393 | 8 features, add kpphbl|
