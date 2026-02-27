# Comparison Test Log

A structured record of comparison experiments and key results.

---

## TablUNet VS UNet-LSTMosticonfigperimedata/ Cmodelagnpixel r2finclimate r2Samcomments / Notes |
|---|---|---|---|--cfg_1|
 one-layer, 5844witUNeturf0.5047n r0.9514ste7 features checcfg_2 |one layer, 5844EQ-UNet + LSTM T = 14| T0.5132| T0.9530| T7 features, the train is higher than UNet| T
UNet-LSTM is better than UNet
BD | TBD |

---
5844 data VS 6454 datas | config ID | data | model | pixel r2 | climate r2 | comments | Result / Notes |
|---|---| cfg_1 |  one-layer, 5844 | UNet | 0.5047 | 0.9514 | 7 features |
| cfg_1 |  one layer, 6454 | UNet | 0.4565 | 0.9367 | 7 features | | TBD | TBD | TBD |128 filter VS 256 filterCo| config ID | data | model | pixel r2 | climate r2 | comments |E/etc.) | Result / Notes || cfg_1 |  one layer, 6454 | UNet 128| 0.4565 | 0.9367 | 7 features |
| cfg_4 |  one layer, 6454 | UNet 256| 0.4771 | 0.9286 | 7 features |

## Table 4 — Transpose Decoder VS Bilinear Decoder VS Pixel Shuffle Decoder

| config ID | data | model | pixel r2 | climate r2 | comments |
|---|---|---|---|---|---|
| cfg_4 |  one layer, 6454 | UNet 256| 0.4771 | 0.9286 | 7 features |
| cfg_7 |  one layer, 6454 | UNet 256| 0.4867 | 0.9286 | 7 features |
| cfg_5 |  one layer, 6454 | UNet 256| 0.4664 | 0.377 | 7 features |TBD | TBD | TBD | TBD | TBD |