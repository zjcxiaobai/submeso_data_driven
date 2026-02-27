# Comparison Test Log

A structured record of comparison experiments and key results.

---

## Table 1 — Equilibrium / Spin-up Diagnostics

| Experiment ID | Run / Config | Diagnostic | Definition | Time Range / Sampling | Result / Notes |
|---|---|---|---|---|---|
| EQ-01 | 50km_noGM (MITgcm_50km_withoutGM) | Surface EKE | EKE = mean(0.5*(u'²+v'²)); u',v' = anomaly (current: domain-mean removed) | iters 0→2332800 step 25920 | EKE saved to `50km_noGM_eke.pkl`; plateau check TBD |
| EQ-02 | TBD | TBD | TBD | TBD | TBD |

---

## Table 2 — Physics Field Comparison (Snapshots / Statistics)

| Experiment ID | Run A | Run B | Variable / Depth | Comparison Method | Result / Notes |
|---|---|---|---|---|---|
| PHY-01 | TBD | TBD | TBD | TBD (e.g., snapshot diff, mean/var, PDF, spectra) | TBD |
| PHY-02 | TBD | TBD | TBD | TBD | TBD |

---

## Table 3 — ML / Parameterization Comparison (Model Skill)

| Experiment ID | Model / Method | Inputs → Target | Train/Test Split | Metrics (R²/MSE/etc.) | Result / Notes |
|---|---|---|---|---|---|
| ML-01 | TBD | TBD | TBD (time-ordered? gap?) | TBD | TBD |
| ML-02 | TBD | TBD | TBD | TBD | TBD |