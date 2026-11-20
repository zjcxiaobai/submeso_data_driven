# Temporal-average correlations

The analysis retains the seven physical diagnostics shown in the paper figure:

```text
shear strain, potential temperature, Rossby number, vertical velocity,
horizontal temperature-gradient magnitude, temperature frontogenesis,
and normal strain
```

`physics_variable_test_series.npz` stores these fields for the test samples.
`temporal_average_correlation_results.npz` stores the resulting mean absolute spatial correlations.

Running `temporal_average_correlation.py` uses the saved correlation results when available and writes only the combined paper figure to:

```text
results/analysis_outputs/interpretability/bottle_feature/temporal_average_correlation/
```

Delete `temporal_average_correlation_results.npz` before running the script only when the correlations must be recalculated from the retained physics cache.
