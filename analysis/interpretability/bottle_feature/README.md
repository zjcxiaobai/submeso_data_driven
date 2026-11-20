# Bottleneck-feature analysis

This folder contains the paper analyses of the four learned bottleneck maps.

- `plot_bottleneck_features.py` plots the best test sample.
- `architecture_feature_maps/` exports the four map images used in the model schematic.
- `temporal_average_correlation/` compares the bottleneck maps with seven physical diagnostics across temporal-averaging windows.

All scripts read the final model output from `results/model/test_predictions_denorm.npz`.
Figures are written under `results/analysis_outputs/interpretability/bottle_feature/`.

Run from the repository root:

```bash
python analysis/interpretability/bottle_feature/plot_bottleneck_features.py
python analysis/interpretability/bottle_feature/architecture_feature_maps/plot_architecture_feature_maps.py
python analysis/interpretability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py
```
