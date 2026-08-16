# model_analysis top-90 last-decoder PCA workflow

This folder is independent from the old `last_decoder_pca` outputs. It is configured for the new `analysis/model_analysis` checkpoint. By default it selects the top 90 `model_analysis` test samples ranked by per-snapshot R2.

Important: this workflow does **not** average first. It calculates physics diagnostics and last-decoder PCA for each selected sample first, then sign-aligns and averages the resulting PCA maps/statistics/correlations.

Run order:

```bash
cd /scratch/jczhang/submeso_data_driven/_src/analysis/explanability/decoder_channel_ablation/model_analysis_last_decoder_pca

# 1. Select top 90 test samples and save per-sample physics diagnostics.
python prepare_physics_variables.py --top-n 90 --overwrite

# 2. Capture final decoder features for those same 90 samples.
#    PCA is calculated independently per sample, then averaged.
python last_decoder_pca_analysis.py

# 3. Plot averaged PCA modes with ranked averaged physics diagnostics/correlations.
python plot_pca_first8_with_physics_diagnostics.py
```

Main outputs:

- `physics_variables_top90/physics_variables.npz` with per-sample arrays shaped `(90, 20, 40)`
- `physics_variables_top90/selected_samples.csv`
- `last_decoder_pca/last_decoder_pca_results.npz`
- `last_decoder_pca/last_decoder_pca_first8_maps.png`
- `last_decoder_pca/last_decoder_pca_physics_correlations.csv`
- `final_assessment/pca_first8_with_physics_diagnostics.png`
- `final_assessment/pca_physics_diagnostic_order.csv`

No `speed` diagnostic is saved or correlated. `grad_speed` is kept and derived from U and V for each sample. Density is included as `rho` and `density`.

## Second-best single-sample run

The second-best `model_analysis` test sample is:

- `sample_index=1374`
- `time_index=9175536`
- `R2=0.9715155121`

Run it in separate output folders so it does not overwrite the top-90 outputs:

```bash
cd /scratch/jczhang/submeso_data_driven/_src/analysis/explanability/decoder_channel_ablation/model_analysis_last_decoder_pca

python prepare_physics_variables.py \
  --sample-index 1374 \
  --output-dir physics_variables_sample_1374 \
  --overwrite

python last_decoder_pca_analysis.py \
  --sample-index 1374 \
  --physics physics_variables_sample_1374/physics_variables.npz \
  --output-dir last_decoder_pca_sample_1374 \
  --feature-cache last_decoder_pca_sample_1374/last_decoder_all_channel_feature_maps.npz

python plot_pca_first8_with_physics_diagnostics.py \
  --pca last_decoder_pca_sample_1374/last_decoder_pca_results.npz \
  --physics physics_variables_sample_1374/physics_variables.npz \
  --output final_assessment_sample_1374/pca_first8_with_physics_diagnostics.png \
  --rank-csv final_assessment_sample_1374/pca_physics_diagnostic_order.csv
```

## Best single-sample run

The best `model_analysis` test sample is:

- `sample_index=1379`
- `time_index=9184176`
- `R2=0.9725904129`

Run it in separate output folders:

```bash
cd /scratch/jczhang/submeso_data_driven/_src/analysis/explanability/decoder_channel_ablation/model_analysis_last_decoder_pca

python prepare_physics_variables.py \
  --sample-index 1379 \
  --output-dir physics_variables_sample_1379 \
  --overwrite

python last_decoder_pca_analysis.py \
  --sample-index 1379 \
  --physics physics_variables_sample_1379/physics_variables.npz \
  --output-dir last_decoder_pca_sample_1379 \
  --feature-cache last_decoder_pca_sample_1379/last_decoder_all_channel_feature_maps.npz

python plot_pca_first8_with_physics_diagnostics.py \
  --pca last_decoder_pca_sample_1379/last_decoder_pca_results.npz \
  --physics physics_variables_sample_1379/physics_variables.npz \
  --output final_assessment_sample_1379/pca_first8_with_physics_diagnostics.png \
  --rank-csv final_assessment_sample_1379/pca_physics_diagnostic_order.csv
```
