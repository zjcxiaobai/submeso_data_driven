# FK predictions on the final model_analysis test split

This run regenerates FK predictions on the same held-out raw-time-block test samples used by the final temporal-bottleneck model in `analysis/model_analysis`.

From `_src`, run:

```bash
python analysis/fk_mle/upper_40x20/fk_evaluation_domain.py
```

The default FK parameters are `C_e=2.25895966` and `lf_min=1374.0`, matching the best values recorded in `Ce_base_case_calculation/ce_random_search_summary.txt` and `LfMin_test/lfmin_random_search_summary.txt`.

The script reads:

```text
results/final_model/test_predictions_denorm.npz
analysis/model_analysis/data/time_index.npy
```

and writes:

```text
analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_progress_fullgrid.npz
analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_prediction_physical.npz
analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_r2_series.csv
analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_per_snapshot_r2.png
analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_per_snapshot_r2_summary.txt
analysis/fk_mle/upper_40x20/final_model_analysis_test/fk_per_snapshot_r2_stats.csv
```

The calculation is resumable. If it stops, rerun the same command and it will continue from `fk_progress_fullgrid.npz`.

For a short test run:

```bash
python analysis/fk_mle/upper_40x20/fk_evaluation_domain.py --max-snapshots 20 --allow-partial-output
```

After all FK predictions already exist, regenerate only the plot/stat files with:

```bash
python analysis/fk_mle/upper_40x20/fk_evaluation_domain.py --plot-only
```
