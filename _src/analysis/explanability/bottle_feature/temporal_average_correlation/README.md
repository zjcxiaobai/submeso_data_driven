## Top-Five Selection and 356-Day Window

Each feature independently selects its top five variables by absolute median
spatial correlation at 356 model days. Signed correlations remain on the y-axis.
Nonfinite final correlations are excluded, and ties follow physical-process order.
The complete 11-variable results remain in the numerical archive. The selected
variables and final correlations are written to top5_physical_variables_by_feature.csv.

The saved cadence is 0.5 model day per snapshot, so 356 days requires 712 snapshots.
The script appends this endpoint to the previous averaging windows. Recalculate
once (the older archive does not contain this endpoint), from _src:

```bash
python analysis/explanability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py --snapshot-days 0.5
```

Afterwards, --replot-only can reuse the extended results. The endpoint is adjustable
with --final-window-days. A final window longer than the series is rejected.

# Feature-Wise Temporal-Average Correlation Plots

Four separate figures show Features 1-4. Each has up to five physical-variable curves
in physical-process order, with consistent colors, markers, and line styles.
The legend is outside the plotting area. X is the averaging window in model days;
y is median spatial Pearson correlation. PCA is not calculated or plotted.
Uncertainty bands and dots are omitted for readability; confidence intervals
remain available in the numerical results.

To replot existing heatmap results without repeating the analysis, run from _src:

```bash
python analysis/explanability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py --replot-only
```

This uses the physical-time values saved with those results. Override with
--snapshot-days only when correcting the snapshot cadence.

Outputs are temporal_average_correlation_feature_1 through _4, as PNG and PDF.
Existing heatmap and per-variable images are not deleted. Numerical result names
retain the bottleneck_physics_heatmaps prefix for compatibility.

## Calculation Details

Both physical fields and bottleneck maps are averaged across consecutive test
samples using trailing windows of 1, 3, 7, 15, 30, 60, 90, 180, and 365 snapshots.
Each plotted point is the median spatial Pearson correlation across all complete
rolling windows. These are not correlations across the model input sequence.

## Physical Time

Supply the verified snapshot interval with --snapshot-days. There is no default:
existing project plots disagree about the cadence. For 12-hour snapshots, run
from _src:

```bash
python analysis/explanability/bottle_feature/temporal_average_correlation/temporal_average_correlation.py --snapshot-days 0.5
```

For daily snapshots, use --snapshot-days 1 instead. The script checks that raw
timestamps are uniformly spaced and sample IDs are consecutive; it does not
infer physical time from simulation iteration numbers.

## Uncertainty

Numerical results include pointwise 95% moving-block-bootstrap percentile confidence intervals
that exclude zero (2000 resamples; seed 42). Contiguous blocks are sampled from
the per-window correlation series, pairing the four features in each draw.
Block length is max(30, averaging-window length), adjustable with --bootstrap-block.
At least five available blocks and adequate finite observations are required.
No interval or dot is reported otherwise; this is particularly relevant at long
windows. Absence of a dot can mean an interval includes zero or insufficient data.
These are pointwise intervals, not multiple-comparison-adjusted significance.
Block length is a configurable dependence assumption; assess sensitivity before
using the markers for inferential claims.

## Outputs

New outputs are written in outputs/:

- temporal_average_correlation_feature_1.png/.pdf through feature_4.png/.pdf
- bottleneck_physics_heatmaps.csv
- bottleneck_physics_heatmaps.npz
- bottleneck_physics_heatmaps_metadata.json

Previous per-variable figures and PCA result files are not regenerated or deleted.
All model-dependent correlations are recalculated. The physics-only cache can be
reused; pass --no-reuse-cache if the raw data or physics settings have changed.
