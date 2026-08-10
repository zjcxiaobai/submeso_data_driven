#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from plot_best_worst_snapshot_maps import draw_figure, read_r2_rows, select_cases, write_summary


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "analysis/per_snapshot_r2_time_series"


def main() -> None:
    predictions = PROJECT_DIR / "analysis/shuffle_upper40x20_mld_rho/outputs_split_80_10_10/test_predictions_denorm.npz"
    r2_series = PROJECT_DIR / "analysis/shuffle_upper40x20_mld_rho/outputs_split_80_10_10/test_r2_series.csv"
    output_prefix = OUT_DIR / "shuffle_all_mld_rho_split_80_10_10_best_worst_maps"
    summary_output = OUT_DIR / "shuffle_all_mld_rho_split_80_10_10_best_worst_maps_summary.txt"

    r2_rows = read_r2_rows(r2_series)
    best, worst = select_cases(r2_rows, predictions)
    draw_figure(best, worst, output_prefix, percentile=99.0, error_percentile=99.0)
    write_summary(summary_output, best, worst)

    print(f"Best case: sample={best.sample_index}, R2={best.r2:.6f}", flush=True)
    print(f"Worst case: sample={worst.sample_index}, R2={worst.r2:.6f}", flush=True)
    print(f"Saved {output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {summary_output}", flush=True)


if __name__ == "__main__":
    main()
