from pathlib import Path
from plot_no_shuffle_overfitting_r2 import draw_figure, finite_summary, lowest_window, read_r2_series, rolling_median
from plot_no_shuffle_overfitting_r2 import write_stats_csv, write_summary


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "analysis/per_snapshot_r2_time_series"


def main() -> None:
    r2_series = PROJECT_DIR / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_r2_series.csv"
    output_prefix = OUT_DIR / "shuffle_all_encoder_convlstm_per_snapshot_r2"
    summary_output = OUT_DIR / "shuffle_all_encoder_convlstm_per_snapshot_r2_summary.txt"
    stats_csv = OUT_DIR / "shuffle_all_encoder_convlstm_per_snapshot_r2_stats.csv"
    rolling_window = 81

    _, sample_index, r2 = read_r2_series(r2_series)
    summary = finite_summary(r2)
    smooth = rolling_median(r2, rolling_window)
    low_start, low_end, low_mean = lowest_window(sample_index, r2, rolling_window)

    write_summary(
        summary_output,
        summary,
        int(sample_index.min()),
        int(sample_index.max()),
        low_start,
        low_end,
        low_mean,
    )
    write_stats_csv(stats_csv, summary, low_start, low_end, low_mean)
    draw_figure(sample_index, r2, smooth, summary, low_start, low_end, output_prefix)

    print(f"Saved {output_prefix.with_suffix('.png')}", flush=True)
    print(f"Saved {summary_output}", flush=True)
    print(f"Saved {stats_csv}", flush=True)
    print(f"Mean R2: {summary['mean']:.6f}", flush=True)


if __name__ == "__main__":
    main()
