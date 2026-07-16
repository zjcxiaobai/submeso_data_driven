#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from calculate_test_fk_best_ce_and_plot import main


PROJECT_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_DIR / "analysis/fk_mle/upper_40x20"


def add_default_args() -> None:
    defaults = {
        "--predictions": PROJECT_DIR
        / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_predictions_denorm.npz",
        "--progress": OUT_DIR / "fk_shuffle_all_test_Ce2p13062237_progress_fullgrid.npz",
        "--output": OUT_DIR / "fk_shuffle_all_test_Ce2p13062237_prediction_physical.npz",
        "--r2-csv": OUT_DIR / "fk_shuffle_all_test_Ce2p13062237_r2_series.csv",
        "--output-prefix": OUT_DIR / "fk_shuffle_all_test_Ce2p13062237_per_snapshot_r2",
        "--summary-output": OUT_DIR / "fk_shuffle_all_test_Ce2p13062237_per_snapshot_r2_summary.txt",
        "--stats-csv": OUT_DIR / "fk_shuffle_all_test_Ce2p13062237_per_snapshot_r2_stats.csv",
    }
    present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    for key, value in defaults.items():
        if key not in present:
            sys.argv.extend([key, str(value)])
    for flag in ("--skip-missing-nc", "--allow-partial-output"):
        if flag not in present:
            sys.argv.append(flag)


if __name__ == "__main__":
    add_default_args()
    main()
