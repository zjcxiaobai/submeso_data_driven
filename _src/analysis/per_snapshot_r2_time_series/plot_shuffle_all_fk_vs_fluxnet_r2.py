#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from plot_fk_vs_fluxnet_r2 import main


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "analysis/per_snapshot_r2_time_series"
FK_DIR = PROJECT_DIR / "analysis/fk_mle/upper_40x20"


def add_default_args() -> None:
    defaults = {
        "--fluxnet-r2": PROJECT_DIR
        / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_r2_series.csv",
        "--fk-r2": FK_DIR / "fk_shuffle_all_test_Ce2p13062237_r2_series.csv",
        "--output-prefix": ANALYSIS_DIR / "shuffle_all_fk_vs_submesofluxnet_per_snapshot_r2",
        "--summary-output": ANALYSIS_DIR / "shuffle_all_fk_vs_submesofluxnet_per_snapshot_r2_summary.txt",
        "--stats-csv": ANALYSIS_DIR / "shuffle_all_fk_vs_submesofluxnet_per_snapshot_r2_stats.csv",
        "--fk-display-label": "MLE",
    }
    present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    for key, value in defaults.items():
        if key not in present:
            sys.argv.extend([key, str(value)])


if __name__ == "__main__":
    add_default_args()
    main()
