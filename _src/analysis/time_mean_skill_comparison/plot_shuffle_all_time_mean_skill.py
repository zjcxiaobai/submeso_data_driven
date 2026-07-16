#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from plot_time_mean_skill import main


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = PROJECT_DIR / "analysis/time_mean_skill_comparison"
FK_DIR = PROJECT_DIR / "analysis/fk_mle/upper_40x20"


def add_default_args() -> None:
    defaults = {
        "--fluxnet-predictions": PROJECT_DIR / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_predictions_denorm.npz",
        "--fk-predictions": FK_DIR / "fk_shuffle_all_test_Ce2p13062237_prediction_physical.npz",
        "--output-prefix": HERE / "shuffle_all_time_mean_skill_comparison",
        "--csv-output": HERE / "shuffle_all_time_mean_skill_comparison.csv",
        "--summary-output": HERE / "shuffle_all_time_mean_skill_comparison_summary.txt",
        "--fk-display-label": "MLE",
    }
    present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    for key, value in defaults.items():
        if key not in present:
            sys.argv.extend([key, str(value)])


if __name__ == "__main__":
    add_default_args()
    main()
