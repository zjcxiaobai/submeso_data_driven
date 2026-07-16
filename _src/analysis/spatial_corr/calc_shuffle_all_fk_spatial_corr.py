#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from calc_fk_spatial_corr import main


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = PROJECT_DIR / "analysis/spatial_corr"
FK_DIR = PROJECT_DIR / "analysis/fk_mle/upper_40x20"


def add_default_args() -> None:
    defaults = {
        "--fk-predictions": FK_DIR / "fk_shuffle_all_test_Ce2p13062237_prediction_physical.npz",
        "--truth-predictions": PROJECT_DIR / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_predictions_denorm.npz",
        "--csv-output": HERE / "fk_shuffle_all_Ce2p13062237_spatial_corr.csv",
        "--summary-output": HERE / "fk_shuffle_all_Ce2p13062237_spatial_corr_summary.txt",
        "--plot-output-prefix": HERE / "fk_shuffle_all_Ce2p13062237_spatial_corr",
    }
    present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    for key, value in defaults.items():
        if key not in present:
            sys.argv.extend([key, str(value)])


if __name__ == "__main__":
    add_default_args()
    main()
