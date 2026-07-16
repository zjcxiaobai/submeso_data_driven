#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from calc_no_shuffle_spatial_corr import main


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = PROJECT_DIR / "analysis/spatial_corr"


def add_default_args() -> None:
    defaults = {
        "--predictions": PROJECT_DIR / "outputs_train_on_small_domain/upper_40x20_encoder_convlstm_shuffle_all/test_predictions_denorm.npz",
        "--csv-output": HERE / "shuffle_all_encoder_convlstm_spatial_corr.csv",
        "--summary-output": HERE / "shuffle_all_encoder_convlstm_spatial_corr_summary.txt",
        "--plot-output-prefix": HERE / "shuffle_all_encoder_convlstm_spatial_corr",
    }
    present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    for key, value in defaults.items():
        if key not in present:
            sys.argv.extend([key, str(value)])


if __name__ == "__main__":
    add_default_args()
    main()
