#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from plot_fluxnet_fk_spatial_corr import main


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = PROJECT_DIR / "analysis/spatial_corr"


def add_default_args() -> None:
    defaults = {
        "--fluxnet-csv": HERE / "shuffle_all_encoder_convlstm_spatial_corr.csv",
        "--fk-csv": HERE / "fk_shuffle_all_Ce2p13062237_spatial_corr.csv",
        "--output-prefix": HERE / "shuffle_all_fluxnet_fk_spatial_corr",
        "--summary-output": HERE / "shuffle_all_fluxnet_fk_spatial_corr_summary.txt",
        "--fk-display-label": "MLE",
    }
    present = set(arg for arg in sys.argv[1:] if arg.startswith("--"))
    for key, value in defaults.items():
        if key not in present:
            sys.argv.extend([key, str(value)])


if __name__ == "__main__":
    add_default_args()
    main()
