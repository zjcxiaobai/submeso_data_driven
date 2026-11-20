#!/usr/bin/env python3
"""Export four clean bottleneck maps for the architecture schematic."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
OUTPUT_DIR = PROJECT_ROOT / "results/analysis_outputs/interpretability/bottle_feature/architecture_feature_maps"


def main() -> None:
    with np.load(INPUT) as data:
        latent = data["latent_features"].astype(float)
        r2 = data["r2"].astype(float)

    features = latent[int(np.nanargmax(r2))]
    limit = float(np.nanpercentile(np.abs(features), 99))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for feature, image in enumerate(features, start=1):
        height, width = image.shape
        fig = plt.figure(figsize=(width / 600, height / 600), dpi=600, frameon=False)
        axis = fig.add_axes([0, 0, 1, 1])
        axis.imshow(image, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        axis.set_axis_off()
        fig.savefig(
            OUTPUT_DIR / f"bottleneck_feature_{feature}.png",
            dpi=600,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    print(f"Saved four architecture feature maps to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
