#!/usr/bin/env python3
"""Plot the four bottleneck maps from the best test sample."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT = PROJECT_ROOT / "results/model/test_predictions_denorm.npz"
OUTPUT = PROJECT_ROOT / "results/analysis_outputs/interpretability/bottle_feature/bottleneck_4_features_best.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
    }
)


def main() -> None:
    with np.load(INPUT) as data:
        latent = data["latent_features"].astype(float)
        sample_index = data["sample_index"].astype(int)
        r2 = data["r2"].astype(float)

    position = int(np.nanargmax(r2))
    features = latent[position]
    limit = float(np.nanpercentile(np.abs(features), 99))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95), constrained_layout=True)
    for feature, axis in enumerate(axes):
        image = axis.imshow(features[feature], origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest")
        axis.set_title(f"Feature {feature + 1}", fontsize=8, pad=3)
        axis.set_xlabel("x index")
        axis.set_xticks([0, features.shape[2] // 2, features.shape[2] - 1])
        axis.set_yticks([0, features.shape[1] // 2, features.shape[1] - 1])
        if feature == 0:
            axis.set_ylabel("y index")
        else:
            axis.set_yticklabels([])

    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.018)
    colorbar.set_label("Latent activation")
    fig.suptitle(
        f"Bottleneck feature maps: sample {sample_index[position]}, $R^2$={r2[position]:.3f}",
        fontsize=9,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
