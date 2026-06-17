#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "outputs/history.csv"
OUTPUT = ROOT / "outputs/history.png"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(HISTORY)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(df["epoch"], df["train_loss"], label="train")
    ax.plot(df["epoch"], df["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(f"Saved {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
