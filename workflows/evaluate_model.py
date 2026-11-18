#!/usr/bin/env python3
"""Evaluate a trained SubmesoFluxNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from submesofluxnet.config import load_config
from submesofluxnet.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
