#!/usr/bin/env python3
"""Train SubmesoFluxNet from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from submesofluxnet.config import load_config
from submesofluxnet.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
