"""Load the SubmesoFluxNet configuration."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    with path.open() as file:
        return yaml.safe_load(file)


