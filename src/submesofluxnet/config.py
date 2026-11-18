"""Configuration and project-path helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def project_dir() -> Path:
    """Return the repository root for an editable source-tree installation."""
    return Path(__file__).resolve().parents[2]


def resolve(root: Path, value: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> dict:
    """Load a YAML experiment configuration."""
    with path.open("r") as handle:
        return yaml.safe_load(handle)
