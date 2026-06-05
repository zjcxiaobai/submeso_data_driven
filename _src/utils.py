from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open("r") as handle:
        config = yaml.safe_load(handle)
    return resolve_paths(config, path.resolve().parents[1])


def resolve_paths(config: dict, root: Path) -> dict:
    for section_name in ("train",):
        section = config.get(section_name, {})
        for key, value in list(section.items()):
            if key.endswith(("dir", "csv", "checkpoint")) and isinstance(value, str):
                section[key] = str((root / value).resolve())
    data = config.get("data", {})
    for key in ("x_path", "y_path", "stats_path"):
        if key in data and isinstance(data[key], str):
            data[key] = str(Path(data[key]))
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
