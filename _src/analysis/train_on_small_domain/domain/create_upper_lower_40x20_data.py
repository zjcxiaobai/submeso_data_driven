#!/usr/bin/env python3
"""Create upper and lower 40x20 datasets from the same 56x56 crop."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RectDomain:
    name: str
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int


DOMAINS = [
    RectDomain("upper_40x20", 30, 50, 10, 50),
    RectDomain("lower_40x20", 10, 30, 10, 50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice X.npy/y.npy into upper and lower 40x20 domains."
    )
    parser.add_argument("--x-path", type=Path, default=PROJECT_DIR / "data/X.npy")
    parser.add_argument("--y-path", type=Path, default=PROJECT_DIR / "data/y.npy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "data/train_on_small_domain",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected 4-D arrays shaped (time, channel, y, x), got X={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y sample count differs: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[-2:] != (56, 56) or y.shape[-2:] != (56, 56):
        raise ValueError(f"Expected 56x56 spatial shape, got X={x.shape[-2:]}, y={y.shape[-2:]}")


def save_domain(
    x: np.ndarray,
    y: np.ndarray,
    domain: RectDomain,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    domain_dir = output_dir / domain.name
    x_output = domain_dir / "X.npy"
    y_output = domain_dir / "y.npy"
    metadata_output = domain_dir / "metadata.json"
    if not overwrite and (x_output.exists() or y_output.exists()):
        raise FileExistsError(f"{domain_dir} already contains X.npy or y.npy; pass --overwrite to replace.")

    domain_dir.mkdir(parents=True, exist_ok=True)
    y_slice = slice(domain.y_start, domain.y_stop)
    x_slice = slice(domain.x_start, domain.x_stop)
    x_rect = np.asarray(x[:, :, y_slice, x_slice], dtype=np.float32)
    y_rect = np.asarray(y[:, :, y_slice, x_slice], dtype=np.float32)

    if x_rect.shape[-2:] != (20, 40) or y_rect.shape[-2:] != (20, 40):
        raise ValueError(f"{domain.name} should be y,x = 20,40; got X={x_rect.shape}, y={y_rect.shape}")

    np.save(x_output, x_rect)
    np.save(y_output, y_rect)

    metadata = {
        "name": domain.name,
        "source_crop": "same interior crop as four-domain split: full-domain y=10:50, x=10:50",
        "description": "40x20 means x width 40 and y height 20; NumPy shape is (..., 20, 40).",
        "original_y_start": domain.y_start,
        "original_y_stop_exclusive": domain.y_stop,
        "original_x_start": domain.x_start,
        "original_x_stop_exclusive": domain.x_stop,
        "shape_X": list(x_rect.shape),
        "shape_y": list(y_rect.shape),
        "dtype_X": str(x_rect.dtype),
        "dtype_y": str(y_rect.dtype),
    }
    with metadata_output.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return metadata


def main() -> None:
    args = parse_args()
    x_path = args.x_path.expanduser().resolve()
    y_path = args.y_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    validate_inputs(x, y)

    metadata = []
    for domain in DOMAINS:
        print(
            f"Writing {domain.name}: "
            f"y={domain.y_start}:{domain.y_stop}, x={domain.x_start}:{domain.x_stop}",
            flush=True,
        )
        metadata.append(save_domain(x, y, domain, output_dir, overwrite=args.overwrite))

    summary = {
        "source_X": str(x_path),
        "source_y": str(y_path),
        "domains": [asdict(domain) for domain in DOMAINS],
        "outputs": metadata,
    }
    with (output_dir / "upper_lower_40x20_metadata.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Done. Wrote upper/lower 40x20 datasets under {output_dir}")


if __name__ == "__main__":
    main()
