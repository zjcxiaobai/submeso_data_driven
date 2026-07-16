#!/usr/bin/env python3
"""Create four 20x20 SubmesoFluxNet datasets from the same crop used for animations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Subdomain:
    name: str
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int


SUBDOMAINS = [
    Subdomain("lower_left", 10, 30, 10, 30),
    Subdomain("lower_right", 10, 30, 30, 50),
    Subdomain("upper_left", 30, 50, 10, 30),
    Subdomain("upper_right", 30, 50, 30, 50),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice X.npy and y.npy into the four 20x20 domains used by "
            "animate_heat_flux_subdomains.py."
        )
    )
    parser.add_argument("--x-path", type=Path, default=PROJECT_DIR / "data/X.npy")
    parser.add_argument("--y-path", type=Path, default=PROJECT_DIR / "data/y.npy")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data/train_on_small_domain")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_inputs(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError(f"Expected 4-D arrays shaped (time, channel, y, x), got X={x.shape}, y={y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y sample count differs: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[-2:] != (56, 56) or y.shape[-2:] != (56, 56):
        raise ValueError(f"Expected 56x56 spatial shape, got X={x.shape[-2:]}, y={y.shape[-2:]}")


def save_subdomain(
    x: np.ndarray,
    y: np.ndarray,
    subdomain: Subdomain,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    domain_dir = output_dir / subdomain.name
    x_output = domain_dir / "X.npy"
    y_output = domain_dir / "y.npy"
    metadata_output = domain_dir / "metadata.json"
    if not overwrite and (x_output.exists() or y_output.exists()):
        raise FileExistsError(f"{domain_dir} already contains X.npy or y.npy; pass --overwrite to replace.")

    domain_dir.mkdir(parents=True, exist_ok=True)
    y_slice = slice(subdomain.y_start, subdomain.y_stop)
    x_slice = slice(subdomain.x_start, subdomain.x_stop)
    x_small = np.asarray(x[:, :, y_slice, x_slice], dtype=np.float32)
    y_small = np.asarray(y[:, :, y_slice, x_slice], dtype=np.float32)

    if x_small.shape[-2:] != (20, 20) or y_small.shape[-2:] != (20, 20):
        raise ValueError(f"{subdomain.name} should be 20x20, got X={x_small.shape}, y={y_small.shape}")

    np.save(x_output, x_small)
    np.save(y_output, y_small)

    metadata = {
        "name": subdomain.name,
        "source_crop": "same as animations: full-domain y=10:50, x=10:50 split into four 20x20 domains",
        "original_y_start": subdomain.y_start,
        "original_y_stop_exclusive": subdomain.y_stop,
        "original_x_start": subdomain.x_start,
        "original_x_stop_exclusive": subdomain.x_stop,
        "shape_X": list(x_small.shape),
        "shape_y": list(y_small.shape),
        "dtype_X": str(x_small.dtype),
        "dtype_y": str(y_small.dtype),
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
    for subdomain in SUBDOMAINS:
        print(
            f"Writing {subdomain.name}: "
            f"y={subdomain.y_start}:{subdomain.y_stop}, x={subdomain.x_start}:{subdomain.x_stop}",
            flush=True,
        )
        metadata.append(save_subdomain(x, y, subdomain, output_dir, overwrite=args.overwrite))

    summary = {
        "source_X": str(x_path),
        "source_y": str(y_path),
        "domains": [asdict(domain) for domain in SUBDOMAINS],
        "outputs": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Done. Wrote small-domain datasets under {output_dir}")


if __name__ == "__main__":
    main()
