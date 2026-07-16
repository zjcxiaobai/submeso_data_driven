#!/usr/bin/env python3
"""Animate four 20x20 heat-flux subdomains from the 56x56 y label array."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm


PROJECT_DIR = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Subdomain:
    name: str
    y_slice: slice
    x_slice: slice
    original_y: tuple[int, int]
    original_x: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the heat-flux label y.npy, crop the 56x56 domain to 40x40, "
            "split it into four 20x20 domains, and write one animation per domain."
        )
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=PROJECT_DIR / "data/y.npy",
        help="Path to y.npy shaped (time, channel, y, x) or (time, y, x).",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "animations")
    parser.add_argument("--start", type=int, default=0, help="First time/sample index to consider.")
    parser.add_argument("--time-stride", type=int, default=2, help="Interval along the time axis.")
    parser.add_argument("--max-frames", type=int, default=100, help="Number of frames to animate.")
    parser.add_argument("--channel", type=int, default=0, help="Label channel to animate when y is 4-D.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Symmetric color limit percentile computed from the selected cropped frames.",
    )
    return parser.parse_args()


def load_heat_flux(path: Path, channel: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing y data: {path}")

    y = np.load(path, mmap_mode="r")
    if y.ndim == 4:
        if channel < 0 or channel >= y.shape[1]:
            raise IndexError(f"channel must be in 0..{y.shape[1] - 1}, got {channel}")
        heat_flux = y[:, channel]
    elif y.ndim == 3:
        heat_flux = y
    else:
        raise ValueError(f"Expected y shape (time, channel, y, x) or (time, y, x), got {y.shape}")

    if heat_flux.shape[-2:] != (56, 56):
        raise ValueError(f"Expected spatial shape 56x56, got {heat_flux.shape[-2:]}")
    return heat_flux


def selected_time_indices(n_time: int, start: int, stride: int, max_frames: int) -> np.ndarray:
    if start < 0 or start >= n_time:
        raise ValueError(f"start must be in 0..{n_time - 1}, got {start}")
    if stride <= 0:
        raise ValueError(f"time-stride must be positive, got {stride}")
    if max_frames <= 0:
        raise ValueError(f"max-frames must be positive, got {max_frames}")

    stop = min(n_time, start + stride * max_frames)
    indices = np.arange(start, stop, stride, dtype=np.int64)
    return indices[:max_frames]


def crop_and_split(selected: np.ndarray) -> tuple[np.ndarray, list[Subdomain]]:
    cropped = selected[:, 10:50, 10:50]
    if cropped.shape[-2:] != (40, 40):
        raise ValueError(f"Internal crop should be 40x40, got {cropped.shape[-2:]}")

    subdomains = [
        Subdomain("lower_left", slice(0, 20), slice(0, 20), (10, 29), (10, 29)),
        Subdomain("lower_right", slice(0, 20), slice(20, 40), (10, 29), (30, 49)),
        Subdomain("upper_left", slice(20, 40), slice(0, 20), (30, 49), (10, 29)),
        Subdomain("upper_right", slice(20, 40), slice(20, 40), (30, 49), (30, 49)),
    ]
    return cropped, subdomains


def symmetric_norm(values: np.ndarray, percentile: float) -> TwoSlopeNorm:
    if not 50.0 < percentile <= 100.0:
        raise ValueError("--percentile must be in (50, 100]")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        limit = 1.0
    else:
        limit = float(np.nanpercentile(np.abs(finite), percentile))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def animate_subdomain(
    values: np.ndarray,
    time_indices: np.ndarray,
    subdomain: Subdomain,
    norm: TwoSlopeNorm,
    output: Path,
    fps: int,
) -> None:
    field = values[:, subdomain.y_slice, subdomain.x_slice]
    if field.shape[-2:] != (20, 20):
        raise ValueError(f"{subdomain.name} should be 20x20, got {field.shape[-2:]}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(3.2, 3.0), constrained_layout=True)
    image = ax.imshow(
        field[0],
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    colorbar.set_label("Heat flux")
    title = ax.set_title("")
    ax.set_xlabel("Local x index")
    ax.set_ylabel("Local y index")
    ax.set_xticks([0, 9, 19])
    ax.set_yticks([0, 9, 19])

    def update(frame: int):
        image.set_data(field[frame])
        title.set_text(
            f"{subdomain.name.replace('_', ' ')} | t={int(time_indices[frame])}\n"
            f"original y {subdomain.original_y[0]}-{subdomain.original_y[1]}, "
            f"x {subdomain.original_x[0]}-{subdomain.original_x[1]}"
        )
        return image, title

    animation = FuncAnimation(fig, update, frames=field.shape[0], blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    heat_flux = load_heat_flux(args.y_path.expanduser().resolve(), args.channel)
    time_indices = selected_time_indices(
        heat_flux.shape[0],
        start=args.start,
        stride=args.time_stride,
        max_frames=args.max_frames,
    )
    selected = np.asarray(heat_flux[time_indices], dtype=np.float64)
    cropped, subdomains = crop_and_split(selected)
    norm = symmetric_norm(cropped, args.percentile)

    output_dir = args.output_dir.expanduser().resolve()
    for subdomain in subdomains:
        output = output_dir / f"heat_flux_{subdomain.name}_{len(time_indices)}frames_stride{args.time_stride}.gif"
        print(f"Writing {output}", flush=True)
        animate_subdomain(cropped, time_indices, subdomain, norm, output, args.fps)

    print(
        "Done: used "
        f"{len(time_indices)} frames from {int(time_indices[0])} to {int(time_indices[-1])} "
        f"with stride {args.time_stride}."
    )


if __name__ == "__main__":
    main()
