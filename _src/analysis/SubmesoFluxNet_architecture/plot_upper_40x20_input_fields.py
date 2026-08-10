#!/usr/bin/env python3
"""Plot each upper_40x20 SubmesoFluxNet input and target field at one time index."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FieldStyle:
    name: str
    label: str
    units: str
    cmap: str
    signed: bool


FIELD_STYLES = (
    FieldStyle("U", "U velocity", "m s$^{-1}$", "RdBu_r", True),
    FieldStyle("V", "V velocity", "m s$^{-1}$", "RdBu_r", True),
    FieldStyle("W", "W velocity", "m s$^{-1}$", "RdBu_r", True),
    FieldStyle("vorticity", "Relative vorticity", "s$^{-1}$", "RdBu_r", True),
    FieldStyle("strain", "Strain rate", "s$^{-1}$", "viridis", False),
    FieldStyle("T", "Potential temperature", "$^\\circ$C", "magma", False),
    FieldStyle("rho", "Density", "kg m$^{-3}$", "plasma", False),
    FieldStyle("kpphbl", "KPP boundary layer depth", "m", "YlGnBu", False),
)

TARGET_STYLE = FieldStyle(
    "submesoscale_heat_flux",
    "Submesoscale heat flux",
    "W m$^{-2}$",
    "RdBu_r",
    True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save one PNG per raw input variable from the upper_40x20 X.npy "
            "dataset. Use --time-index to plot a different snapshot."
        )
    )
    parser.add_argument(
        "--x-path",
        type=Path,
        default=PROJECT_DIR / "analysis/shuffle_upper40x20_mld_rho/data/X.npy",
        help="Path to raw X.npy. Default is the upper_40x20 rho/no-speed input array.",
    )
    parser.add_argument(
        "--y-path",
        type=Path,
        default=PROJECT_DIR / "analysis/shuffle_upper40x20_mld_rho/data/y.npy",
        help="Path to raw y.npy target array. Default is the upper_40x20 rho/no-speed submesoscale heat flux.",
    )
    parser.add_argument(
        "--time-index",
        "--t",
        type=int,
        default=0,
        help="Time/sample index to plot. Negative indices follow NumPy indexing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "upper_40x20_input_fields",
        help="Directory where per-variable PNG files are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    parser.add_argument(
        "--robust-percentile",
        type=float,
        default=100.0,
        help=(
            "Input-field color-limit percentile from 50 to 100. Default 100 uses "
            "the selected field min/max."
        ),
    )
    parser.add_argument(
        "--target-robust-percentile",
        type=float,
        default=98.0,
        help=(
            "Submesoscale heat-flux color-limit percentile from 50 to 100. "
            "Default 98 improves contrast by clipping rare extremes."
        ),
    )
    parser.add_argument(
        "--target-abs-limit",
        type=float,
        default=None,
        help=(
            "Optional symmetric absolute color limit for submesoscale heat flux. "
            "For example, --target-abs-limit 60 uses vmin=-60 and vmax=60."
        ),
    )
    parser.add_argument(
        "--boundary-linewidth",
        type=float,
        default=1.5,
        help="Line width for the black boundary around each field.",
    )
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def normalized_time_index(index: int, n_time: int) -> int:
    if not -n_time <= index < n_time:
        raise IndexError(f"--time-index {index} is outside valid range [{-n_time}, {n_time - 1}]")
    return index % n_time


def finite_limits(
    field: np.ndarray,
    style: FieldStyle,
    robust_percentile: float,
    abs_limit: float | None = None,
) -> tuple[float, float]:
    values = np.asarray(field[np.isfinite(field)], dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{style.name} contains no finite values")

    if not 50.0 <= robust_percentile <= 100.0:
        raise ValueError("color-limit percentile must be between 50 and 100")
    if abs_limit is not None:
        if abs_limit <= 0.0:
            raise ValueError("--target-abs-limit must be positive")
        return -float(abs_limit), float(abs_limit)

    if style.signed:
        high = float(np.nanpercentile(np.abs(values), robust_percentile))
        if high == 0.0:
            high = float(np.nanmax(np.abs(values)))
        if high == 0.0:
            high = 1.0
        return -high, high

    low_tail = (100.0 - robust_percentile) / 2.0
    high_tail = 100.0 - low_tail
    vmin = float(np.nanpercentile(values, low_tail))
    vmax = float(np.nanpercentile(values, high_tail))
    if vmin == vmax:
        pad = max(abs(vmin) * 0.05, 1.0)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def plot_field(
    field: np.ndarray,
    style: FieldStyle,
    *,
    time_index: int,
    output_dir: Path,
    dpi: int,
    robust_percentile: float,
    boundary_linewidth: float,
    abs_limit: float | None = None,
) -> Path:
    vmin, vmax = finite_limits(field, style, robust_percentile, abs_limit=abs_limit)
    height, width = field.shape
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    image = ax.imshow(
        field,
        origin="lower",
        cmap=style.cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    _ = image
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_axis_off()
    boundary = plt.Rectangle(
        (-0.5, -0.5),
        width,
        height,
        fill=False,
        edgecolor="black",
        linewidth=boundary_linewidth,
        joinstyle="miter",
        clip_on=False,
    )
    ax.add_patch(boundary)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"t{time_index:05d}_{safe_name(style.name)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    apply_style()

    x_path = args.x_path.expanduser().resolve()
    y_path = args.y_path.expanduser().resolve()
    if not x_path.exists():
        raise FileNotFoundError(f"Missing input array: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing target array: {y_path}")

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    if x.ndim != 4:
        raise ValueError(f"Expected X.npy with shape (time, channel, y, x), got {x.shape}")
    if x.shape[1] != len(FIELD_STYLES):
        raise ValueError(f"Expected {len(FIELD_STYLES)} channels, got {x.shape[1]} in {x_path}")
    if y.ndim != 4:
        raise ValueError(f"Expected y.npy with shape (time, channel, y, x), got {y.shape}")
    if y.shape[1] != 1:
        raise ValueError(f"Expected y.npy to have one target channel, got {y.shape[1]} in {y_path}")
    if y.shape[0] != x.shape[0] or y.shape[2:] != x.shape[2:]:
        raise ValueError(f"X/y shapes are incompatible: X={x.shape}, y={y.shape}")

    time_index = normalized_time_index(args.time_index, x.shape[0])
    frame = np.asarray(x[time_index], dtype=np.float32)
    target = np.asarray(y[time_index, 0], dtype=np.float32)
    print(f"Loaded {x_path}")
    print(f"Loaded {y_path}")
    print(f"X shape: {x.shape}; y shape: {y.shape}; plotting time index {time_index}")

    output_dir = args.output_dir.expanduser().resolve()
    saved_paths = []
    for channel_index, style in enumerate(FIELD_STYLES):
        path = plot_field(
            frame[channel_index],
            style,
            time_index=time_index,
            output_dir=output_dir,
            dpi=args.dpi,
            robust_percentile=args.robust_percentile,
            boundary_linewidth=args.boundary_linewidth,
        )
        saved_paths.append(path)
        print(f"Saved {path}")

    path = plot_field(
        target,
        TARGET_STYLE,
        time_index=time_index,
        output_dir=output_dir,
        dpi=args.dpi,
        robust_percentile=args.target_robust_percentile,
        boundary_linewidth=args.boundary_linewidth,
        abs_limit=args.target_abs_limit,
    )
    saved_paths.append(path)
    print(f"Saved {path}")

    print(f"Saved {len(saved_paths)} PNG files in {output_dir}")


if __name__ == "__main__":
    main()
