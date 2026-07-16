#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils import build_dataloaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DEFAULT_CONFIG = PROJECT_DIR / "configs/unet_encoder_convlstm_no_shuffle_upper_40x20.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis/explanability/interaction_response_encoder_convlstm"
FEATURE_INDEX = {
    "U": 0,
    "V": 1,
    "W": 2,
    "vorticity": 3,
    "strain": 4,
    "T": 5,
    "speed": 6,
    "KPPhbl": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 2D interaction-response diagrams for the encoder-ConvLSTM model. "
            "Each panel bins current-frame input variables and colors the bins by mean predicted heat flux."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Override device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional number of test samples to evaluate first. Omit to use all test samples.",
    )
    parser.add_argument(
        "--quantile-range",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        help="Percentile range used for bin edges, reducing outlier domination.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Hide 2D bins with fewer than this many pixels.",
    )
    return parser.parse_args()


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )


def resolve_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


def denormalize_x(values: torch.Tensor, stats, normalize_x: bool) -> torch.Tensor:
    if not normalize_x:
        return values
    x_mean = torch.as_tensor(stats.x_mean, dtype=values.dtype, device=values.device)
    x_std = torch.as_tensor(stats.x_std, dtype=values.dtype, device=values.device)
    return values * x_std + x_mean


def denormalize_y(values: torch.Tensor, stats, normalize_y: bool) -> torch.Tensor:
    if not normalize_y:
        return values
    y_mean = torch.as_tensor(stats.y_mean, dtype=values.dtype, device=values.device)
    y_std = torch.as_tensor(stats.y_std, dtype=values.dtype, device=values.device)
    return values * y_std + y_mean


def current_frame(x_physical: torch.Tensor, sequence_length: int, frame_channels: int) -> torch.Tensor:
    batch, _, height, width = x_physical.shape
    frames = x_physical.view(batch, sequence_length, frame_channels, height, width)
    return frames[:, -1]


def temperature_gradient(temperature: np.ndarray) -> np.ndarray:
    dtdy, dtdx = np.gradient(temperature, axis=(-2, -1))
    return np.sqrt(dtdx * dtdx + dtdy * dtdy)


def collect_prediction_and_inputs(
    model: torch.nn.Module,
    test_loader,
    stats,
    config: dict,
    device: torch.device,
    sample_limit: int | None,
) -> dict[str, np.ndarray]:
    sequence_length = int(config["model"]["sequence_length"])
    frame_channels = int(config["model"]["frame_channels"])
    normalize_x = bool(config["data"].get("normalize_x", True))
    normalize_y = bool(config["data"].get("normalize_y", True))

    chunks: dict[str, list[np.ndarray]] = {
        "prediction": [],
        "T": [],
        "W": [],
        "W_prime": [],
        "strain": [],
        "vorticity": [],
        "KPPhbl": [],
        "gradT": [],
    }
    seen = 0
    model.eval()
    with torch.no_grad():
        for x, _y in test_loader:
            if sample_limit is not None and seen >= sample_limit:
                break
            if sample_limit is not None and seen + x.shape[0] > sample_limit:
                x = x[: sample_limit - seen]

            x = x.to(device, non_blocking=True)
            pred = denormalize_y(model(x), stats, normalize_y)[:, 0]
            x_phys = current_frame(denormalize_x(x, stats, normalize_x), sequence_length, frame_channels)

            t = x_phys[:, FEATURE_INDEX["T"]].detach().cpu().numpy()
            chunks["prediction"].append(pred.detach().cpu().numpy())
            chunks["T"].append(t)
            w = x_phys[:, FEATURE_INDEX["W"]].detach().cpu().numpy()
            chunks["W"].append(w)
            chunks["W_prime"].append(w - w.mean(axis=(-2, -1), keepdims=True))
            chunks["strain"].append(x_phys[:, FEATURE_INDEX["strain"]].detach().cpu().numpy())
            chunks["vorticity"].append(x_phys[:, FEATURE_INDEX["vorticity"]].detach().cpu().numpy())
            chunks["KPPhbl"].append(x_phys[:, FEATURE_INDEX["KPPhbl"]].detach().cpu().numpy())
            chunks["gradT"].append(temperature_gradient(t))

            seen += x.shape[0]
            print(f"  evaluated {seen} test samples", flush=True)

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def binned_response(
    x: np.ndarray,
    y: np.ndarray,
    response: np.ndarray,
    *,
    bins: int,
    quantile_range: tuple[float, float],
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_flat = x.ravel()
    y_flat = y.ravel()
    r_flat = response.ravel()
    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(r_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    r_flat = r_flat[valid]

    q_low, q_high = quantile_range
    x_low, x_high = np.nanpercentile(x_flat, [q_low, q_high])
    y_low, y_high = np.nanpercentile(y_flat, [q_low, q_high])
    x_edges = np.linspace(x_low, x_high, bins + 1)
    y_edges = np.linspace(y_low, y_high, bins + 1)

    in_range = (x_flat >= x_low) & (x_flat <= x_high) & (y_flat >= y_low) & (y_flat <= y_high)
    x_flat = x_flat[in_range]
    y_flat = y_flat[in_range]
    r_flat = r_flat[in_range]

    weighted_sum, _, _ = np.histogram2d(y_flat, x_flat, bins=(y_edges, x_edges), weights=r_flat)
    count, _, _ = np.histogram2d(y_flat, x_flat, bins=(y_edges, x_edges))
    mean_response = np.full_like(weighted_sum, np.nan, dtype=np.float64)
    np.divide(weighted_sum, count, out=mean_response, where=count > 0)
    mean_response[count < min_count] = np.nan
    return x_edges, y_edges, mean_response, count


def save_panel_data(
    output_dir: Path,
    panel_name: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    mean_response: np.ndarray,
    count: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / f"{panel_name}_binned_response.npz",
        x_edges=x_edges.astype(np.float64),
        y_edges=y_edges.astype(np.float64),
        mean_prediction=mean_response.astype(np.float64),
        count=count.astype(np.int64),
    )

    with (output_dir / f"{panel_name}_binned_response.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x_bin", "y_bin", "x_center", "y_center", "mean_prediction", "count"])
        writer.writeheader()
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        for y_index, y_center in enumerate(y_centers):
            for x_index, x_center in enumerate(x_centers):
                writer.writerow(
                    {
                        "x_bin": x_index,
                        "y_bin": y_index,
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "mean_prediction": float(mean_response[y_index, x_index])
                        if np.isfinite(mean_response[y_index, x_index])
                        else "",
                        "count": int(count[y_index, x_index]),
                    }
                )


def binned_response_1d(
    x: np.ndarray,
    response: np.ndarray,
    *,
    bins: int,
    quantile_range: tuple[float, float],
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_flat = x.ravel()
    r_flat = response.ravel()
    valid = np.isfinite(x_flat) & np.isfinite(r_flat)
    x_flat = x_flat[valid]
    r_flat = r_flat[valid]

    q_low, q_high = quantile_range
    x_low, x_high = np.nanpercentile(x_flat, [q_low, q_high])
    x_edges = np.linspace(x_low, x_high, bins + 1)
    in_range = (x_flat >= x_low) & (x_flat <= x_high)
    x_flat = x_flat[in_range]
    r_flat = r_flat[in_range]

    weighted_sum, _ = np.histogram(x_flat, bins=x_edges, weights=r_flat)
    count, _ = np.histogram(x_flat, bins=x_edges)
    mean_response = np.full(weighted_sum.shape, np.nan, dtype=np.float64)
    np.divide(weighted_sum, count, out=mean_response, where=count > 0)
    mean_response[count < min_count] = np.nan
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    return x_edges, x_centers, mean_response, count


def save_1d_data(
    output_dir: Path,
    panel_name: str,
    x_edges: np.ndarray,
    x_centers: np.ndarray,
    mean_response: np.ndarray,
    count: np.ndarray,
) -> None:
    np.savez_compressed(
        output_dir / f"{panel_name}_1d_response.npz",
        x_edges=x_edges.astype(np.float64),
        x_centers=x_centers.astype(np.float64),
        mean_prediction=mean_response.astype(np.float64),
        count=count.astype(np.int64),
    )

    with (output_dir / f"{panel_name}_1d_response.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x_bin", "x_center", "mean_prediction", "count"])
        writer.writeheader()
        for x_index, x_center in enumerate(x_centers):
            writer.writerow(
                {
                    "x_bin": x_index,
                    "x_center": float(x_center),
                    "mean_prediction": float(mean_response[x_index]) if np.isfinite(mean_response[x_index]) else "",
                    "count": int(count[x_index]),
                }
            )


def plot_interactions(
    fields: dict[str, np.ndarray],
    output_dir: Path,
    *,
    bins: int,
    quantile_range: tuple[float, float],
    min_count: int,
) -> None:
    apply_plot_style()
    panels = [
        ("T_vs_Wprime", "T", "W_prime", "Temperature", r"$W'$"),
        ("gradT_vs_Wprime", "gradT", "W_prime", r"$|\nabla T|$", r"$W'$"),
        ("gradT_vs_KPPhbl", "gradT", "KPPhbl", r"$|\nabla T|$", "KPP boundary layer depth"),
        ("T_vs_strain", "T", "strain", "Temperature", "Strain"),
        ("T_vs_vorticity", "T", "vorticity", "Temperature", "Vorticity"),
    ]
    response = fields["prediction"]
    response_scale = np.nanpercentile(np.abs(response), 98.0)
    vmin, vmax = -float(response_scale), float(response_scale)

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.7), constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, (panel_name, x_key, y_key, x_label, y_label) in zip(axes_flat, panels):
        x_edges, y_edges, mean_response, count = binned_response(
            fields[x_key],
            fields[y_key],
            response,
            bins=bins,
            quantile_range=quantile_range,
            min_count=min_count,
        )
        save_panel_data(output_dir, panel_name, x_edges, y_edges, mean_response, count)
        mesh = ax.pcolormesh(x_edges, y_edges, mean_response, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(panel_name.replace("_", " "), fontsize=8, pad=3)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02, label="Mean predicted heat flux")

    axes_flat[-1].axis("off")
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_dir / f"interaction_response_2d{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_product_and_single_variable_responses(
    fields: dict[str, np.ndarray],
    output_dir: Path,
    *,
    bins: int,
    quantile_range: tuple[float, float],
    min_count: int,
) -> None:
    apply_plot_style()
    response = fields["prediction"]
    curves = [
        ("output_vs_gradT_times_Wprime", fields["gradT"] * fields["W_prime"], r"$|\nabla T| \times W'$"),
        ("output_vs_gradT_times_KPPhbl", fields["gradT"] * fields["KPPhbl"], r"$|\nabla T| \times h_{KPP}$"),
        ("output_vs_Wprime", fields["W_prime"], r"$W'$"),
        ("output_vs_gradT", fields["gradT"], r"$|\nabla T|$"),
        ("output_vs_KPPhbl", fields["KPPhbl"], "KPP boundary layer depth"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.5), constrained_layout=True)
    axes_flat = axes.ravel()
    scatter = None
    for ax, (panel_name, x_values, x_label) in zip(axes_flat, curves):
        x_edges, x_centers, mean_response, count = binned_response_1d(
            x_values,
            response,
            bins=bins,
            quantile_range=quantile_range,
            min_count=min_count,
        )
        save_1d_data(output_dir, panel_name, x_edges, x_centers, mean_response, count)
        valid = np.isfinite(mean_response)
        ax.plot(x_centers[valid], mean_response[valid], color="#0F4D92", linewidth=1.2)
        scatter = ax.scatter(
            x_centers[valid],
            mean_response[valid],
            c=np.log10(np.maximum(count[valid], 1)),
            cmap="magma",
            s=12,
            zorder=3,
        )
        ax.axhline(0.0, color="#9A9A9A", linewidth=0.7, zorder=0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean predicted heat flux")
        ax.set_title(panel_name.replace("_", " "), fontsize=8, pad=3)
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.35, alpha=0.65)

    axes_flat[-1].axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes_flat[-1], fraction=0.7, label=r"$\log_{10}$ bin count")
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_dir / f"interaction_response_1d_products_and_singles{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = resolve_path(args.config, PROJECT_DIR)
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(args.device or config.get("device", "auto"))

    config["data"]["batch_size"] = int(args.batch_size)
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False

    _, _, test_loader, stats = build_dataloaders(config, PROJECT_DIR, int(config.get("seed", 42)))
    model = build_model(config["model"]).to(device)
    checkpoint_path = resolve_path(args.checkpoint or Path(config["train"]["best_checkpoint"]), PROJECT_DIR)
    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    fields = collect_prediction_and_inputs(model, test_loader, stats, config, device, args.sample_limit)
    np.savez_compressed(output_dir / "interaction_response_fields.npz", **{k: v.astype(np.float32) for k, v in fields.items()})
    plot_interactions(
        fields,
        output_dir,
        bins=int(args.bins),
        quantile_range=(float(args.quantile_range[0]), float(args.quantile_range[1])),
        min_count=int(args.min_count),
    )
    plot_product_and_single_variable_responses(
        fields,
        output_dir,
        bins=int(args.bins),
        quantile_range=(float(args.quantile_range[0]), float(args.quantile_range[1])),
        min_count=int(args.min_count),
    )
    print(f"Saved interaction-response plots to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
