#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_utils import build_dataloaders  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device, load_config, seed_everything  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


DEFAULT_CONFIG = PROJECT_DIR / "configs/unet_encoder_convlstm_no_shuffle_upper_40x20.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis/explanability/decoder_channel_ablation"
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


class TensorStore:
    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def hook(self, _module, _inputs, output) -> None:
        if not torch.is_tensor(output):
            raise TypeError(f"Expected tensor output from decoder layer, got {type(output)!r}")
        self.tensor = output.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate channels in the last decoder feature layer before the final head. "
            "The script ranks channels by the RMS prediction change caused by zeroing each channel."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="Override device, for example cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for best-snapshot search.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--f0", type=float, default=-1.363e-4, help="Beta-plane f0 in s^-1.")
    parser.add_argument("--beta", type=float, default=1.313e-11, help="Beta-plane beta in s^-1 m^-1.")
    parser.add_argument(
        "--domain-height-m",
        type=float,
        default=1.0e6,
        help="Physical meridional height represented by the 20-pixel domain, used only for beta-plane Ro.",
    )
    parser.add_argument(
        "--beta-reference",
        choices=["center", "south", "north"],
        default="center",
        help="Reference y for f=f0+beta*(y-y_ref). center matches the FK-style yc-yc.mean convention.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Original dataset sample index to analyze. If omitted, choose the best-R2 test sample.",
    )
    parser.add_argument(
        "--test-position",
        type=int,
        default=None,
        help="Position inside the test split. Used only when --sample-index is omitted and --select is manual.",
    )
    parser.add_argument(
        "--select",
        choices=["best", "manual"],
        default="best",
        help="best chooses the test sample with highest snapshot R2; manual uses --sample-index or --test-position.",
    )
    return parser.parse_args()


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
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


def sample_r2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = prediction - target
    reduce_dims = tuple(range(1, target.ndim))
    sse = torch.sum(residual * residual, dim=reduce_dims)
    target_mean = torch.mean(target, dim=reduce_dims, keepdim=True)
    sst = torch.sum((target - target_mean) ** 2, dim=reduce_dims)
    r2 = torch.full_like(sse, float("nan"), dtype=torch.float64)
    valid = sst > 0.0
    r2[valid] = 1.0 - sse[valid].double() / sst[valid].double()
    return r2


def choose_best_sample(
    model: torch.nn.Module,
    test_loader,
    stats,
    config: dict,
    device: torch.device,
) -> tuple[int, int, float]:
    normalize_y = bool(config["data"].get("normalize_y", True))
    test_indices = np.asarray(test_loader.dataset.indices, dtype=np.int64)
    best_position = -1
    best_sample_index = -1
    best_r2 = -np.inf
    offset = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            batch_size = x.shape[0]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = denormalize_y(model(x), stats, normalize_y)
            target = denormalize_y(y, stats, normalize_y)
            r2 = sample_r2(pred, target).cpu().numpy()
            finite = np.isfinite(r2)
            if finite.any():
                local = int(np.nanargmax(r2))
                if float(r2[local]) > best_r2:
                    best_r2 = float(r2[local])
                    best_position = offset + local
                    best_sample_index = int(test_indices[best_position])
            offset += batch_size
            print(f"  searched {offset}/{len(test_indices)} test samples; current best R2={best_r2:.4f}", flush=True)

    if best_position < 0:
        raise RuntimeError("Could not find a finite-R2 test sample")
    return best_position, best_sample_index, best_r2


def select_test_sample(test_subset, sample_index: int | None, test_position: int | None) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    test_indices = np.asarray(test_subset.indices, dtype=np.int64)
    if sample_index is not None:
        matches = np.flatnonzero(test_indices == int(sample_index))
        if matches.size == 0:
            raise ValueError(f"sample_index={sample_index} is not in the test split")
        position = int(matches[0])
    else:
        position = 0 if test_position is None else int(test_position)
        if not 0 <= position < len(test_indices):
            raise IndexError(f"test_position={position} outside test split length {len(test_indices)}")

    x, y = test_subset[position]
    return position, int(test_indices[position]), x.unsqueeze(0), y.unsqueeze(0)


def capture_decoder_feature(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    store = TensorStore()
    handle = model.up1.register_forward_hook(store.hook)
    try:
        prediction = model(x)
    finally:
        handle.remove()
    if store.tensor is None:
        raise RuntimeError("Failed to capture model.up1 decoder feature map")
    return prediction, store.tensor


def rank_decoder_channels(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    baseline_prediction: torch.Tensor,
    output_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_channels = hidden.shape[1]
    scores = np.empty(n_channels, dtype=np.float64)
    deltas = []
    with torch.no_grad():
        for channel in range(n_channels):
            ablated_hidden = hidden.clone()
            ablated_hidden[:, channel] = 0.0
            ablated_prediction = model.outc(ablated_hidden)
            delta = (baseline_prediction - ablated_prediction) * output_scale
            scores[channel] = float(torch.sqrt(torch.mean(delta * delta)).cpu())
            deltas.append(delta.detach().cpu().numpy()[0, 0])
    return scores, np.asarray(deltas, dtype=np.float32)


def current_frame_fields(
    x: torch.Tensor,
    stats,
    config: dict,
    f0: float,
    beta: float,
    domain_height_m: float,
    beta_reference: str,
) -> dict[str, np.ndarray]:
    x_phys = denormalize_x(x, stats, bool(config["data"].get("normalize_x", True)))
    sequence_length = int(config["model"]["sequence_length"])
    frame_channels = int(config["model"]["frame_channels"])
    batch, _, height, width = x_phys.shape
    frames = x_phys.view(batch, sequence_length, frame_channels, height, width)
    current = frames[0, -1].detach().cpu().numpy()
    u = current[FEATURE_INDEX["U"]]
    v = current[FEATURE_INDEX["V"]]
    temperature = current[FEATURE_INDEX["T"]]
    w = current[FEATURE_INDEX["W"]]
    vorticity = current[FEATURE_INDEX["vorticity"]]

    ty, tx = np.gradient(temperature)
    uy, ux = np.gradient(u)
    vy, vx = np.gradient(v)
    grad_t = np.sqrt(tx * tx + ty * ty)
    t_prime = temperature - np.nanmean(temperature)

    numerator = tx * tx * ux + ty * ty * vy + tx * ty * (uy + vx)
    frontogenesis_ft = np.full_like(temperature, np.nan, dtype=np.float64)
    np.divide(-numerator, grad_t, out=frontogenesis_ft, where=grad_t > 1.0e-12)

    qx = -(ux * tx + vx * ty)
    qy = -(uy * tx + vy * ty)
    dqy_dy, _dqy_dx = np.gradient(qy)
    _dqx_dy, dqx_dx = np.gradient(qx)
    qconv = -(dqx_dx + dqy_dy)

    y = np.linspace(0.0, float(domain_height_m), temperature.shape[0], dtype=np.float64)
    if beta_reference == "south":
        y_ref = float(y.min())
    elif beta_reference == "north":
        y_ref = float(y.max())
    elif beta_reference == "center":
        y_ref = float(y.mean())
    else:
        raise ValueError(f"Unknown beta_reference: {beta_reference}")
    f = float(f0) + float(beta) * (y[:, None] - y_ref)
    ro = np.full_like(vorticity, np.nan, dtype=np.float64)
    np.divide(vorticity, f, out=ro, where=np.abs(f) > 1.0e-12)
    return {
        "U": u.astype(np.float32),
        "V": v.astype(np.float32),
        "W": w.astype(np.float32),
        "vorticity": vorticity.astype(np.float32),
        "strain": current[FEATURE_INDEX["strain"]].astype(np.float32),
        "T": temperature.astype(np.float32),
        "speed": current[FEATURE_INDEX["speed"]].astype(np.float32),
        "KPPhbl": current[FEATURE_INDEX["KPPhbl"]].astype(np.float32),
        "gradT": grad_t.astype(np.float32),
        "frontogenesis_FT": frontogenesis_ft.astype(np.float32),
        "Qconv": qconv.astype(np.float32),
        "Ro": ro.astype(np.float32),
        "W_times_Tprime": (w * t_prime).astype(np.float32),
    }


def write_rank_csv(output_dir: Path, scores: np.ndarray) -> None:
    order = np.argsort(scores)[::-1]
    with (output_dir / "decoder_channel_ablation_rank.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "channel", "rms_delta_prediction"])
        writer.writeheader()
        for rank, channel in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "channel": int(channel),
                    "rms_delta_prediction": float(scores[channel]),
                }
            )


def imshow_with_colorbar(fig, ax, image: np.ndarray, *, title: str, cmap: str, symmetric: bool = False) -> None:
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(image), 99.0))
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(image, 1.0))
        vmax = float(np.nanpercentile(image, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None
    mesh = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)


def plot_top_channels(
    output_dir: Path,
    *,
    top_channels: np.ndarray,
    scores: np.ndarray,
    hidden: np.ndarray,
    fields: dict[str, np.ndarray],
    truth: np.ndarray,
    deltas: np.ndarray,
    sample_index: int,
    baseline_r2: float,
) -> None:
    apply_plot_style()
    n_rows = len(top_channels)
    col_titles = ["Hidden feature", r"Input $|\nabla T|$", "Input W", r"$W T'$", "Truth heat flux", r"$\Delta$prediction"]
    fig, axes = plt.subplots(n_rows, 6, figsize=(9.2, 1.55 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = axes[None, :]

    for row, channel in enumerate(top_channels):
        images = [
            hidden[int(channel)],
            fields["gradT"],
            fields["W"],
            fields["W_times_Tprime"],
            truth,
            deltas[int(channel)],
        ]
        cmaps = ["viridis", "magma", "RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]
        symmetric = [False, False, True, True, True, True]
        for col, ax in enumerate(axes[row]):
            title = col_titles[col] if row == 0 else ""
            if col == 0:
                title = f"ch {int(channel)}\nscore={scores[int(channel)]:.3g}" if row == 0 else f"ch {int(channel)}\n{scores[int(channel)]:.3g}"
            imshow_with_colorbar(fig, ax, images[col], title=title, cmap=cmaps[col], symmetric=symmetric[col])

    fig.suptitle(
        f"Decoder channel ablation, sample {sample_index}, baseline R2={baseline_r2:.3f}",
        fontsize=9,
    )
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_dir / f"top{n_rows}_decoder_channel_ablation{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_all_decoder_channels(
    output_dir: Path,
    *,
    hidden: np.ndarray,
    scores: np.ndarray,
    sample_index: int,
    baseline_r2: float,
) -> None:
    apply_plot_style()
    n_channels = hidden.shape[0]
    n_cols = 16
    n_rows = int(np.ceil(n_channels / n_cols))
    order = np.argsort(scores)[::-1]
    rank_lookup = np.empty(n_channels, dtype=np.int64)
    rank_lookup[order] = np.arange(1, n_channels + 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.8, 0.85 * n_rows), constrained_layout=True)
    axes_flat = axes.ravel()
    for channel, ax in enumerate(axes_flat):
        if channel >= n_channels:
            ax.axis("off")
            continue
        image = hidden[channel]
        vmin = float(np.nanpercentile(image, 1.0))
        vmax = float(np.nanpercentile(image, 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
        ax.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"ch {channel}\nr{rank_lookup[channel]}", fontsize=4.8, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"All {n_channels} last-decoder feature channels, sample {sample_index}, baseline R2={baseline_r2:.3f}",
        fontsize=9,
    )
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(output_dir / f"all{n_channels}_decoder_feature_maps{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a_flat) & np.isfinite(b_flat)
    if valid.sum() < 3:
        return float("nan")
    a_flat = a_flat[valid] - np.nanmean(a_flat[valid])
    b_flat = b_flat[valid] - np.nanmean(b_flat[valid])
    denom = np.sqrt(np.nansum(a_flat * a_flat) * np.nansum(b_flat * b_flat))
    if denom <= 0.0:
        return float("nan")
    return float(np.nansum(a_flat * b_flat) / denom)


def weighted_decoder_feature_maps(hidden: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    weights = model.outc.weight.detach().cpu().numpy()[0, :, 0, 0].astype(np.float32)
    return hidden * weights[:, None, None]


def compute_spatial_correlations(weighted_hidden: np.ndarray, fields: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    input_names = [
        "U",
        "V",
        "W",
        "vorticity",
        "strain",
        "T",
        "speed",
        "KPPhbl",
        "gradT",
        "frontogenesis_FT",
        "Qconv",
        "Ro",
    ]
    correlations = np.full((len(input_names), weighted_hidden.shape[0]), np.nan, dtype=np.float64)
    for q_index, name in enumerate(input_names):
        q = fields[name]
        for channel in range(weighted_hidden.shape[0]):
            correlations[q_index, channel] = spatial_corr(weighted_hidden[channel], q)
    return input_names, correlations


def write_spatial_corr_csv(output_dir: Path, input_names: list[str], correlations: np.ndarray, scores: np.ndarray) -> None:
    with (output_dir / "decoder_channel_input_spatial_correlations.csv").open("w", newline="") as handle:
        fieldnames = ["input_variable", "channel", "ablation_rank", "ablation_score", "corr_weighted_feature_input"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        order = np.argsort(scores)[::-1]
        rank_lookup = np.empty(scores.shape[0], dtype=np.int64)
        rank_lookup[order] = np.arange(1, scores.shape[0] + 1)
        for q_index, name in enumerate(input_names):
            for channel in range(scores.shape[0]):
                writer.writerow(
                    {
                        "input_variable": name,
                        "channel": int(channel),
                        "ablation_rank": int(rank_lookup[channel]),
                        "ablation_score": float(scores[channel]),
                        "corr_weighted_feature_input": float(correlations[q_index, channel]),
                    }
                )


def plot_spatial_corr_for_inputs(
    output_dir: Path,
    *,
    input_names: list[str],
    correlations: np.ndarray,
    scores: np.ndarray,
    sample_index: int,
) -> None:
    apply_plot_style()
    order = np.argsort(scores)[::-1]
    ranks = np.arange(1, scores.shape[0] + 1)
    rank_lookup = np.empty(scores.shape[0], dtype=np.int64)
    rank_lookup[order] = ranks

    for q_index, name in enumerate(input_names):
        corr_ordered = correlations[q_index, order]
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [2.2, 1.0]}, constrained_layout=True)
        colors = np.where(corr_ordered >= 0.0, "#0F4D92", "#B64342")
        axes[0].bar(ranks, corr_ordered, color=colors, width=0.85)
        axes[0].axhline(0.0, color="#5F5F5F", linewidth=0.7)
        axes[0].set_xlabel("Decoder channel ablation rank")
        axes[0].set_ylabel(r"corr$(w_c F_c, q)$")
        axes[0].set_title(f"All 128 channels vs {name}", fontsize=8, pad=3)
        axes[0].set_ylim(-1.0, 1.0)
        axes[0].grid(axis="y", color="#D0D0D0", linewidth=0.35, alpha=0.65)

        heat = correlations[q_index].reshape(8, 16)
        mesh = axes[1].imshow(heat, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        axes[1].set_title("Channel layout", fontsize=8, pad=3)
        axes[1].set_xlabel("channel mod 16")
        axes[1].set_ylabel("channel // 16")
        fig.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.02)
        safe_name = name.replace("/", "_").replace(" ", "_")
        fig.suptitle(f"Spatial correlation with {name}, sample {sample_index}", fontsize=9)
        for suffix in [".png", ".pdf", ".svg"]:
            fig.savefig(output_dir / f"spatial_corr_weighted_decoder_channels_vs_{safe_name}{suffix}", dpi=600, bbox_inches="tight")
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
    model.eval()

    if args.select == "best" and args.sample_index is None and args.test_position is None:
        best_position, best_sample_index, best_r2 = choose_best_sample(model, test_loader, stats, config, device)
        print(f"Selected best test sample: sample_index={best_sample_index}, test_position={best_position}, R2={best_r2:.6f}", flush=True)
        position, sample_index, x_cpu, y_cpu = select_test_sample(test_loader.dataset, best_sample_index, None)
    else:
        position, sample_index, x_cpu, y_cpu = select_test_sample(test_loader.dataset, args.sample_index, args.test_position)
        best_r2 = float("nan")

    x = x_cpu.to(device)
    y = y_cpu.to(device)
    with torch.no_grad():
        baseline_prediction_norm, hidden = capture_decoder_feature(model, x)
        baseline_prediction = denormalize_y(baseline_prediction_norm, stats, bool(config["data"].get("normalize_y", True)))
        truth = denormalize_y(y, stats, bool(config["data"].get("normalize_y", True)))
        baseline_r2 = float(sample_r2(baseline_prediction, truth).cpu().numpy()[0])
    y_std = float(np.asarray(stats.y_std).reshape(-1)[0]) if bool(config["data"].get("normalize_y", True)) else 1.0
    scores, deltas = rank_decoder_channels(model, hidden, baseline_prediction_norm, output_scale=y_std)

    top_channels = np.argsort(scores)[::-1][: int(args.top_k)]
    hidden_np = hidden.detach().cpu().numpy()[0]
    fields = current_frame_fields(
        x,
        stats,
        config,
        f0=float(args.f0),
        beta=float(args.beta),
        domain_height_m=float(args.domain_height_m),
        beta_reference=str(args.beta_reference),
    )
    truth_np = truth.detach().cpu().numpy()[0, 0]
    prediction_np = baseline_prediction.detach().cpu().numpy()[0, 0]
    weighted_hidden_np = weighted_decoder_feature_maps(hidden_np, model)
    input_names, spatial_correlations = compute_spatial_correlations(weighted_hidden_np, fields)

    write_rank_csv(output_dir, scores)
    write_spatial_corr_csv(output_dir, input_names, spatial_correlations, scores)
    np.savez_compressed(
        output_dir / "decoder_channel_ablation_best_snapshot.npz",
        sample_index=np.asarray(sample_index, dtype=np.int64),
        test_position=np.asarray(position, dtype=np.int64),
        baseline_r2=np.asarray(baseline_r2, dtype=np.float64),
        selected_search_r2=np.asarray(best_r2, dtype=np.float64),
        scores=scores.astype(np.float64),
        top_channels=top_channels.astype(np.int64),
        hidden=hidden_np.astype(np.float32),
        weighted_hidden=weighted_hidden_np.astype(np.float32),
        input_names=np.asarray(input_names),
        spatial_correlations=spatial_correlations.astype(np.float64),
        prediction=prediction_np.astype(np.float32),
        truth=truth_np.astype(np.float32),
        delta_prediction=deltas.astype(np.float32),
        U=fields["U"].astype(np.float32),
        V=fields["V"].astype(np.float32),
        W=fields["W"].astype(np.float32),
        vorticity=fields["vorticity"].astype(np.float32),
        strain=fields["strain"].astype(np.float32),
        T=fields["T"].astype(np.float32),
        speed=fields["speed"].astype(np.float32),
        KPPhbl=fields["KPPhbl"].astype(np.float32),
        gradT=fields["gradT"].astype(np.float32),
        frontogenesis_FT=fields["frontogenesis_FT"].astype(np.float32),
        Qconv=fields["Qconv"].astype(np.float32),
        Ro=fields["Ro"].astype(np.float32),
        W_times_Tprime=fields["W_times_Tprime"].astype(np.float32),
        checkpoint=np.asarray(str(checkpoint_path)),
        config=np.asarray(str(config_path)),
    )
    plot_top_channels(
        output_dir,
        top_channels=top_channels,
        scores=scores,
        hidden=hidden_np,
        fields=fields,
        truth=truth_np,
        deltas=deltas,
        sample_index=sample_index,
        baseline_r2=baseline_r2,
    )
    plot_all_decoder_channels(
        output_dir,
        hidden=hidden_np,
        scores=scores,
        sample_index=sample_index,
        baseline_r2=baseline_r2,
    )
    plot_spatial_corr_for_inputs(
        output_dir,
        input_names=input_names,
        correlations=spatial_correlations,
        scores=scores,
        sample_index=sample_index,
    )
    metadata = {
        "sample_index": int(sample_index),
        "test_position": int(position),
        "baseline_r2": baseline_r2,
        "selected_search_r2": best_r2,
        "top_channels": [int(v) for v in top_channels],
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "ranking_metric": "RMS of physical-unit prediction change after setting one up1 decoder channel to zero",
        "spatial_correlation": "corr(w_c * F_c(y,x), q(y,x)) for final-head channel weight w_c and current-frame input q",
        "spatial_correlation_inputs": input_names,
        "f0": float(args.f0),
        "beta": float(args.beta),
        "domain_height_m": float(args.domain_height_m),
        "beta_reference": str(args.beta_reference),
        "derived_field_notes": {
            "frontogenesis_FT": "-1/|grad_h T| * (Tx^2 ux + Ty^2 vy + Tx Ty (uy + vx)); finite differences use input-grid index spacing",
            "Qconv": "-(dQx/dx + dQy/dy), Qx=-(ux Tx + vx Ty), Qy=-(uy Tx + vy Ty); finite differences use input-grid index spacing",
            "Ro": "vorticity / (f0 + beta * (y - y_ref)); y is reconstructed from domain_height_m over the 20-pixel crop",
        },
    }
    (output_dir / "decoder_channel_ablation_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Top {len(top_channels)} channels: {', '.join(str(int(v)) for v in top_channels)}", flush=True)
    print(f"Saved decoder channel ablation outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
