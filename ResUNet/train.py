#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from data_utils import build_dataloaders
from engine import fit
from models import build_model
from utils import get_device, load_config, project_dir, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable PyTorch regression model.")
    parser.add_argument("--config", type=Path, default=project_dir() / "configs/unet_regression.yaml")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__last__",
        default=None,
        help="Resume from a checkpoint path, or from train.last_checkpoint when no path is given.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training expects CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed()
    root = project_dir()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)) + rank)
    device = torch.device(f"cuda:{local_rank}") if distributed else get_device(config.get("device", "auto"))

    train_loader, val_loader, _, _ = build_dataloaders(
        config,
        root,
        int(config.get("seed", 42)),
        distributed=distributed,
    )
    model = build_model(config["model"]).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        print(model, flush=True)
        print(f"Device: {device}; distributed={distributed}; world_size={world_size}", flush=True)

    resume_checkpoint = None
    if args.resume is not None:
        if args.resume == "__last__":
            resume_checkpoint = Path(config["train"]["last_checkpoint"])
        else:
            resume_checkpoint = Path(args.resume).expanduser()
            if not resume_checkpoint.is_absolute():
                resume_checkpoint = root / resume_checkpoint

    try:
        fit(model, train_loader, val_loader, config, device, resume_checkpoint=resume_checkpoint)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
