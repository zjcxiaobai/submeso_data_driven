"""Loss function used to train SubmesoFluxNet."""

import torch.nn as nn


def build_loss(name: str) -> nn.Module:
    if name != "huber":
        raise ValueError(f"Unknown loss: {name}")
    return nn.HuberLoss()
