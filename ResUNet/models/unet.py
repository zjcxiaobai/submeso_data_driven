from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            build_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            build_activation(activation),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        bilinear: bool = True,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_channels = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_channels = in_channels // 2
        self.conv = DoubleConv(
            up_channels + skip_channels,
            out_channels,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


@register_model("unet")
class UNet(nn.Module):
    """Small configurable U-Net for channel-first 2D regression."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 48,
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("UNet depth must be at least 2")

        channels = [base_channels * (2**level) for level in range(depth)]
        self.inc = DoubleConv(in_channels, channels[0], dropout=dropout, activation=activation)
        self.downs = nn.ModuleList(
            Down(channels[level - 1], channels[level], dropout=dropout, activation=activation)
            for level in range(1, depth)
        )
        self.ups = nn.ModuleList(
            Up(
                channels[level],
                channels[level - 1],
                channels[level - 1],
                bilinear=bilinear,
                dropout=dropout,
                activation=activation,
            )
            for level in range(depth - 1, 0, -1)
        )
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        x = skips[0]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.outc(x)
