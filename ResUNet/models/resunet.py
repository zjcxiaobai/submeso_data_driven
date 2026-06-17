from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


class HybridPad2d(nn.Module):
    """Periodic padding in x and reflect padding in y."""

    def __init__(self, pad_x: int, pad_y: int) -> None:
        super().__init__()
        self.pad_x = (pad_x, pad_x)
        self.pad_y = (pad_y, pad_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_x[0] > 0 or self.pad_x[1] > 0:
            x = F.pad(x, (self.pad_x[0], self.pad_x[1], 0, 0), mode="circular")
        if self.pad_y[0] > 0 or self.pad_y[1] > 0:
            x = F.pad(x, (0, 0, self.pad_y[0], self.pad_y[1]), mode="reflect")
        return x


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    if name in {"none", "linear"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def build_norm(kind: Literal["bn", "gn", "in", "none"], channels: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(channels, eps=1.0e-3, momentum=0.1, affine=True)
    if kind == "in":
        return nn.InstanceNorm2d(channels, eps=1.0e-5, affine=True, track_running_stats=False)
    if kind == "gn":
        groups = min(16, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels, eps=1.0e-5)
    raise ValueError(f"Unknown norm kind: {kind}")


class HybridConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        pad = (kernel_size // 2) * dilation
        self.pad = HybridPad2d(pad_x=pad, pad_y=pad)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: str = "silu",
        norm: Literal["bn", "gn", "in", "none"] = "gn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = norm == "none"
        self.conv1 = HybridConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.norm1 = build_norm(norm, out_channels)
        self.act1 = build_activation(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = HybridConv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
        )
        self.norm2 = build_norm(norm, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.out_act = build_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return self.out_act(y + residual)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels, **kwargs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpPixelShuffle(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__()
        scale = 2
        self.up = nn.Sequential(
            HybridPad2d(pad_x=1, pad_y=1),
            nn.Conv2d(in_channels, out_channels * scale * scale, kernel_size=3, padding=0),
            nn.PixelShuffle(scale),
        )
        self.block = ResidualBlock(out_channels + skip_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, x], dim=1))


class DilatedBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        activation: str,
        norm: Literal["bn", "gn", "in", "none"],
        dropout: float,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@register_model("ResUNet_3Layers")
@register_model("resunet_3layers")
class ResUNet_3Layers(nn.Module):
    """Residual three-level U-Net for 2D map regression."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        base_channels: int = 128,
        kernel_size: int = 3,
        activation: str = "silu",
        norm: Literal["bn", "gn", "in", "none"] = "gn",
        dropout: float = 0.0,
        bottleneck_dropout: float = 0.05,
        bottleneck_dilations: list[int] | tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        block_kwargs = dict(
            kernel_size=kernel_size,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.enc1 = ResidualBlock(in_channels, base_channels, **block_kwargs)
        self.down1 = Down(base_channels, base_channels * 2, **block_kwargs)
        self.down2 = Down(base_channels * 2, base_channels * 4, **block_kwargs)
        self.down3 = Down(base_channels * 4, base_channels * 8, **block_kwargs)
        self.bottleneck = DilatedBottleneck(
            base_channels * 8,
            kernel_size=kernel_size,
            activation=activation,
            norm=norm,
            dropout=bottleneck_dropout,
            dilations=tuple(int(dilation) for dilation in bottleneck_dilations),
        )
        self.up3 = UpPixelShuffle(base_channels * 8, base_channels * 4, base_channels * 4, **block_kwargs)
        self.up2 = UpPixelShuffle(base_channels * 4, base_channels * 2, base_channels * 2, **block_kwargs)
        self.up1 = UpPixelShuffle(base_channels * 2, base_channels, base_channels, **block_kwargs)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"ResUNet_3Layers expects (B,C,H,W), got shape {tuple(x.shape)}")

        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bottleneck(x4)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.outc(x)
