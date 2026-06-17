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
        x = F.pad(x, (self.pad_x[0], self.pad_x[1], 0, 0), mode="circular")
        return F.pad(x, (0, 0, self.pad_y[0], self.pad_y[1]), mode="reflect")


def build_activation(name: str, leakyrelu_slope: float = 0.2) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU(inplace=False)
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name in {"leaky_relu", "leakyrelu"}:
        return nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=False)
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
        return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1.0e-5)
    raise ValueError(f"Unknown norm kind: {kind}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        activation: str = "silu",
        norm: Literal["bn", "gn", "in", "none"] = "bn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        pad = kernel_size // 2
        self.net = nn.Sequential(
            HybridPad2d(pad_x=pad, pad_y=pad),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=0,
                bias=(norm == "none"),
            ),
            build_norm(norm, out_channels),
            build_activation(activation),
            nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        activation: str = "silu",
        norm: Literal["bn", "gn", "in", "none"] = "bn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
                dropout=dropout,
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                activation=activation,
                norm=norm,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, **kwargs))

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
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


@register_model("UNet_3Layers")
@register_model("unet_3layers")
class UNet_3Layers(nn.Module):
    """Three-level U-Net with hybrid padding and pixel-shuffle decoder."""

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int = 1,
        base_channels: int | None = None,
        kernel_size: int = 3,
        activation: str = "silu",
        norm: Literal["bn", "gn", "in", "none"] | None = None,
        dropout: float = 0.0,
        input_channels: int | None = None,
        head_channel: int | None = None,
        base: int | None = None,
        batch_norm: Literal["bn", "gn", "in", "none"] | None = None,
        dropout_2D: float | None = None,
    ) -> None:
        super().__init__()
        in_channels = input_channels if input_channels is not None else in_channels
        out_channels = head_channel if head_channel is not None else out_channels
        base_channels = base if base is not None else base_channels
        norm = batch_norm if batch_norm is not None else norm
        dropout = dropout_2D if dropout_2D is not None else dropout

        if in_channels is None:
            in_channels = 7
        if base_channels is None:
            base_channels = 32
        if norm is None:
            norm = "bn"

        block_kwargs = dict(
            kernel_size=kernel_size,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )
        self.enc1 = DoubleConv(in_channels, base_channels, **block_kwargs)
        self.down1 = Down(base_channels, base_channels * 2, **block_kwargs)
        self.down2 = Down(base_channels * 2, base_channels * 4, **block_kwargs)
        self.down3 = Down(base_channels * 4, base_channels * 8, **block_kwargs)

        self.up3 = UpPixelShuffle(base_channels * 8, base_channels * 4, base_channels * 4, **block_kwargs)
        self.up2 = UpPixelShuffle(base_channels * 4, base_channels * 2, base_channels * 2, **block_kwargs)
        self.up1 = UpPixelShuffle(base_channels * 2, base_channels, base_channels, **block_kwargs)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"UNet_3Layers expects (B,C,H,W), got shape {tuple(x.shape)}")

        x1 = self.enc1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.outc(x)



@register_model("TemporalUNet_3Layers")
@register_model("temporal_unet_3layers")
class TemporalUNet_3Layers(UNet_3Layers):
    """UNet_3Layers alias for channel-stacked temporal inputs such as t-1,t,t+1."""

    def __init__(self, *args, in_channels: int | None = None, input_channels: int | None = None, **kwargs) -> None:
        channels = input_channels if input_channels is not None else in_channels
        if channels is None:
            channels = 24
        super().__init__(*args, in_channels=channels, **kwargs)
