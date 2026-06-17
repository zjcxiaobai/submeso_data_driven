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
        stride: int = 1,
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
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        activation: str = "silu",
        norm: Literal["bn", "gn", "in", "none"] = "gn",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        bias = norm == "none"
        self.conv1 = HybridConv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.norm1 = build_norm(norm, out_channels)
        self.act1 = build_activation(activation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = HybridConv2d(out_channels, out_channels, kernel_size=kernel_size, bias=bias)
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


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        norm = kwargs.get("norm", "gn")
        activation = kwargs.get("activation", "silu")
        self.down = nn.Sequential(
            HybridConv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=(norm == "none")),
            build_norm(norm, out_channels),
            build_activation(activation),
        )
        self.block = ResidualConvBlock(out_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            HybridPad2d(1, 1),
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=0),
            nn.PixelShuffle(2),
        )
        self.block = ResidualConvBlock(out_channels + skip_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([skip, x], dim=1))


class SinCosPositionalEncoding2d(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, int]) -> None:
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for 2D sin/cos encoding")
        height, width = grid_size
        y = torch.linspace(0.0, 1.0, steps=height)
        x = torch.linspace(0.0, 1.0, steps=width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        omega = torch.arange(embed_dim // 4, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(1, embed_dim // 4 - 1)))
        pos = torch.cat(
            [
                torch.sin(xx[..., None] * omega),
                torch.cos(xx[..., None] * omega),
                torch.sin(yy[..., None] * omega),
                torch.cos(yy[..., None] * omega),
            ],
            dim=-1,
        )
        self.register_buffer("pos", pos.reshape(1, height * width, embed_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos.to(dtype=x.dtype, device=x.device)


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        grid_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pos = SinCosPositionalEncoding2d(channels, (grid_size, grid_size))
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.pos(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)


@register_model("ConvTransformerUNet")
@register_model("conv_transformer_unet")
class ConvTransformerUNet(nn.Module):
    """Convolutional U-Net with transformer attention at the bottleneck."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        image_size: int = 56,
        base_channels: int = 64,
        bottleneck_channels: int = 192,
        transformer_depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        kernel_size: int = 3,
        norm: Literal["bn", "gn", "in", "none"] = "gn",
        activation: str = "silu",
        conv_dropout: float = 0.05,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")
        if bottleneck_channels % num_heads != 0:
            raise ValueError("bottleneck_channels must be divisible by num_heads")
        if bottleneck_channels % 4 != 0:
            raise ValueError("bottleneck_channels must be divisible by 4")

        block_kwargs = dict(
            kernel_size=kernel_size,
            activation=activation,
            norm=norm,
            dropout=conv_dropout,
        )
        self.stem = ResidualConvBlock(in_channels, base_channels, **block_kwargs)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2, **block_kwargs)
        self.down2 = DownsampleBlock(base_channels * 2, bottleneck_channels, **block_kwargs)
        self.transformer = TransformerBottleneck(
            bottleneck_channels,
            grid_size=image_size // 4,
            depth=transformer_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=transformer_dropout,
        )
        self.post_transformer = ResidualConvBlock(bottleneck_channels, bottleneck_channels, **block_kwargs)
        self.up1 = UpBlock(bottleneck_channels, base_channels * 2, base_channels * 2, **block_kwargs)
        self.up2 = UpBlock(base_channels * 2, base_channels, base_channels, **block_kwargs)
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"ConvTransformerUNet expects (B,C,H,W), got {tuple(x.shape)}")
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.transformer(x3)
        x3 = self.post_transformer(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.head(x)
