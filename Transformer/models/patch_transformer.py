from __future__ import annotations

import math

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


class ConvRefineBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            HybridPad2d(1, 1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(16, channels), num_channels=channels),
            nn.SiLU(inplace=False),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            HybridPad2d(1, 1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(16, channels), num_channels=channels),
        )
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


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
        out = torch.cat(
            [
                torch.sin(xx[..., None] * omega),
                torch.cos(xx[..., None] * omega),
                torch.sin(yy[..., None] * omega),
                torch.cos(yy[..., None] * omega),
            ],
            dim=-1,
        )
        self.register_buffer("pos", out.reshape(1, height * width, embed_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos.to(dtype=x.dtype, device=x.device)


@register_model("PatchTransformerRegressor")
@register_model("patch_transformer")
class PatchTransformerRegressor(nn.Module):
    """Patch transformer with convolutional decoder for 56x56 map regression."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        image_size: int = 56,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.05,
        decoder_channels: int = 96,
        decoder_depth: int = 2,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.pos_encoding = SinCosPositionalEncoding2d(embed_dim, (self.grid_size, self.grid_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # PyTorch's TransformerEncoderLayer has a single dropout argument for attention output,
        # so this module keeps attention_dropout in the config for readability but follows torch defaults.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.decoder_in = nn.Conv2d(embed_dim, decoder_channels, kernel_size=1)
        self.refine = nn.Sequential(
            *[ConvRefineBlock(decoder_channels, dropout=attention_dropout) for _ in range(decoder_depth)]
        )
        self.decoder_out = nn.Sequential(
            HybridPad2d(1, 1),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=0),
            nn.SiLU(inplace=False),
            nn.Conv2d(decoder_channels, out_channels * patch_size * patch_size, kernel_size=1),
            nn.PixelShuffle(patch_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"PatchTransformerRegressor expects (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"PatchTransformerRegressor expects spatial size {(self.image_size, self.image_size)}, "
                f"got {tuple(x.shape[-2:])}"
            )
        tokens = self.patch_embed(x)
        batch_size, channels, height, width = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.pos_encoding(tokens)
        tokens = self.encoder(tokens)
        tokens = self.token_norm(tokens)
        tokens = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        y = self.decoder_in(tokens)
        y = self.refine(y)
        return self.decoder_out(y)
