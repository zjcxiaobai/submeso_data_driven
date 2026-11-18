"""Spatiotemporal network used to predict submesoscale heat flux."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBottleneck(nn.Module):
    """Encode spatial fields and temporal evolution into latent feature maps."""

    def __init__(
        self,
        *,
        input_channels: int,
        latent_channels: int,
        spatial_channels: list[int],
        temporal_channels: int,
        temporal_layers: int,
        decoder_channels: list[int],
        dropout: float,
        temporal_reduce: str = "final",
    ) -> None:
        super().__init__()
        spatial_layers: list[nn.Module] = []
        channels = [input_channels] + spatial_channels
        for input_width, output_width in zip(channels[:-1], channels[1:]):
            spatial_layers.extend(
                [
                    nn.Conv2d(input_width, output_width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_width),
                    nn.SiLU(),
                    nn.Dropout2d(dropout),
                ]
            )
        self.spatial_encoder = nn.Sequential(*spatial_layers)

        temporal_layers_list: list[nn.Module] = []
        input_width = spatial_channels[-1]
        for _ in range(temporal_layers):
            temporal_layers_list.extend(
                [
                    nn.Conv3d(
                        input_width,
                        temporal_channels,
                        kernel_size=(3, 1, 1),
                        padding=(1, 0, 0),
                    ),
                    nn.BatchNorm3d(temporal_channels),
                    nn.SiLU(),
                    nn.Dropout3d(dropout),
                ]
            )
            input_width = temporal_channels
        self.temporal_encoder = nn.Sequential(*temporal_layers_list)
        self.to_latent = nn.Conv2d(temporal_channels, latent_channels, kernel_size=1)

        decoder_layers: list[nn.Module] = []
        channels = [latent_channels] + decoder_channels
        for input_width, output_width in zip(channels[:-1], channels[1:]):
            decoder_layers.extend(
                [
                    nn.Conv2d(input_width, output_width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_width),
                    nn.SiLU(),
                    nn.Dropout2d(dropout),
                ]
            )
        decoder_layers.append(nn.Conv2d(channels[-1], 1, kernel_size=1))
        self.decoder = nn.Sequential(*decoder_layers)

        if temporal_reduce not in {"final", "mean"}:
            raise ValueError("temporal_reduce must be 'final' or 'mean'")
        self.temporal_reduce = temporal_reduce

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent maps from input shaped ``(batch, time, channel, y, x)``."""
        batch, time, channels, height, width = x.shape
        spatial = self.spatial_encoder(x.reshape(batch * time, channels, height, width))
        _, features, encoded_height, encoded_width = spatial.shape
        spatial = spatial.reshape(
            batch,
            time,
            features,
            encoded_height,
            encoded_width,
        ).permute(0, 2, 1, 3, 4)
        temporal = self.temporal_encoder(spatial)
        if self.temporal_reduce == "final":
            summary = temporal[:, :, -1]
        else:
            summary = temporal.mean(dim=2)
        return self.to_latent(summary)

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Map latent feature maps to the predicted heat-flux field."""
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_features(self.encode_features(x))


def build_model(config: dict) -> TemporalBottleneck:
    """Construct the model described by a full experiment configuration."""
    model_cfg = config["model"]
    return TemporalBottleneck(
        input_channels=int(model_cfg["input_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        spatial_channels=[int(value) for value in model_cfg["spatial_channels"]],
        temporal_channels=int(model_cfg.get("temporal_channels", model_cfg["spatial_channels"][-1])),
        temporal_layers=int(model_cfg.get("temporal_layers", 2)),
        decoder_channels=[int(value) for value in model_cfg.get("decoder_channels", [32, 16])],
        dropout=float(model_cfg.get("dropout", 0.0)),
        temporal_reduce=str(model_cfg.get("temporal_reduce", "final")),
    )
