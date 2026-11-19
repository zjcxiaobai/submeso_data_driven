"""Spatiotemporal network used to predict submesoscale heat flux."""

import torch
import torch.nn as nn


def _conv2d_layers(channels: list[int], dropout: float) -> list[nn.Module]:
    """Create repeated Conv2D-BatchNorm-SiLU-Dropout layers."""
    layers = []
    for input_channels, output_channels in zip(channels[:-1], channels[1:]):
        layers.extend(
            [
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.SiLU(),
                nn.Dropout2d(dropout),
            ]
        )
    return layers


def _conv3d_layers(
    input_channels: int,
    output_channels: int,
    number_of_layers: int,
    dropout: float,
) -> list[nn.Module]:
    """Create temporal Conv3D-BatchNorm-SiLU-Dropout layers."""
    layers = []
    for _ in range(number_of_layers):
        layers.extend(
            [
                nn.Conv3d(
                    input_channels,
                    output_channels,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                ),
                nn.BatchNorm3d(output_channels),
                nn.SiLU(),
                nn.Dropout3d(dropout),
            ]
        )
        input_channels = output_channels
    return layers


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
    ) -> None:
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            *_conv2d_layers([input_channels] + spatial_channels, dropout)
        )
        self.temporal_encoder = nn.Sequential(
            *_conv3d_layers(
                spatial_channels[-1],
                temporal_channels,
                temporal_layers,
                dropout,
            )
        )
        self.to_latent = nn.Conv2d(temporal_channels, latent_channels, kernel_size=1)

        decoder_widths = [latent_channels] + decoder_channels
        self.decoder = nn.Sequential(
            *_conv2d_layers(decoder_widths, dropout),
            nn.Conv2d(decoder_widths[-1], 1, kernel_size=1),
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent maps from input shaped ``(batch, time, channel, y, x)``."""
        batch, time, channels, height, width = x.shape
        spatial = self.spatial_encoder(x.reshape(batch * time, channels, height, width))
        spatial = spatial.reshape(batch, time, -1, height, width)
        spatial = spatial.permute(0, 2, 1, 3, 4)
        temporal = self.temporal_encoder(spatial)
        return self.to_latent(temporal[:, :, -1])

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        """Map latent feature maps to the predicted heat-flux field."""
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_features(self.encode_features(x))


def build_model(config: dict) -> TemporalBottleneck:
    """Build SubmesoFluxNet from the model configuration."""
    model_cfg = config["model"]
    return TemporalBottleneck(
        input_channels=int(model_cfg["input_channels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        spatial_channels=[int(value) for value in model_cfg["spatial_channels"]],
        temporal_channels=int(model_cfg["temporal_channels"]),
        temporal_layers=int(model_cfg["temporal_layers"]),
        decoder_channels=[int(value) for value in model_cfg["decoder_channels"]],
        dropout=float(model_cfg["dropout"]),
    )
