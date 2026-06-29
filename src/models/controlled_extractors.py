from __future__ import annotations

import torch
from torch import nn

from models.channel_metadata import axis_indices, modality_indices, sensor_indices
from models.common import TemporalEncoder1D, ensure_btc


def raw_summary_features(x: torch.Tensor) -> torch.Tensor:
    """Signal-derived summary statistics used by controlled residual/statistical models.

    Uses only IMU signal values: per-channel mean, std, min, and max. It never reads
    subject ids, labels, filenames, boundaries, original lengths, or metadata.
    """

    means = x.mean(dim=1)
    stds = x.std(dim=1, unbiased=False)
    mins = x.amin(dim=1)
    maxs = x.amax(dim=1)
    return torch.cat([means, stds, mins, maxs], dim=1)


class FlattenMLPExtractor(nn.Module):
    def __init__(self, input_length: int = 512, input_channels: int = 18, representation_dim: int = 64) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.representation_dim = representation_dim
        self.projection = nn.Sequential(
            nn.Linear(input_length * input_channels, representation_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        return self.projection(x.flatten(start_dim=1))


class StatsMLPExtractor(nn.Module):
    def __init__(self, input_length: int = 512, input_channels: int = 18, representation_dim: int = 64) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.representation_dim = representation_dim
        self.residual_projection = nn.Sequential(
            nn.Linear(input_channels * 4, representation_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        return self.residual_projection(raw_summary_features(x))


class AllChannelCNN1DExtractor(nn.Module):
    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        representation_dim: int = 64,
        small: bool = False,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.representation_dim = representation_dim
        if small:
            channels = (16, 32)
            kernels = (7, 5)
        else:
            channels = (32, 64)
            kernels = (9, 5)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=kernels[0], padding=kernels[0] // 2),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.Conv1d(channels[0], channels[1], kernel_size=kernels[1], padding=kernels[1] // 2),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Identity() if channels[1] == representation_dim else nn.Linear(channels[1], representation_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        features = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return self.projection(features)


class Shared1DExtractor(nn.Module):
    """Shared single-channel temporal encoder with optional identity and residual branches."""

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        representation_dim: int = 64,
        use_identity: bool = False,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        if input_channels != 18:
            raise ValueError("controlled shared extractors expect the approved 18-channel target layout")
        self.input_length = input_length
        self.input_channels = input_channels
        self.representation_dim = representation_dim
        self.use_identity = bool(use_identity)
        self.use_residual = bool(use_residual)
        self.shared_encoder = TemporalEncoder1D(in_channels=1, output_dim=representation_dim)
        self.channel_encoder_refs = [self.shared_encoder for _ in range(input_channels)]
        self.register_buffer("channel_ids", torch.arange(input_channels, dtype=torch.long), persistent=False)
        self.register_buffer("sensor_ids", torch.as_tensor(sensor_indices(), dtype=torch.long), persistent=False)
        self.register_buffer("modality_ids", torch.as_tensor(modality_indices(), dtype=torch.long), persistent=False)
        self.register_buffer("axis_ids", torch.as_tensor(axis_indices(), dtype=torch.long), persistent=False)
        if self.use_identity:
            self.identity_embeddings = nn.ModuleDict(
                {
                    "channel": nn.Embedding(input_channels, representation_dim),
                    "sensor": nn.Embedding(3, representation_dim),
                    "modality": nn.Embedding(2, representation_dim),
                    "axis": nn.Embedding(3, representation_dim),
                }
            )
            self.token_norm = nn.LayerNorm(representation_dim)
        else:
            self.identity_embeddings = nn.ModuleDict()
            self.token_norm = nn.Identity()
        if self.use_residual:
            self.residual_projection = nn.Sequential(
                nn.Linear(input_channels * 4, representation_dim),
                nn.ReLU(),
            )
            self.fusion_projection = nn.Sequential(
                nn.Linear(representation_dim * 2, representation_dim),
                nn.ReLU(),
            )
        else:
            self.residual_projection = None
            self.fusion_projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        tokens = self._encode_tokens(x)
        pooled = tokens.mean(dim=1)
        if not self.use_residual:
            return pooled
        residual = self.residual_projection(raw_summary_features(x))
        return self.fusion_projection(torch.cat([pooled, residual], dim=1))

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.shape
        encoded = self.shared_encoder(x.permute(0, 2, 1).reshape(batch_size * channels, 1, time_steps))
        tokens = encoded.reshape(batch_size, channels, encoded.shape[-1])
        if not self.use_identity:
            return tokens
        metadata = (
            self.identity_embeddings["channel"](self.channel_ids)
            + self.identity_embeddings["sensor"](self.sensor_ids)
            + self.identity_embeddings["modality"](self.modality_ids)
            + self.identity_embeddings["axis"](self.axis_ids)
        )
        return self.token_norm(tokens + metadata.unsqueeze(0))


class CNN2DExtractor(nn.Module):
    def __init__(self, input_length: int = 512, input_channels: int = 18, representation_dim: int = 64) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.representation_dim = representation_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(9, 3), padding=(4, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(nn.Linear(32, representation_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        features = self.encoder(x.unsqueeze(1)).flatten(start_dim=1)
        return self.projection(features)
