from __future__ import annotations

import torch
from torch import nn

from models.common import TemporalEncoder1D, ensure_btc, target_acc_gyro_indices


class ChannelShared1DEncoder(nn.Module):
    """Applies the same temporal encoder object to every input channel."""

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        num_classes: int = 5,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.shared_encoder = TemporalEncoder1D(in_channels=1, output_dim=embedding_dim)
        self.channel_encoder_refs = [self.shared_encoder for _ in range(input_channels)]
        self.classifier = nn.Sequential(
            nn.Linear(input_channels * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        batch_size, time_steps, channels = x.shape
        encoded = self.shared_encoder(x.permute(0, 2, 1).reshape(batch_size * channels, 1, time_steps))
        features = encoded.reshape(batch_size, channels * encoded.shape[-1])
        return self.classifier(features)


class ModalitySharedAccGyroEncoder(nn.Module):
    """Uses one shared encoder for all accelerometer channels and another for all gyroscope channels."""

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        num_classes: int = 5,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.acc_indices, self.gyro_indices = target_acc_gyro_indices(input_channels)
        self.acc_encoder = TemporalEncoder1D(in_channels=1, output_dim=embedding_dim)
        self.gyro_encoder = TemporalEncoder1D(in_channels=1, output_dim=embedding_dim)
        self.acc_encoder_refs = [self.acc_encoder for _ in self.acc_indices]
        self.gyro_encoder_refs = [self.gyro_encoder for _ in self.gyro_indices]
        self.classifier = nn.Sequential(
            nn.Linear(input_channels * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        acc_features = self._encode_indices(x, self.acc_indices, self.acc_encoder)
        gyro_features = self._encode_indices(x, self.gyro_indices, self.gyro_encoder)
        features = torch.cat([acc_features, gyro_features], dim=1)
        return self.classifier(features)

    def _encode_indices(self, x: torch.Tensor, indices: list[int], encoder: TemporalEncoder1D) -> torch.Tensor:
        subset = x[:, :, indices]
        batch_size, time_steps, channels = subset.shape
        encoded = encoder(subset.permute(0, 2, 1).reshape(batch_size * channels, 1, time_steps))
        return encoded.reshape(batch_size, channels * encoded.shape[-1])
