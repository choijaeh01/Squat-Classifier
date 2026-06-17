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


class ChannelAttentionPooling(nn.Module):
    """Small attention scorer over channel embeddings, not over flattened channel features."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(embeddings), dim=1)
        return (embeddings * weights).sum(dim=1)


class ChannelSharedMeanPoolV2(nn.Module):
    """Shares one temporal encoder across all 18 channels, then mean-pools channel embeddings.

    Weight sharing happens at ``self.shared_encoder``: every channel is reshaped into a
    one-channel sequence and passed through this exact same module object.
    """

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        num_classes: int = 5,
        embedding_dim: int = 32,
        head_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.shared_encoder = TemporalEncoder1D(in_channels=1, output_dim=embedding_dim)
        self.channel_encoder_refs = [self.shared_encoder for _ in range(input_channels)]
        self.aggregation = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self._encode_channels(x)
        pooled = embeddings.mean(dim=1)
        return self.classifier(pooled)

    def _encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        batch_size, time_steps, channels = x.shape
        encoded = self.shared_encoder(x.permute(0, 2, 1).reshape(batch_size * channels, 1, time_steps))
        return encoded.reshape(batch_size, channels, encoded.shape[-1])


class ChannelSharedAttentionPoolV2(ChannelSharedMeanPoolV2):
    """Shares one temporal encoder across channels and pools embeddings with a small scorer."""

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        num_classes: int = 5,
        embedding_dim: int = 32,
        attention_hidden_dim: int = 8,
        head_hidden_dim: int = 16,
    ) -> None:
        super().__init__(
            input_length=input_length,
            input_channels=input_channels,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            head_hidden_dim=head_hidden_dim,
        )
        self.aggregation = ChannelAttentionPooling(embedding_dim=embedding_dim, hidden_dim=attention_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self._encode_channels(x)
        pooled = self.aggregation(embeddings)
        return self.classifier(pooled)


class ModalitySharedMeanPoolV2(nn.Module):
    """Uses one acc encoder and one gyro encoder, pooling channels within each modality.

    Weight sharing happens separately in ``self.acc_encoder`` and ``self.gyro_encoder``.
    The two encoders are intentionally different objects, but every accelerometer channel
    reuses the acc object and every gyroscope channel reuses the gyro object.
    """

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        num_classes: int = 5,
        embedding_dim: int = 32,
        head_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.acc_indices, self.gyro_indices = target_acc_gyro_indices(input_channels)
        self.acc_encoder = TemporalEncoder1D(in_channels=1, output_dim=embedding_dim)
        self.gyro_encoder = TemporalEncoder1D(in_channels=1, output_dim=embedding_dim)
        self.acc_encoder_refs = [self.acc_encoder for _ in self.acc_indices]
        self.gyro_encoder_refs = [self.gyro_encoder for _ in self.gyro_indices]
        self.aggregation = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        acc_pooled = self._encode_and_mean_pool(x, self.acc_indices, self.acc_encoder)
        gyro_pooled = self._encode_and_mean_pool(x, self.gyro_indices, self.gyro_encoder)
        return self.classifier(torch.cat([acc_pooled, gyro_pooled], dim=1))

    def _encode_and_mean_pool(
        self,
        x: torch.Tensor,
        indices: list[int],
        encoder: TemporalEncoder1D,
    ) -> torch.Tensor:
        subset = x[:, :, indices]
        batch_size, time_steps, channels = subset.shape
        encoded = encoder(subset.permute(0, 2, 1).reshape(batch_size * channels, 1, time_steps))
        embeddings = encoded.reshape(batch_size, channels, encoded.shape[-1])
        return embeddings.mean(dim=1)
