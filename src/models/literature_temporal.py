from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.common import ensure_btc


class CNNLSTMLiteratureV1(nn.Module):
    """Clean-room CNN-LSTM family baseline common in IMU activity literature."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = _SmallConvFrontEnd(input_channels=input_channels, hidden_channels=32)
        self.recurrent = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(_btc_to_bct(x, self.input_length, self.input_channels)).transpose(1, 2)
        _, (hidden, _) = self.recurrent(features)
        return self.classifier(hidden[-1])


class CNNGRULiteratureV1(nn.Module):
    """Clean-room CNN-GRU baseline for recurrent-unit comparison."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = _SmallConvFrontEnd(input_channels=input_channels, hidden_channels=32)
        self.recurrent = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(_btc_to_bct(x, self.input_length, self.input_channels)).transpose(1, 2)
        _, hidden = self.recurrent(features)
        return self.classifier(hidden[-1])


class ResCNNBiGRUAttentionLiteV1(nn.Module):
    """Lite clean-room Residual CNN-BiGRU-Attention reference baseline.

    This intentionally excludes focal loss, balanced sampling, mixup, Time-CutMix,
    per-window z-score, and any old repository code. It is a protocol-matched
    temporal reference baseline, not a reproduction of prior thesis results.
    """

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            ResidualConvBlock(input_channels, 32, kernel_size=7),
            ResidualConvBlock(32, 32, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
        )
        self.recurrent = nn.GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
        self.aggregation = TemporalAttention(input_dim=64, hidden_dim=24)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(_btc_to_bct(x, self.input_length, self.input_channels)).transpose(1, 2)
        recurrent, _ = self.recurrent(features)
        pooled = self.aggregation(recurrent)
        return self.classifier(pooled)


class TCNLiteratureV1(nn.Module):
    """Temporal Convolutional Network baseline with dilated residual blocks."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=1),
            nn.ReLU(),
            TCNResidualBlock(32, dilation=1),
            TCNResidualBlock(32, dilation=2),
            TCNResidualBlock(32, dilation=4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(_btc_to_bct(x, self.input_length, self.input_channels)).squeeze(-1)
        return self.classifier(features)


class LSTMOnlyLiteratureV1(nn.Module):
    """Small LSTM-only baseline without convolutional feature extraction."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.LSTM(input_size=input_channels, hidden_size=32, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        _, (hidden, _) = self.encoder(x)
        return self.classifier(hidden[-1])


class GRUOnlyLiteratureV1(nn.Module):
    """Small GRU-only baseline without convolutional feature extraction."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.GRU(input_size=input_channels, hidden_size=32, batch_first=True)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        _, hidden = self.encoder(x)
        return self.classifier(hidden[-1])


class TransformerEncoderLiteV1(nn.Module):
    """Small Transformer encoder baseline for temporal context modeling."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        model_dim = 48
        self.encoder = nn.ModuleDict(
            {
                "projection": nn.Linear(input_channels, model_dim),
                "position": SinusoidalPositionEncoding(input_length=input_length, model_dim=model_dim),
                "transformer": nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=model_dim,
                        nhead=4,
                        dim_feedforward=96,
                        dropout=0.0,
                        batch_first=True,
                        activation="gelu",
                    ),
                    num_layers=2,
                ),
            }
        )
        self.classifier = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        tokens = self.encoder["projection"](x)
        tokens = self.encoder["position"](tokens)
        encoded = self.encoder["transformer"](tokens)
        return self.classifier(encoded.mean(dim=1))


class LeeStyleCNNLSTM2DV1(nn.Module):
    """Lee-style adapted 2D CNN-LSTM baseline for the current 18-channel IMU input.

    This is a clean-room adaptation, not an exact reproduction. The model keeps
    the dataset unchanged, deterministically downsamples the 512-step input to
    40 steps inside the network, applies 3x3 Conv2D blocks over the time-channel
    matrix, then summarizes the channel axis before a one-layer LSTM.
    """

    def __init__(
        self,
        input_length: int = 512,
        input_channels: int = 18,
        num_classes: int = 5,
        downsampled_length: int = 40,
        lstm_hidden_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.downsampled_length = downsampled_length
        self.encoder = nn.Sequential(
            _Conv2DBlock(1, 16, dropout=dropout),
            _Conv2DBlock(16, 24, dropout=dropout),
            _Conv2DBlock(24, 32, dropout=dropout),
        )
        self.channel_pool = nn.AdaptiveAvgPool2d((downsampled_length, 4))
        self.recurrent = nn.LSTM(input_size=32 * 4, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def downsample_to_40(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), self.downsampled_length)
        return pooled.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x40 = self.downsample_to_40(x)
        image = x40.unsqueeze(1)
        encoded = self.encoder(image)
        pooled = self.channel_pool(encoded)
        sequence = pooled.permute(0, 2, 1, 3).flatten(start_dim=2)
        _, (hidden, _) = self.recurrent(sequence)
        return self.classifier(hidden[-1])


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + x)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(sequence), dim=1)
        return (sequence * weights).sum(dim=1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_length: int, model_dim: int) -> None:
        super().__init__()
        position = torch.arange(input_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim))
        encoding = torch.zeros(input_length, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1], :]


class _SmallConvFrontEnd(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(24, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _btc_to_bct(x: torch.Tensor, input_length: int, input_channels: int) -> torch.Tensor:
    return ensure_btc(x, input_length, input_channels).transpose(1, 2)


class _Conv2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
