from __future__ import annotations

import torch
from torch import nn

from models.common import ensure_btc


class ClassicalFeatureBaseline(nn.Module):
    """Differentiable placeholder for a future classical feature baseline."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        feature_dim = input_channels * 4
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        means = x.mean(dim=1)
        stds = x.std(dim=1, unbiased=False)
        mins = x.amin(dim=1)
        maxs = x.amax(dim=1)
        features = torch.cat([means, stds, mins, maxs], dim=1)
        return self.classifier(features)


class AllChannelConv1D(nn.Module):
    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        features = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(features)


class AllChannelConv1DSmall(nn.Module):
    """Smaller all-channel Conv1D baseline for capacity-matched v2 comparisons."""

    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 24, kernel_size=5, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(24, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        features = self.encoder(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(features)


class CNN2DBaseline(nn.Module):
    def __init__(self, input_length: int = 512, input_channels: int = 18, num_classes: int = 5) -> None:
        super().__init__()
        self.input_length = input_length
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(9, 3), padding=(4, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btc(x, self.input_length, self.input_channels)
        features = self.encoder(x.unsqueeze(1)).flatten(start_dim=1)
        return self.classifier(features)
