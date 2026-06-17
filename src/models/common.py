from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder1D(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(hidden_channels, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x).squeeze(-1)
        return self.projection(features)


def ensure_btc(x: torch.Tensor, input_length: int, input_channels: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected input shape (batch, time, channels), got {tuple(x.shape)}")
    if x.shape[1] != input_length or x.shape[2] != input_channels:
        raise ValueError(
            f"expected input shape (batch, {input_length}, {input_channels}), got {tuple(x.shape)}"
        )
    return x


def target_acc_gyro_indices(input_channels: int) -> tuple[list[int], list[int]]:
    if input_channels % 6 != 0:
        raise ValueError("target IMU channel layout expects each sensor to have 3 acc and 3 gyro channels")
    acc_indices: list[int] = []
    gyro_indices: list[int] = []
    for sensor_start in range(0, input_channels, 6):
        acc_indices.extend([sensor_start, sensor_start + 1, sensor_start + 2])
        gyro_indices.extend([sensor_start + 3, sensor_start + 4, sensor_start + 5])
    return acc_indices, gyro_indices
