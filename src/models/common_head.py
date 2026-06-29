from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CommonHeadConfig:
    representation_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1
    activation: str = "relu"
    num_classes: int = 5


class CommonMLPClassifierHead(nn.Module):
    """Common classifier head shared by all controlled neural model architectures.

    The architecture is identical across controlled models. Weights are trained
    separately per model/run, but the layer sequence and parameter count remain fixed.
    """

    def __init__(self, config: CommonHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or CommonHeadConfig()
        if self.config.activation != "relu":
            raise ValueError("controlled common head currently supports activation='relu' only")
        self.net = nn.Sequential(
            nn.Linear(self.config.representation_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(self.config.dropout)),
            nn.Linear(self.config.hidden_dim, self.config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.config.representation_dim:
            raise ValueError(f"expected representation shape (batch, {self.config.representation_dim}), got {tuple(x.shape)}")
        return self.net(x)

    def architecture_signature(self) -> tuple[str, str, str, str]:
        return (
            f"linear:{self.config.representation_dim}->{self.config.hidden_dim}",
            self.config.activation,
            f"dropout:{self.config.dropout}",
            f"linear:{self.config.hidden_dim}->{self.config.num_classes}",
        )
