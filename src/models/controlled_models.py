from __future__ import annotations

import torch
from torch import nn

from models.common_head import CommonHeadConfig, CommonMLPClassifierHead
from models.controlled_extractors import (
    AllChannelCNN1DExtractor,
    CNN2DExtractor,
    FlattenMLPExtractor,
    Shared1DExtractor,
    StatsMLPExtractor,
)


CONTROLLED_NEURAL_MODEL_NAMES = [
    "controlled_flatten_mlp",
    "controlled_stats_mlp",
    "controlled_all_channel_1d_cnn",
    "controlled_all_channel_1d_cnn_small",
    "controlled_shared_1d",
    "controlled_shared_1d_identity",
    "controlled_shared_1d_residual",
    "controlled_shared_1d_residual_identity",
    "controlled_2d_cnn",
]


class ControlledNeuralModel(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        *,
        representation_dim: int = 64,
        num_classes: int = 5,
        head_hidden_dim: int = 64,
        head_dropout: float = 0.1,
        model_name: str = "controlled_neural_model",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.extractor = extractor
        self.representation_dim = int(representation_dim)
        self.classifier = CommonMLPClassifierHead(
            CommonHeadConfig(
                representation_dim=self.representation_dim,
                hidden_dim=int(head_hidden_dim),
                dropout=float(head_dropout),
                activation="relu",
                num_classes=int(num_classes),
            )
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


def controlled_flatten_mlp(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        FlattenMLPExtractor(input_length=input_length, input_channels=input_channels, representation_dim=representation_dim),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_flatten_mlp",
    )


def controlled_stats_mlp(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        StatsMLPExtractor(input_length=input_length, input_channels=input_channels, representation_dim=representation_dim),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_stats_mlp",
    )


def controlled_all_channel_1d_cnn(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        AllChannelCNN1DExtractor(input_length=input_length, input_channels=input_channels, representation_dim=representation_dim),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_all_channel_1d_cnn",
    )


def controlled_all_channel_1d_cnn_small(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        AllChannelCNN1DExtractor(
            input_length=input_length,
            input_channels=input_channels,
            representation_dim=representation_dim,
            small=True,
        ),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_all_channel_1d_cnn_small",
    )


def controlled_shared_1d(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        Shared1DExtractor(input_length=input_length, input_channels=input_channels, representation_dim=representation_dim),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_shared_1d",
    )


def controlled_shared_1d_identity(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        Shared1DExtractor(
            input_length=input_length,
            input_channels=input_channels,
            representation_dim=representation_dim,
            use_identity=True,
        ),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_shared_1d_identity",
    )


def controlled_shared_1d_residual(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        Shared1DExtractor(
            input_length=input_length,
            input_channels=input_channels,
            representation_dim=representation_dim,
            use_residual=True,
        ),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_shared_1d_residual",
    )


def controlled_shared_1d_residual_identity(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        Shared1DExtractor(
            input_length=input_length,
            input_channels=input_channels,
            representation_dim=representation_dim,
            use_identity=True,
            use_residual=True,
        ),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_shared_1d_residual_identity",
    )


def controlled_2d_cnn(
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
    representation_dim: int = 64,
) -> ControlledNeuralModel:
    return ControlledNeuralModel(
        CNN2DExtractor(input_length=input_length, input_channels=input_channels, representation_dim=representation_dim),
        representation_dim=representation_dim,
        num_classes=num_classes,
        model_name="controlled_2d_cnn",
    )


CONTROLLED_MODEL_FACTORIES = {
    "controlled_flatten_mlp": controlled_flatten_mlp,
    "controlled_stats_mlp": controlled_stats_mlp,
    "controlled_all_channel_1d_cnn": controlled_all_channel_1d_cnn,
    "controlled_all_channel_1d_cnn_small": controlled_all_channel_1d_cnn_small,
    "controlled_shared_1d": controlled_shared_1d,
    "controlled_shared_1d_identity": controlled_shared_1d_identity,
    "controlled_shared_1d_residual": controlled_shared_1d_residual,
    "controlled_shared_1d_residual_identity": controlled_shared_1d_residual_identity,
    "controlled_2d_cnn": controlled_2d_cnn,
}
