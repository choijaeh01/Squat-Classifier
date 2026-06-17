from __future__ import annotations

from collections.abc import Callable

from torch import nn

from models.baselines import AllChannelConv1D, CNN2DBaseline, ClassicalFeatureBaseline
from models.shared_encoders import ChannelShared1DEncoder, ModalitySharedAccGyroEncoder


ModelFactory = Callable[..., nn.Module]


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "classical_feature_baseline": ClassicalFeatureBaseline,
    "all_channel_conv1d": AllChannelConv1D,
    "channel_shared_1d_encoder": ChannelShared1DEncoder,
    "modality_shared_acc_gyro_encoder": ModalitySharedAccGyroEncoder,
    "cnn2d_baseline": CNN2DBaseline,
}


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


def build_model(name: str, **kwargs) -> nn.Module:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"unknown model {name!r}; available models: {available}") from exc
    return factory(**kwargs)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
