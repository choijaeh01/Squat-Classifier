from __future__ import annotations

from collections.abc import Callable

from torch import nn

from models.baselines import AllChannelConv1D, AllChannelConv1DSmall, CNN2DBaseline, ClassicalFeatureBaseline
from models.literature_temporal import (
    CNNGRULiteratureV1,
    CNNLSTMLiteratureV1,
    GRUOnlyLiteratureV1,
    LSTMOnlyLiteratureV1,
    ResCNNBiGRUAttentionLiteV1,
    TCNLiteratureV1,
    TransformerEncoderLiteV1,
)
from models.shared_encoders import (
    ChannelShared1DEncoder,
    ChannelSharedAttentionPoolV2,
    ChannelSharedMeanPoolV2,
    ChannelSharedPosResAttentionV3,
    ModalitySharedAccGyroEncoder,
    ModalitySharedMeanPoolV2,
    ModalitySharedSensorAttentionV3,
)


ModelFactory = Callable[..., nn.Module]


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "classical_feature_baseline": ClassicalFeatureBaseline,
    "all_channel_conv1d": AllChannelConv1D,
    "all_channel_conv1d_v1": AllChannelConv1D,
    "all_channel_conv1d_small": AllChannelConv1DSmall,
    "channel_shared_1d_encoder": ChannelShared1DEncoder,
    "channel_shared_meanpool_v2": ChannelSharedMeanPoolV2,
    "channel_shared_attentionpool_v2": ChannelSharedAttentionPoolV2,
    "channel_shared_posres_attention_v3": ChannelSharedPosResAttentionV3,
    "modality_shared_acc_gyro_encoder": ModalitySharedAccGyroEncoder,
    "modality_shared_meanpool_v2": ModalitySharedMeanPoolV2,
    "modality_shared_sensorattn_v3": ModalitySharedSensorAttentionV3,
    "cnn2d_baseline": CNN2DBaseline,
    "cnn2d_baseline_v1": CNN2DBaseline,
    "cnn_lstm_literature_v1": CNNLSTMLiteratureV1,
    "cnn_gru_literature_v1": CNNGRULiteratureV1,
    "rescnn_bigru_attention_lite_v1": ResCNNBiGRUAttentionLiteV1,
    "tcn_literature_v1": TCNLiteratureV1,
    "lstm_only_literature_v1": LSTMOnlyLiteratureV1,
    "gru_only_literature_v1": GRUOnlyLiteratureV1,
    "transformer_encoder_lite_v1": TransformerEncoderLiteV1,
    "feature_random_forest_v1": ClassicalFeatureBaseline,
    "feature_linear_svm_v1": ClassicalFeatureBaseline,
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


def count_parameter_groups(model: nn.Module) -> dict[str, int]:
    encoder_modules = [
        getattr(model, name)
        for name in ("encoder", "shared_encoder", "acc_encoder", "gyro_encoder")
        if hasattr(model, name)
    ]
    aggregation_modules = [
        getattr(model, name)
        for name in ("aggregation", "attention_pool")
        if hasattr(model, name)
    ]
    head_modules = [getattr(model, "classifier")] if hasattr(model, "classifier") else []
    encoder_params = _count_unique_module_parameters(encoder_modules)
    aggregation_params = _count_unique_module_parameters(aggregation_modules)
    head_params = _count_unique_module_parameters(head_modules)
    total_params = count_parameters(model)
    other_params = total_params - encoder_params - aggregation_params - head_params
    return {
        "total_params": total_params,
        "encoder_params": encoder_params,
        "aggregation_params": aggregation_params,
        "head_params": head_params,
        "other_params": other_params,
    }


def _count_unique_module_parameters(modules: list[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
    return total
