from __future__ import annotations

from dataclasses import dataclass

import torch


CHANNEL_ORDER = [
    "s0_ax",
    "s0_ay",
    "s0_az",
    "s0_gx",
    "s0_gy",
    "s0_gz",
    "s1_ax",
    "s1_ay",
    "s1_az",
    "s1_gx",
    "s1_gy",
    "s1_gz",
    "s2_ax",
    "s2_ay",
    "s2_az",
    "s2_gx",
    "s2_gy",
    "s2_gz",
]

SENSOR_NAMES = ["s0_lower_back", "s1_right_thigh", "s2_right_calf"]
MODALITY_NAMES = ["acc", "gyro"]
AXIS_NAMES = ["x", "y", "z"]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    sensor_id: int
    modality_id: int
    axis_id: int


def build_target_channels() -> list[ChannelSpec]:
    specs: list[ChannelSpec] = []
    for name in CHANNEL_ORDER:
        sensor_token, measurement = name.split("_")
        sensor_id = int(sensor_token[1:])
        modality_token = measurement[0]
        axis_token = measurement[1]
        modality_id = 0 if modality_token == "a" else 1
        axis_id = AXIS_NAMES.index(axis_token)
        specs.append(ChannelSpec(name=name, sensor_id=sensor_id, modality_id=modality_id, axis_id=axis_id))
    return specs


TARGET_CHANNELS = build_target_channels()


def validate_channel_order(channel_order: list[str]) -> None:
    if list(channel_order) != CHANNEL_ORDER:
        raise ValueError(f"channel order mismatch: expected {CHANNEL_ORDER}, got {channel_order}")


def sensor_indices() -> list[int]:
    return [spec.sensor_id for spec in TARGET_CHANNELS]


def modality_indices() -> list[int]:
    return [spec.modality_id for spec in TARGET_CHANNELS]


def axis_indices() -> list[int]:
    return [spec.axis_id for spec in TARGET_CHANNELS]


def acc_channel_indices() -> list[int]:
    return [index for index, spec in enumerate(TARGET_CHANNELS) if spec.modality_id == 0]


def gyro_channel_indices() -> list[int]:
    return [index for index, spec in enumerate(TARGET_CHANNELS) if spec.modality_id == 1]


def metadata_tensors(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(sensor_indices(), dtype=torch.long, device=device),
        torch.as_tensor(modality_indices(), dtype=torch.long, device=device),
        torch.as_tensor(axis_indices(), dtype=torch.long, device=device),
    )
