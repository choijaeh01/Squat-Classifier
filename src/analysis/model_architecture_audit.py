from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from models.channel_metadata import CHANNEL_ORDER
from models.controlled_models import CONTROLLED_NEURAL_MODEL_NAMES
from models.registry import build_model, count_parameters


ARCHITECTURE_MODELS = [
    "controlled_stats_mlp",
    "controlled_all_channel_1d_cnn",
    "controlled_all_channel_1d_cnn_small",
    "controlled_shared_1d",
    "controlled_shared_1d_identity",
    "controlled_shared_1d_residual",
    "controlled_shared_1d_residual_identity",
    "controlled_2d_cnn",
    "rescnn_bigru_attention_lite_v1",
    "lee_style_cnn_lstm_2d_v1",
]


def load_display_mapping(path: Path) -> dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data["mapping"])


def run_architecture_audit(
    *,
    output_dir: Path,
    mapping_path: Path,
    input_length: int = 512,
    input_channels: int = 18,
    num_classes: int = 5,
) -> dict[str, Any]:
    """Build each target model and run a forward-only architecture audit.

    This function intentionally performs no optimizer step, no loss.backward, and
    no training loop. It only inspects initialized modules with synthetic input.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_display_mapping(mapping_path)
    architecture_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    common_head_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    residual_feature_rows = residual_feature_list_rows()

    x = torch.zeros((2, input_length, input_channels), dtype=torch.float32)
    for model_name in ARCHITECTURE_MODELS:
        model = build_model(model_name, input_length=input_length, input_channels=input_channels, num_classes=num_classes)
        model.eval()
        captured: list[dict[str, Any]] = []
        handles = register_shape_hooks(model, captured)
        with torch.no_grad():
            output = model(x)
        for handle in handles:
            handle.remove()
        representation_shape = ""
        representation_dim = ""
        if hasattr(model, "extract_features"):
            with torch.no_grad():
                representation = model.extract_features(x)
            representation_shape = shape_to_string(representation)
            representation_dim = int(representation.shape[-1])

        params = parameter_breakdown(model)
        display_name = mapping.get(model_name, model_name)
        structure = structure_summary(model_name, model)
        branch_names = branch_summary(model)
        output_shape = shape_to_string(output)
        architecture_rows.append(
            {
                "model_internal_name": model_name,
                "model_display_name": display_name,
                "input_shape": "(2, 512, 18)",
                "extractor_structure_summary": structure,
                "branch_names": branch_names,
                "intermediate_output_shapes": compact_shape_summary(captured),
                "representation_dim": representation_dim,
                "representation_shape": representation_shape,
                "common_head_structure": common_head_signature(model),
                "total_params": params["total_params"],
                "extractor_params": params["extractor_params"],
                "common_head_params": params["common_head_params"],
                "residual_branch_params": params["residual_branch_params"],
                "identity_embedding_params": params["identity_embedding_params"],
                "output_shape": output_shape,
                "notes": model_notes(model_name),
            }
        )
        parameter_rows.append(
            {
                "model_internal_name": model_name,
                "model_display_name": display_name,
                "total_params": params["total_params"],
                "extractor_params": params["extractor_params"],
                "common_head_params": params["common_head_params"],
                "residual_branch_params": params["residual_branch_params"],
                "identity_embedding_params": params["identity_embedding_params"],
                "representation_dim": representation_dim,
                "notes": params["notes"],
            }
        )
        for idx, row in enumerate(captured, start=1):
            row = dict(row)
            row["model_internal_name"] = model_name
            row["model_display_name"] = display_name
            row["layer_index"] = idx
            shape_rows.append(row)
        if model_name in CONTROLLED_NEURAL_MODEL_NAMES:
            common_head_rows.append(
                {
                    "model_display_name": display_name,
                    "representation_dim": representation_dim,
                    "common_head_params": params["common_head_params"],
                    "common_head_signature": common_head_signature(model),
                    "representation_dim_is_64": str(representation_dim == 64),
                    "common_head_param_count_is_4485": str(params["common_head_params"] == 4485),
                    "common_head_signature_matches": str(common_head_signature(model) == "linear:64->64|relu|dropout:0.1|linear:64->5"),
                }
            )

    write_csv(output_dir / "table_architecture_audit.csv", architecture_rows)
    write_csv(output_dir / "table_parameter_count_clean.csv", parameter_rows)
    write_csv(output_dir / "table_common_head_verification.csv", common_head_rows)
    write_csv(output_dir / "table_residual_feature_list.csv", residual_feature_rows)
    write_csv(output_dir / "table_layer_shape_summary.csv", shape_rows)
    return {
        "output_dir": str(output_dir),
        "models": len(architecture_rows),
        "common_head_models": len(common_head_rows),
        "residual_feature_rows": len(residual_feature_rows),
        "shape_rows": len(shape_rows),
    }


def register_shape_hooks(model: nn.Module, captured: list[dict[str, Any]]) -> list[Any]:
    handles = []
    for name, module in model.named_modules():
        if name == "" or any(module.children()):
            continue

        def hook(mod: nn.Module, inputs: tuple[Any, ...], output: Any, *, module_name: str = name) -> None:
            captured.append(
                {
                    "module_path": module_name,
                    "module_type": mod.__class__.__name__,
                    "input_shape": shape_to_string(inputs[0]) if inputs else "",
                    "output_shape": shape_to_string(output),
                    "param_count": count_module_params(mod),
                }
            )

        handles.append(module.register_forward_hook(hook))
    return handles


def shape_to_string(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(int(item) for item in value.shape))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(shape_to_string(item) for item in value) + "]"
    return type(value).__name__


def compact_shape_summary(rows: list[dict[str, Any]], max_items: int = 10) -> str:
    items = [f"{row['module_path']}:{row['output_shape']}" for row in rows[:max_items]]
    if len(rows) > max_items:
        items.append(f"... {len(rows) - max_items} more")
    return " | ".join(items)


def count_module_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += int(parameter.numel())
    return total


def parameter_breakdown(model: nn.Module) -> dict[str, Any]:
    total = count_parameters(model)
    common_head = count_module_params(getattr(model, "classifier", None)) if hasattr(model, "extractor") else 0
    extractor = count_module_params(getattr(model, "extractor", None)) if hasattr(model, "extractor") else max(0, total - count_module_params(getattr(model, "classifier", None)))
    residual = 0
    identity = 0
    if hasattr(model, "extractor"):
        extractor_module = getattr(model, "extractor")
        residual += count_module_params(getattr(extractor_module, "residual_projection", None))
        residual += count_module_params(getattr(extractor_module, "fusion_projection", None))
        identity += count_module_params(getattr(extractor_module, "identity_embeddings", None))
    return {
        "total_params": total,
        "extractor_params": extractor,
        "common_head_params": common_head,
        "residual_branch_params": residual,
        "identity_embedding_params": identity,
        "notes": "controlled common-head model" if hasattr(model, "extractor") else "literature/reference model; no controlled common head",
    }


def common_head_signature(model: nn.Module) -> str:
    classifier = getattr(model, "classifier", None)
    if classifier is not None and hasattr(classifier, "architecture_signature"):
        return "|".join(classifier.architecture_signature())
    return "not_applicable"


def structure_summary(model_name: str, model: nn.Module) -> str:
    summaries = {
        "controlled_stats_mlp": "raw summary statistics mean/std/min/max -> Linear 72->64 -> ReLU -> common head",
        "controlled_all_channel_1d_cnn": "Conv1D(18->32,k9) -> BN/ReLU -> Conv1D(32->64,k5) -> BN/ReLU -> global average pooling -> common head",
        "controlled_all_channel_1d_cnn_small": "Conv1D(18->16,k7) -> BN/ReLU -> Conv1D(16->32,k5) -> BN/ReLU -> global average pooling -> Linear 32->64 -> common head",
        "controlled_shared_1d": "18 single-channel streams -> one shared TemporalEncoder1D -> token mean pooling -> common head",
        "controlled_shared_1d_identity": "shared TemporalEncoder1D tokens + channel/sensor/modality/axis embeddings -> LayerNorm -> mean pooling -> common head",
        "controlled_shared_1d_residual": "shared TemporalEncoder1D branch + raw summary residual branch -> fusion projection -> common head",
        "controlled_shared_1d_residual_identity": "shared TemporalEncoder1D with identity embeddings + raw summary residual branch -> fusion projection -> common head",
        "controlled_2d_cnn": "time x channel matrix -> Conv2D blocks -> global pooling -> Linear 32->64 -> common head",
        "rescnn_bigru_attention_lite_v1": "Residual Conv1D blocks -> MaxPool -> BiGRU -> temporal attention -> classifier",
        "lee_style_cnn_lstm_2d_v1": "internal 512->40 adaptive average downsampling -> 2D Conv blocks -> channel pooling -> LSTM -> classifier",
    }
    return summaries.get(model_name, model.__class__.__name__)


def branch_summary(model: nn.Module) -> str:
    if not hasattr(model, "extractor"):
        branches = []
        for name in ("encoder", "recurrent", "aggregation", "classifier"):
            if hasattr(model, name):
                branches.append(name)
        return ", ".join(branches)
    extractor = getattr(model, "extractor")
    branches = ["extractor"]
    if hasattr(extractor, "shared_encoder"):
        branches.append("shared_encoder")
    if getattr(extractor, "use_identity", False):
        branches.append("identity_embeddings")
    if getattr(extractor, "use_residual", False) or hasattr(extractor, "residual_projection"):
        branches.append("residual_projection")
    branches.append("common_head")
    return ", ".join(branches)


def model_notes(model_name: str) -> str:
    if model_name == "controlled_stats_mlp":
        return "Uses the same mean/std/min/max signal-derived summary as the residual branch."
    if model_name == "controlled_shared_1d_residual":
        return "One shared encoder object is reused across 18 channels; residual summary branch preserves channel-specific statistics."
    if model_name == "lee_style_cnn_lstm_2d_v1":
        return "Clean-room adapted baseline, not exact Lee et al. reproduction."
    return ""


def residual_feature_list_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statistics = ["mean", "std", "min", "max"]
    for statistic in statistics:
        for channel_index, channel_name in enumerate(CHANNEL_ORDER):
            rows.append(
                {
                    "feature_name": f"{channel_name}_{statistic}",
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "statistic": statistic,
                    "used_by": "Statistical Summary MLP; Residual Channel-Shared Encoder residual branch",
                    "source": "scaled IMU signal after train-only StandardScaler transform",
                    "uses_metadata": "False",
                    "uses_label": "False",
                    "uses_subject_id": "False",
                    "allowed": "True",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
