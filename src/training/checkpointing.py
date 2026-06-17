from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_name: str,
    fold_id: int,
    epoch: int,
    val_macro_f1: float,
    config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "fold_id": int(fold_id),
            "epoch": int(epoch),
            "val_macro_f1": float(val_macro_f1),
            "config_experiment_name": str(config.get("experiment_name", "")),
        },
        path,
    )


def load_model_state(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return dict(payload)
