from __future__ import annotations

import csv
import json
import math
import random
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from analysis.professor_report_v2_assets import (
    CONTROLLED_DIR,
    display,
    fnum,
    fmt,
    load_display_mapping,
    markdown_table,
    markdown_table_pipe_errors,
    missing_image_links,
    read_csv,
    signed,
    write_csv,
    wrap_label,
)


FLOW_MODELS = {
    "shared": "controlled_shared_1d",
    "identity": "controlled_shared_1d_identity",
    "residual": "controlled_shared_1d_residual",
    "residual_identity": "controlled_shared_1d_residual_identity",
}


@dataclass(frozen=True)
class ProfessorReportV3Paths:
    project_root: Path
    output_docs: Path
    output_obsidian: Path | None
    mapping_path: Path

    @property
    def tables(self) -> Path:
        return self.output_docs / "tables"

    @property
    def figures(self) -> Path:
        return self.output_docs / "figures"

    @property
    def diagrams(self) -> Path:
        return self.output_docs / "diagrams"

    @property
    def assets(self) -> Path:
        return self.output_docs / "assets"


def check_professor_report_v3_inputs(project_root: Path) -> dict[str, Any]:
    required = [
        CONTROLLED_DIR / "fold_metrics.csv",
        CONTROLLED_DIR / "aggregate_metrics_by_model.csv",
        CONTROLLED_DIR / "paired_model_differences.csv",
        Path("docs/professor_report_v3/display_name_mapping.yaml"),
    ]
    status = {str(path): (project_root / path).exists() for path in required}
    return {"all_present": all(status.values()), "files": status}


def build_professor_report_v3_assets(paths: ProfessorReportV3Paths) -> dict[str, Any]:
    for folder in (paths.output_docs, paths.tables, paths.figures, paths.diagrams, paths.assets):
        folder.mkdir(parents=True, exist_ok=True)
    mapping = load_display_mapping(paths.mapping_path)
    sources = load_sources(paths.project_root)

    tables = build_tables(paths, sources, mapping)
    figures = build_figures(paths, mapping)
    diagrams = build_diagrams(paths)
    captions = write_captions(paths)
    report = write_main_report(paths)
    index = write_figure_table_index(paths)
    suggestions = write_v2_revision_suggestions(paths)
    readme = write_readme(paths)
    validation = validate_report_v3(paths)
    validation_path = paths.assets / "professor_report_v3_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "output_docs": str(paths.output_docs),
        "output_obsidian": str(paths.output_obsidian) if paths.output_obsidian else "",
        "tables": [str(item) for item in tables],
        "figures": [str(item) for item in figures],
        "diagrams": [str(item) for item in diagrams],
        "captions": str(captions),
        "main_report": str(report),
        "figure_table_index": str(index),
        "v2_revision_suggestions": str(suggestions),
        "readme": str(readme),
        "validation_path": str(validation_path),
        "validation": validation,
    }


def load_sources(project_root: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "fold_metrics": read_csv(project_root / CONTROLLED_DIR / "fold_metrics.csv"),
        "aggregate": read_csv(project_root / CONTROLLED_DIR / "aggregate_metrics_by_model.csv"),
        "paired": read_csv(project_root / CONTROLLED_DIR / "paired_model_differences.csv"),
    }


def build_tables(paths: ProfessorReportV3Paths, sources: dict[str, list[dict[str, str]]], mapping: dict[str, str]) -> list[Path]:
    observations = paired_observations(sources["fold_metrics"])
    aggregates = {row["model_name"]: row for row in sources["aggregate"]}
    effects = compute_effects(observations)
    outputs = [
        write_csv(paths.tables / "table_identity_residual_2x2.csv", table_2x2_rows(aggregates, mapping)),
        write_csv(paths.tables / "table_identity_residual_effects.csv", effect_rows(effects)),
        write_csv(paths.tables / "table_identity_residual_paired_details.csv", paired_detail_rows(observations)),
        write_csv(paths.tables / "table_story_numbers.csv", story_number_rows(aggregates, effects, mapping)),
        write_csv(paths.tables / "table_claim_adjustment_v3.csv", claim_adjustment_rows()),
        write_csv(paths.tables / "table_internal_name_mapping.csv", [{"internal_model_name": key, "display_name": value} for key, value in sorted(mapping.items())]),
    ]
    return outputs


def paired_observations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    values: dict[str, dict[tuple[str, str], float]] = {name: {} for name in FLOW_MODELS.values()}
    for row in rows:
        model_name = row.get("model_name", "")
        if model_name not in values or row.get("status") != "ok":
            continue
        key = (row["seed"], row["fold_id"])
        values[model_name][key] = fnum(row["test_macro_f1"])
    common_keys = sorted(set.intersection(*(set(items) for items in values.values())), key=lambda item: (int(item[0]), int(item[1])))
    observations = []
    for seed, fold_id in common_keys:
        shared = values[FLOW_MODELS["shared"]][(seed, fold_id)]
        identity = values[FLOW_MODELS["identity"]][(seed, fold_id)]
        residual = values[FLOW_MODELS["residual"]][(seed, fold_id)]
        residual_identity = values[FLOW_MODELS["residual_identity"]][(seed, fold_id)]
        observations.append(
            {
                "seed": seed,
                "fold_id": fold_id,
                "Shared 1D Encoder": shared,
                "Shared 1D + Identity": identity,
                "Residual Channel-Shared Encoder": residual,
                "Residual Channel-Shared + Identity": residual_identity,
                "identity_effect_without_residual": identity - shared,
                "identity_effect_with_residual": residual_identity - residual,
                "residual_effect_without_identity": residual - shared,
                "residual_effect_with_identity": residual_identity - identity,
                "interaction_effect": (residual_identity - residual) - (identity - shared),
            }
        )
    return observations


def compute_effects(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("identity_effect_without_residual", "Shared 1D + Identity - Shared 1D", "Identity는 residual이 없을 때 token origin 손실을 일부 보완한다."),
        ("identity_effect_with_residual", "Residual + Identity - Residual", "Residual branch가 들어간 뒤 identity 추가 이득은 거의 없거나 약간 음수다."),
        ("residual_effect_without_identity", "Residual - Shared 1D", "Residual branch는 shared-only 병목을 가장 크게 완화한다."),
        ("residual_effect_with_identity", "Residual + Identity - Shared 1D + Identity", "Identity가 있어도 residual branch의 추가 효과는 크다."),
        ("interaction_effect", "(Residual+Identity - Residual) - (Identity - Shared)", "음수 interaction은 residual branch가 identity embedding의 일부 역할을 대체했을 가능성을 시사한다."),
    ]
    rows = []
    for key, contrast, note in specs:
        values = [float(obs[key]) for obs in observations]
        ci_low, ci_high = bootstrap_ci(values)
        rows.append(
            {
                "effect_name": key,
                "contrast": contrast,
                "mean_delta": mean(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_pairs": len(values),
                "interpretation_note": note,
            }
        )
    return rows


def table_2x2_rows(aggregates: dict[str, dict[str, str]], mapping: dict[str, str]) -> list[dict[str, str]]:
    def macro(model_name: str) -> float:
        return fnum(aggregates[model_name]["mean_macro_f1"])

    rows = [
        {
            "residual_branch": "Absent",
            "identity_absent_model": display(FLOW_MODELS["shared"], mapping),
            "identity_absent_macro_f1": fmt(macro(FLOW_MODELS["shared"])),
            "identity_present_model": display(FLOW_MODELS["identity"], mapping),
            "identity_present_macro_f1": fmt(macro(FLOW_MODELS["identity"])),
            "identity_effect": signed(macro(FLOW_MODELS["identity"]) - macro(FLOW_MODELS["shared"])),
            "interpretation": "identity가 shared-only token origin 손실을 일부 보완한다.",
        },
        {
            "residual_branch": "Present",
            "identity_absent_model": display(FLOW_MODELS["residual"], mapping),
            "identity_absent_macro_f1": fmt(macro(FLOW_MODELS["residual"])),
            "identity_present_model": display(FLOW_MODELS["residual_identity"], mapping),
            "identity_present_macro_f1": fmt(macro(FLOW_MODELS["residual_identity"])),
            "identity_effect": signed(macro(FLOW_MODELS["residual_identity"]) - macro(FLOW_MODELS["residual"])),
            "interpretation": "residual branch가 있으면 identity 추가 이득은 작다.",
        },
    ]
    return rows


def effect_rows(effects: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "effect_name": row["effect_name"],
            "contrast": row["contrast"],
            "mean_delta": f"{row['mean_delta']:.4f}",
            "bootstrap_ci": f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
            "n_pairs": str(row["n_pairs"]),
            "interpretation_note": row["interpretation_note"],
        }
        for row in effects
    ]


def paired_detail_rows(observations: list[dict[str, Any]]) -> list[dict[str, str]]:
    columns = [
        "seed",
        "fold_id",
        "Shared 1D Encoder",
        "Shared 1D + Identity",
        "Residual Channel-Shared Encoder",
        "Residual Channel-Shared + Identity",
        "identity_effect_without_residual",
        "identity_effect_with_residual",
        "residual_effect_without_identity",
        "residual_effect_with_identity",
        "interaction_effect",
    ]
    rows = []
    for obs in observations:
        row = {}
        for col in columns:
            value = obs[col]
            row[col] = value if col in {"seed", "fold_id"} else f"{float(value):.4f}"
        rows.append(row)
    return rows


def story_number_rows(aggregates: dict[str, dict[str, str]], effects: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for key in ["shared", "identity", "residual", "residual_identity"]:
        model = FLOW_MODELS[key]
        rows.append(
            {
                "item": display(model, mapping),
                "value": fmt(aggregates[model]["mean_macro_f1"]),
                "note": "controlled comparison, 3 seeds x 6 folds",
            }
        )
    for effect in effects:
        rows.append({"item": effect["effect_name"], "value": f"{effect['mean_delta']:.4f}", "note": f"bootstrap CI [{effect['ci_low']:.4f}, {effect['ci_high']:.4f}], n={effect['n_pairs']}"})
    return rows


def claim_adjustment_rows() -> list[dict[str, str]]:
    return [
        {"claim": "Residual branch mitigates shared encoder bottleneck.", "status": "safe", "reason": "Shared 1D 0.2806에서 Residual Channel-Shared 0.8004로 증가.", "safe_wording": "잔차 통계 branch는 shared encoder 병목을 크게 완화했다."},
        {"claim": "Identity embedding is the main contribution.", "status": "avoid", "reason": "identity는 residual이 없을 때만 크게 개선되고 residual 이후 추가 이득은 작음.", "safe_wording": "identity는 shared-only 구조에서 일부 보완 효과가 있었지만 핵심 claim은 residual branch다."},
        {"claim": "Residual Channel-Shared is statistically superior to all baselines.", "status": "avoid", "reason": "Statistical Summary MLP, XGBoost, RF와 점수가 가깝고 CI overlap 가능.", "safe_wording": "강한 practical/neural baseline과 경쟁 가능한 성능을 보였다."},
        {"claim": "Summary statistics are important in this dataset.", "status": "safe", "reason": "Statistical Summary MLP, RF/XGBoost, residual branch가 모두 높은 성능.", "safe_wording": "이 데이터셋에서는 channel-wise summary statistics가 강한 signal로 관찰되었다."},
    ]


def build_figures(paths: ProfessorReportV3Paths, mapping: dict[str, str]) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    table_2x2 = read_csv(paths.tables / "table_identity_residual_2x2.csv")
    effects = read_csv(paths.tables / "table_identity_residual_effects.csv")
    outputs = [
        draw_identity_residual_grid(plt, table_2x2, paths.figures / "fig_identity_residual_2x2_grid.png"),
        draw_effect_arrows(plt, effects, paths.figures / "fig_identity_residual_effect_arrows.png"),
        draw_information_bottleneck(plt, paths.figures / "fig_shared_encoder_information_bottleneck.png"),
    ]
    return outputs


def draw_identity_residual_grid(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    values = [
        [fnum(rows[0]["identity_absent_macro_f1"]), fnum(rows[0]["identity_present_macro_f1"])],
        [fnum(rows[1]["identity_absent_macro_f1"]), fnum(rows[1]["identity_present_macro_f1"])],
    ]
    labels = [
        ["Shared 1D Encoder", "Shared 1D + Identity"],
        ["Residual Channel-Shared Encoder", "Residual Channel-Shared + Identity"],
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.imshow(values, vmin=0.25, vmax=0.85, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Identity absent", "Identity present"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Residual absent", "Residual present"])
    ax.set_title("Identity helps when residual statistics are absent; residual dominates when present")
    for y in range(2):
        for x in range(2):
            ax.text(x, y - 0.08, labels[y][x], ha="center", va="center", fontsize=9)
            ax.text(x, y + 0.12, f"{values[y][x]:.4f}", ha="center", va="center", fontsize=18, fontweight="bold")
    arrowprops = {"arrowstyle": "->", "linewidth": 1.5, "color": "black"}
    ax.annotate("", xy=(1.05, -0.38), xytext=(-0.05, -0.38), arrowprops=arrowprops)
    ax.text(0.5, -0.48, "Identity effect without residual: +0.2376", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(1.05, 1.28), xytext=(-0.05, 1.28), arrowprops=arrowprops)
    ax.text(0.5, 1.38, "Identity effect with residual: -0.0031", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(-0.48, 1.05), xytext=(-0.48, -0.05), arrowprops=arrowprops)
    ax.text(-0.66, 0.5, "Residual effect\nwithout identity\n+0.5198", ha="center", va="center", fontsize=9, rotation=90)
    ax.annotate("", xy=(1.48, 1.05), xytext=(1.48, -0.05), arrowprops=arrowprops)
    ax.text(1.68, 0.5, "Residual effect\nwith identity\n+0.2791", ha="center", va="center", fontsize=9, rotation=90)
    ax.set_xlim(-0.85, 1.85)
    ax.set_ylim(1.65, -0.65)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_effect_arrows(plt: Any, rows: list[dict[str, str]], path: Path) -> Path:
    order = [
        "residual_effect_without_identity",
        "residual_effect_with_identity",
        "identity_effect_without_residual",
        "identity_effect_with_residual",
        "interaction_effect",
    ]
    by_name = {row["effect_name"]: row for row in rows}
    labels = {
        "residual_effect_without_identity": "Residual effect\nwithout identity",
        "residual_effect_with_identity": "Residual effect\nwith identity",
        "identity_effect_without_residual": "Identity effect\nwithout residual",
        "identity_effect_with_residual": "Identity effect\nwith residual",
        "interaction_effect": "Interaction\neffect",
    }
    values = [fnum(by_name[name]["mean_delta"]) for name in order]
    lows, highs = [], []
    for name, value in zip(order, values):
        ci = by_name[name]["bootstrap_ci"].strip("[]").split(",")
        lo, hi = float(ci[0]), float(ci[1])
        lows.append(value - lo)
        highs.append(hi - value)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = list(range(len(order)))
    ax.barh(y, values, xerr=[lows, highs], capsize=4)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[name] for name in order])
    ax.set_xlabel("Mean paired delta in Macro F1")
    ax.set_title("Identity and Residual Effects from 18 Seed-Fold Pairs")
    for idx, value in enumerate(values):
        ax.text(value + (0.015 if value >= 0 else -0.015), idx, f"{value:+.4f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def draw_information_bottleneck(plt: Any, path: Path) -> Path:
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")
    ax.set_title("Shared Encoder Information Bottleneck View", fontsize=14)
    panels = [
        ("Shared 1D Encoder", ["512x18 input", "18 single-channel streams", "same fθ reused", "mean token pooling", "weak channel-specific cues"], "Macro F1 0.2806"),
        ("Shared 1D + Identity", ["shared tokens", "+ channel/sensor/modality/axis embedding", "token origin partly restored", "no explicit statistics"], "Macro F1 0.5182"),
        ("Residual Channel-Shared", ["shared temporal branch", "+ mean/std/min/max per channel", "72 fixed-order statistics", "fusion to 64-dim"], "Macro F1 0.8004"),
    ]
    for col, (title, nodes, footer) in enumerate(panels):
        x0 = 0.04 + col * 0.32
        ax.text(x0 + 0.14, 0.9, title, ha="center", fontsize=11, fontweight="bold")
        for idx, node in enumerate(nodes):
            y = 0.75 - idx * 0.13
            box = FancyBboxPatch((x0, y), 0.28, 0.07, boxstyle="round,pad=0.015", facecolor="#f3f6fb", edgecolor="#4b5563")
            ax.add_patch(box)
            ax.text(x0 + 0.14, y + 0.035, node, ha="center", va="center", fontsize=8)
            if idx < len(nodes) - 1:
                ax.annotate("", xy=(x0 + 0.14, y - 0.035), xytext=(x0 + 0.14, y), arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.text(x0 + 0.14, 0.12, footer, ha="center", fontsize=10)
    ax.text(0.5, 0.03, "Interpretation: identity restores token origin, while the residual branch directly preserves channel-wise statistical cues.", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_diagrams(paths: ProfessorReportV3Paths) -> list[Path]:
    outputs = []
    specs = [
        (
            "diagram_architecture_comparison_four_models",
            architecture_comparison_mermaid(),
            draw_architecture_comparison_png,
        ),
        (
            "diagram_residual_branch_detailed",
            residual_branch_mermaid(),
            draw_residual_branch_png,
        ),
        (
            "diagram_identity_embedding_detailed",
            identity_embedding_mermaid(),
            draw_identity_embedding_png,
        ),
        (
            "diagram_common_head_detailed_v3",
            common_head_mermaid(),
            draw_common_head_png,
        ),
    ]
    for name, mermaid_text, draw_func in specs:
        mmd = paths.diagrams / f"{name}.mmd"
        png = paths.diagrams / f"{name}.png"
        mmd.write_text(mermaid_text, encoding="utf-8")
        draw_func(png)
        outputs.extend([mmd, png])
    return outputs


def architecture_comparison_mermaid() -> str:
    return """flowchart LR
  subgraph A["A. All-Channel 1D CNN"]
    A0["Input 512x18"] --> A1["Conv1D sees all 18 channels jointly"] --> A2["Channel identity preserved through joint input"] --> A3["Common MLP head"]
  end
  subgraph B["B. Shared 1D Encoder"]
    B0["Input 512x18"] --> B1["18 single-channel streams"] --> B2["same encoder f_theta reused"] --> B3["token pooling"] --> B4["channel cues can be weakened"] --> B5["Common MLP head"]
  end
  subgraph C["C. Shared 1D + Identity"]
    C0["shared tokens"] --> C1["add channel/sensor/modality/axis embeddings"] --> C2["token origin partly restored"] --> C3["still lacks explicit statistics"] --> C4["Common MLP head"]
  end
  subgraph D["D. Residual Channel-Shared Encoder"]
    D0["shared encoder branch"] --> D1["parallel train-scaled channel-wise statistics"] --> D2["mean/std/min/max, 18x4=72 features"] --> D3["fusion to 64-dim"] --> D4["Common MLP head"]
  end
"""


def residual_branch_mermaid() -> str:
    return """flowchart TD
  A["Train-scaled tensor, 512x18"] --> B["Per-channel mean"]
  A --> C["Per-channel std"]
  A --> D["Per-channel min"]
  A --> E["Per-channel max"]
  B --> F["72 signal-derived features"]
  C --> F
  D --> F
  E --> F
  F --> G["Projection / fusion"]
  G --> H["64-dim representation"]
  I["No metadata, label, subject id, filename, boundary, or original length"] -.-> F
"""


def identity_embedding_mermaid() -> str:
    return """flowchart TD
  A["Shared encoder token"] --> B["Add identity metadata embeddings"]
  C["channel_id"] --> B
  D["sensor_id"] --> B
  E["modality_id"] --> B
  F["axis_id"] --> B
  B --> G["LayerNorm(token + metadata)"]
  G --> H["Token origin is partly restored"]
  H --> I["Raw statistics are not directly provided"]
"""


def common_head_mermaid() -> str:
    return """flowchart TD
  A["64-dim feature representation"] --> B["Linear 64 to 64"]
  B --> C["ReLU"]
  C --> D["Dropout 0.1"]
  D --> E["Linear 64 to 5"]
  E --> F["5-class logits"]
  G["Same head for controlled neural models, 4,485 params"] -.-> B
"""


def draw_architecture_comparison_png(path: Path) -> None:
    draw_multi_column_diagram(
        path,
        "Architecture Comparison: Information Flow",
        [
            ("All-Channel 1D CNN", ["Input 512x18", "Conv1D jointly sees 18 channels", "Channel identity preserved", "Common MLP head"], "Learns joint channel patterns directly."),
            ("Shared 1D Encoder", ["Input 512x18", "18 single-channel streams", "same fθ reused", "token pooling", "Common MLP head"], "Parameter sharing but channel cues can weaken."),
            ("Shared 1D + Identity", ["Shared tokens", "+ channel/sensor/modality/axis embeddings", "LayerNorm token + metadata", "Common MLP head"], "Token origin partly restored."),
            ("Residual Channel-Shared", ["Shared encoder branch", "parallel train-scaled statistics", "mean/std/min/max = 72 features", "fusion to 64-dim", "Common MLP head"], "Channel-wise statistical cues preserved."),
        ],
    )


def draw_residual_branch_png(path: Path) -> None:
    draw_single_flow_diagram(
        path,
        "Residual Branch: Train-Scaled Channel-Wise Statistics",
        ["Train-scaled tensor 512x18", "Per channel mean/std/min/max", "18 channels x 4 statistics = 72 features", "Projection / fusion", "64-dim representation"],
        "Allowed inputs: IMU signal only. No metadata, label, subject id, boundary, filename, or original length.",
    )


def draw_identity_embedding_png(path: Path) -> None:
    draw_single_flow_diagram(
        path,
        "Identity Embedding: Token Origin Information",
        ["Shared encoder token", "Add channel_id embedding", "Add sensor_id embedding", "Add modality_id embedding", "Add axis_id embedding", "LayerNorm(token + metadata)", "Token origin partly restored"],
        "Identity embeddings do not directly provide raw summary statistics.",
    )


def draw_common_head_png(path: Path) -> None:
    draw_single_flow_diagram(
        path,
        "Common Classifier Head",
        ["64-dim representation", "Linear 64->64", "ReLU", "Dropout 0.1", "Linear 64->5", "5-class logits"],
        "Same classifier head for controlled neural models: 4,485 parameters.",
    )


def draw_multi_column_diagram(path: Path, title: str, columns: list[tuple[str, list[str], str]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=12)
    width = 0.21
    gap = 0.03
    for col, (heading, nodes, note) in enumerate(columns):
        x0 = 0.03 + col * (width + gap)
        ax.text(x0 + width / 2, 0.9, wrap_label(heading, 22), ha="center", fontsize=10, fontweight="bold")
        for idx, node in enumerate(nodes):
            y = 0.78 - idx * 0.12
            box = FancyBboxPatch((x0, y), width, 0.07, boxstyle="round,pad=0.012", facecolor="#f3f6fb", edgecolor="#4b5563")
            ax.add_patch(box)
            ax.text(x0 + width / 2, y + 0.035, wrap_label(node, 24), ha="center", va="center", fontsize=7.5)
            if idx < len(nodes) - 1:
                ax.annotate("", xy=(x0 + width / 2, y - 0.035), xytext=(x0 + width / 2, y), arrowprops={"arrowstyle": "->", "lw": 0.8})
        ax.text(x0 + width / 2, 0.08, wrap_label(note, 30), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_single_flow_diagram(path: Path, title: str, nodes: list[str], note: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(nodes) * 0.75)))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    for idx, node in enumerate(nodes):
        y = 0.85 - idx * 0.12
        box = FancyBboxPatch((0.22, y), 0.56, 0.065, boxstyle="round,pad=0.012", facecolor="#f3f6fb", edgecolor="#4b5563")
        ax.add_patch(box)
        ax.text(0.5, y + 0.032, node, ha="center", va="center", fontsize=9)
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(0.5, y - 0.035), xytext=(0.5, y), arrowprops={"arrowstyle": "->", "lw": 0.9})
    ax.text(0.5, 0.04, wrap_label(note, 95), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_captions(paths: ProfessorReportV3Paths) -> Path:
    path = paths.figures / "captions_v3.md"
    path.write_text(
        """# Figure Captions v3

## fig_identity_residual_2x2_grid.png
- Korean caption: Identity embedding은 residual branch가 없을 때 channel identity 손실을 일부 보완하지만, residual branch가 들어가면 channel-wise 통계 정보가 이미 위치 정보를 강하게 제공하기 때문에 추가 이득이 작다.
- English caption: Identity embeddings help when residual statistics are absent, but their incremental benefit becomes small once the residual branch provides fixed-order channel-wise statistics.
- 핵심 메시지: claim은 identity 중심이 아니라 residual branch의 shared encoder bottleneck 완화로 두는 것이 안전하다.

## fig_identity_residual_effect_arrows.png
- Korean caption: 18개 seed-fold paired observation에서 계산한 identity/residual effect와 bootstrap CI.
- English caption: Identity and residual effects with bootstrap confidence intervals from 18 seed-fold paired observations.
- 핵심 메시지: residual effect가 identity effect보다 훨씬 크고, residual 이후 identity effect는 거의 0에 가깝다.

## fig_shared_encoder_information_bottleneck.png
- Korean caption: Shared encoder의 정보 병목과 identity/residual branch가 보완하는 정보의 차이.
- English caption: Information bottleneck view of the shared encoder and the different cues restored by identity embeddings and the residual branch.
- 핵심 메시지: identity는 token origin을 알려주지만 residual branch는 signal-derived statistics를 직접 제공한다.

## diagram_architecture_comparison_four_models.png
- Korean caption: All-Channel CNN, Shared 1D, Shared 1D + Identity, Residual Channel-Shared 구조의 정보 흐름 비교.
- English caption: Information-flow comparison of All-Channel CNN, Shared 1D, Shared 1D + Identity, and Residual Channel-Shared architectures.

## diagram_residual_branch_detailed.png
- Korean caption: train-scaled tensor에서 mean/std/min/max 72개 channel-wise statistics를 계산하는 residual branch.
- English caption: Residual branch that computes 72 channel-wise statistics from the train-scaled tensor.

## diagram_identity_embedding_detailed.png
- Korean caption: channel/sensor/modality/axis embedding을 token에 더해 token origin을 제공하는 identity embedding 구조.
- English caption: Identity embedding structure that adds channel, sensor, modality, and axis embeddings to each token.
""",
        encoding="utf-8",
    )
    return path


def write_main_report(paths: ProfessorReportV3Paths) -> Path:
    table = lambda name: markdown_table(read_csv(paths.tables / name))
    report = f"""---
title: "Professor Report v3 - Story and Identity Analysis"
project: "Squat IMU"
date: "{date.today().isoformat()}"
status: "story refinement"
tags:
  - research
  - squat
  - imu
  - kiee
---

# Professor Report v3 - Story and Identity Analysis

## 0. 이번 v3 보강의 목적

v2는 architecture/protocol table을 보강했지만, **왜 identity가 shared-only에서는 도움이 되고 residual 이후에는 거의 도움이 되지 않았는지**에 대한 story가 부족했다. v3는 기존 결과만 사용해 identity/residual interaction을 정량화하고, 정보 흐름 중심 architecture diagram으로 논문 claim을 정리한다. 새 학습, CAU training, optimizer step, backpropagation, split/preprocessing/model 변경은 수행하지 않았다.

## 1. 한 줄 결론

Identity는 residual이 없을 때 shared token의 위치 손실을 일부 보완했다. 그러나 residual branch는 train-scaled tensor에서 channel-wise mean/std/min/max를 직접 보존하므로 훨씬 큰 성능 개선을 만들었다. Residual이 들어간 뒤에는 identity의 추가 이득이 작았다. 따라서 국내 논문 claim은 position identity보다 **residual branch가 shared encoder bottleneck을 완화한다**는 방향이 안전하다.

## 2. 왜 common classifier head가 필요했는가

기존 full matrix는 모델마다 head와 pathway가 달라, 성능 차이가 feature extractor 때문인지 classifier 때문인지 분리하기 어려웠다. 교수님 피드백 이후 controlled comparison에서는 모든 controlled neural model이 64차원 representation과 동일한 MLP classifier head를 사용하도록 맞췄다.

![Common Classifier Head](diagrams/diagram_common_head_detailed_v3.png)

## 3. Architecture question: 무엇을 비교한 것인가

질문은 단순히 어떤 model이 높았는지가 아니라, **shared encoder가 parameter sharing을 얻는 대신 잃어버릴 수 있는 channel-specific cue를 어떤 방식으로 복원하는가**다.

![Architecture Comparison](diagrams/diagram_architecture_comparison_four_models.png)

## 4. Information bottleneck view

All-Channel CNN은 18채널을 처음부터 함께 보므로 channel identity를 자연스럽게 유지한다. 반면 Shared 1D Encoder는 18개 single-channel stream에 같은 encoder를 반복 적용하고 token pooling을 수행하므로, token origin과 channel-wise summary cue가 약해질 수 있다. Identity embedding은 token origin을 알려주지만, raw signal statistics 자체를 직접 제공하지는 않는다. Residual branch는 fixed channel order의 mean/std/min/max 72개 feature를 제공해 이 병목을 직접 보완한다.

![Information Bottleneck](figures/fig_shared_encoder_information_bottleneck.png)

## 5. Identity effect: residual이 없을 때는 도움이 됐다

Shared 1D Encoder의 Macro F1은 0.2806이고, Shared 1D + Identity는 0.5182다. seed-fold paired 기준 identity effect without residual은 +0.2376이었다. 이는 channel/sensor/modality/axis embedding이 shared token의 origin 손실을 일부 보완했음을 시사한다.

![Identity Embedding Detail](diagrams/diagram_identity_embedding_detailed.png)

## 6. Residual effect: 더 큰 변화는 residual branch에서 나왔다

Residual Channel-Shared Encoder의 Macro F1은 0.8004다. Shared 1D 대비 residual effect without identity는 +0.5198로, identity effect보다 훨씬 컸다. residual branch는 train-scaled tensor에서 channel-wise mean/std/min/max 72 features를 계산하고, metadata, label, subject id, filename, boundary, original length는 사용하지 않는다.

![Residual Branch Detail](diagrams/diagram_residual_branch_detailed.png)

## 7. 왜 residual 이후 identity가 거의 도움이 되지 않았는가

2x2 결과는 interaction을 명확히 보여준다. residual이 없을 때 identity effect는 +0.2376이지만, residual이 있을 때 identity effect는 -0.0031이다. 한 가지 가능한 해석은 residual branch가 fixed channel order의 statistics를 이미 제공하므로, identity embedding이 제공하는 token origin 정보와 일부 중복된다는 것이다. 다만 추가 parameter, 작은 데이터셋 variance, seed/fold별 변동 가능성도 있으므로 원인을 단정하지 않는다.

{table("table_identity_residual_2x2.csv")}

![Identity Residual 2x2 Grid](figures/fig_identity_residual_2x2_grid.png)

{table("table_identity_residual_effects.csv")}

![Identity Residual Effect Arrows](figures/fig_identity_residual_effect_arrows.png)

## 8. Practical baselines와의 관계

Statistical Summary MLP는 Macro F1 0.8174, XGBoost는 0.7961, Random Forest는 0.7845였다. 이 결과는 이 데이터셋에서 channel-wise summary statistics가 강한 signal임을 보여준다. Residual Channel-Shared Encoder는 raw temporal branch와 summary statistics branch를 결합한 hybrid extractor로 해석할 수 있다.

## 9. 논문 claim 조정

{table("table_claim_adjustment_v3.csv")}

정리하면 safe claim은 다음과 같다. residual branch가 shared encoder bottleneck을 완화했다. residual shared extractor는 All-Channel CNN, XGBoost, RF와 경쟁 가능하다. summary statistics가 강한 signal이다. 반대로 identity가 핵심이다, attention이 핵심이다, 모든 baseline보다 통계적으로 우수하다, transfer learning을 검증했다는 표현은 피해야 한다.

## 10. 교수님께 설명할 3분 버전

교수님, 이번에는 모델 구조별 head를 동일하게 고정한 상태에서 feature extractor만 비교했습니다. 결과를 보면 Shared 1D Encoder 단독은 Macro F1 0.2806으로 낮았고, identity를 넣으면 0.5182로 올라갔습니다. 즉, shared encoder가 channel origin 정보를 잃는 문제가 있고 identity embedding이 이를 일부 보완한 것으로 볼 수 있습니다.

그런데 residual branch를 넣으면 0.8004까지 올라갑니다. 이 residual branch는 metadata나 label을 쓰는 것이 아니라, train-scaled IMU tensor에서 각 channel의 mean, std, min, max만 계산한 72개 signal feature입니다. 즉 shared temporal branch가 놓칠 수 있는 channel-wise statistical cue를 직접 보존합니다.

흥미로운 점은 residual branch가 들어간 뒤 identity를 추가하면 0.7973으로 거의 차이가 없다는 것입니다. 그래서 이번 국내 논문에서는 position identity를 핵심 novelty로 밀기보다, residual branch가 shared encoder bottleneck을 완화한다는 claim이 더 안전해 보입니다. Statistical Summary MLP와 tree baseline도 강하기 때문에, 압도적 우월성보다는 구조적 bottleneck 완화와 practical baseline과의 경쟁 가능성으로 정리하는 편이 좋겠습니다.

## 11. 다음 보고 때 확인받을 질문

1. 국내 논문 제목을 `Residual Channel-Shared Feature Extractor` 중심으로 정리해도 되는가?
2. identity/position encoding은 future work 또는 appendix로 낮춰도 되는가?
3. Statistical Summary MLP가 1위인 점을 practical baseline으로 분리해서 설명할지?
4. residual branch 중심의 novelty가 국내 저널 범위에서 충분한가?
5. RF/XGBoost는 main table에 넣을지, practical baseline table로 분리할지?

## Appendix A. 수치 표

{table("table_story_numbers.csv")}

{table("table_identity_residual_paired_details.csv")}

## Appendix B. 내부 이름 mapping

Internal model names are only allowed in the mapping file: `tables/table_internal_name_mapping.csv`.
"""
    path = paths.output_docs / "Professor Report v3 - Story and Identity Analysis.md"
    path.write_text(report, encoding="utf-8")
    return path


def write_figure_table_index(paths: ProfessorReportV3Paths) -> Path:
    rows = [
        {"artifact": "table_identity_residual_2x2.csv", "source": "aggregate_metrics_by_model.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "7", "paper_candidate": "yes"},
        {"artifact": "table_identity_residual_effects.csv", "source": "fold_metrics.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "7", "paper_candidate": "yes"},
        {"artifact": "fig_identity_residual_2x2_grid.png", "source": "table_identity_residual_2x2.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "7", "paper_candidate": "yes"},
        {"artifact": "fig_identity_residual_effect_arrows.png", "source": "table_identity_residual_effects.csv", "script": "scripts/build_professor_report_assets.py", "report_section": "7", "paper_candidate": "maybe"},
        {"artifact": "fig_shared_encoder_information_bottleneck.png", "source": "controlled architecture and aggregate metrics", "script": "scripts/build_professor_report_assets.py", "report_section": "4", "paper_candidate": "maybe"},
        {"artifact": "diagram_architecture_comparison_four_models.png", "source": "controlled extractor code", "script": "scripts/build_professor_report_assets.py", "report_section": "3", "paper_candidate": "yes"},
        {"artifact": "diagram_residual_branch_detailed.png", "source": "controlled extractor code", "script": "scripts/build_professor_report_assets.py", "report_section": "6", "paper_candidate": "yes"},
        {"artifact": "diagram_identity_embedding_detailed.png", "source": "controlled extractor code", "script": "scripts/build_professor_report_assets.py", "report_section": "5", "paper_candidate": "appendix"},
    ]
    path = paths.output_docs / "Figure and Table Index v3.md"
    path.write_text("# Figure and Table Index v3\n\n" + markdown_table(rows) + "\n", encoding="utf-8")
    return path


def write_v2_revision_suggestions(paths: ProfessorReportV3Paths) -> Path:
    path = paths.output_docs / "v2_revision_suggestions.md"
    path.write_text(
        """# v2 Revision Suggestions

## Position identity 표현 조정

- v2에서 position identity가 핵심 contribution처럼 읽히는 문장은 `identity는 shared-only 구조에서 일부 보완 효과가 있었지만, 최종 claim은 residual branch 중심`으로 낮추는 것이 좋다.
- `identity가 성능 개선의 핵심`이라는 표현은 피하고, `residual branch가 들어간 뒤 identity 추가 이득은 제한적`이라는 2x2 결과를 함께 제시한다.

## Residual branch 용어

- `raw summary residual branch`라는 표현이 있으면 `train-scaled channel-wise statistical branch`로 고치는 것이 더 정확하다.
- residual/statistical MLP branch는 train-only StandardScaler transform 이후 model 내부에서 mean/std/min/max를 계산한다.

## Feature set 구분

- Statistical Summary MLP와 residual branch는 mean/std/min/max, 18 channels x 4 = 72 features다.
- RF/XGBoost/SVM feature set은 162 signal-derived features이며, energy/RMS/median/peak-to-peak/dominant frequency 계열이 포함된다.
- Statistical Summary MLP 설명에 energy/RMS가 들어간 것처럼 보이는 문구는 제거한다.

## Attention 표현 조정

- 이번 교수님 보고와 국내 논문 claim에서는 attention을 핵심으로 두지 않는다.
- attention이나 position identity는 appendix 또는 future work로 낮추고, 본문은 residual branch의 shared encoder bottleneck 완화에 집중한다.
""",
        encoding="utf-8",
    )
    return path


def write_readme(paths: ProfessorReportV3Paths) -> Path:
    path = paths.output_docs / "README.md"
    path.write_text(
        """# Professor Report v3

이 폴더는 교수님 보고용 story refinement 및 identity/residual interaction 분석 산출물이다.

- 새 학습 없이 기존 controlled comparison 결과만 read-only로 사용했다.
- 핵심 산출물은 2x2 identity/residual table, paired effect table, information-flow diagrams다.
- 교수님 보고용 본문과 figure/table title에는 internal experiment names를 노출하지 않는다.
""",
        encoding="utf-8",
    )
    return path


def validate_report_v3(paths: ProfessorReportV3Paths) -> dict[str, Any]:
    md_files = list(paths.output_docs.rglob("*.md"))
    png_files = list(paths.output_docs.rglob("*.png"))
    csv_files = list(paths.output_docs.rglob("*.csv"))
    allowed = {
        "display_name_mapping.yaml",
        "table_internal_name_mapping.csv",
    }
    violations = []
    for file in [*md_files, *csv_files, *list(paths.output_docs.rglob("*.yaml"))]:
        if file.name in allowed:
            continue
        if "controlled_" in file.read_text(encoding="utf-8", errors="ignore"):
            violations.append(str(file.relative_to(paths.output_docs)))
    table_errors = markdown_table_pipe_errors(md_files)
    image_errors = missing_image_links(paths.output_docs)
    return {
        "markdown_files": len(md_files),
        "png_files": len(png_files),
        "csv_files": len(csv_files),
        "controlled_name_violations": violations,
        "controlled_name_violation_count": len(violations),
        "table_pipe_errors": table_errors,
        "missing_image_links": image_errors,
        "passed": not violations and not table_errors and not image_errors,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def bootstrap_ci(values: list[float], *, n: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    low_idx = int(0.025 * (n - 1))
    high_idx = int(0.975 * (n - 1))
    return means[low_idx], means[high_idx]
