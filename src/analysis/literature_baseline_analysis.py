from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from training.supervised_trainer import write_csv


LOCKED_RUN_DIR = Path("results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1")
LOCKED_MODELS = [
    "channel_shared_posres_attention_v3",
    "all_channel_conv1d_small",
    "all_channel_conv1d_v1",
    "modality_shared_sensorattn_v3",
    "cnn2d_baseline_v1",
    "channel_shared_attentionpool_v2",
    "channel_shared_meanpool_v2",
]


def merge_screening_with_locked_matrix(
    *,
    project_root: Path,
    screening_run_dir: Path,
    locked_run_dir: Path | None = None,
) -> dict[str, Any]:
    locked_dir = (project_root / (locked_run_dir or LOCKED_RUN_DIR)).resolve()
    output_dir = screening_run_dir / "merged_with_locked_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    locked_rows = _read_csv(locked_dir / "aggregate_metrics_by_model.csv")
    screening_rows = _read_csv(screening_run_dir / "aggregate_metrics_by_model.csv")
    locked_filtered = [row for row in locked_rows if row.get("model_name") in LOCKED_MODELS]

    merged = [_locked_row(row) for row in locked_filtered] + [_screening_row(row) for row in screening_rows]
    ranking = sorted(merged, key=lambda row: float(row.get("mean_macro_f1") or 0.0), reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank_by_macro_f1_mixed_reference_only"] = rank
    param_rows = [
        {
            "model_name": row["model_name"],
            "result_type": row["result_type"],
            "total_params": row.get("total_params", ""),
            "mean_macro_f1": row.get("mean_macro_f1", ""),
        }
        for row in ranking
    ]

    write_csv(output_dir / "merged_model_comparison_screening.csv", merged)
    write_csv(output_dir / "merged_macro_f1_ranking_screening.csv", ranking)
    write_csv(output_dir / "merged_parameter_vs_macro_f1_screening.csv", param_rows)
    summary = _screening_summary_markdown(ranking)
    (output_dir / "literature_baseline_screening_summary.md").write_text(summary, encoding="utf-8")
    return {"output_dir": str(output_dir), "n_locked_models": len(locked_filtered), "n_screening_models": len(screening_rows)}


def write_literature_screening_report(run_dir: Path, docs_path: Path) -> None:
    aggregate = _read_csv(run_dir / "aggregate_metrics_by_model.csv")
    params = {row["model_name"]: row for row in _read_csv(run_dir / "model_parameter_counts.csv")}
    skipped = _read_csv(run_dir / "skipped_runs.csv")
    failed = _read_csv(run_dir / "failed_runs.csv")
    lines = [
        "# Literature Temporal Baseline Screening v1 Report",
        "",
        "## 실행 범위",
        "",
        "- 이 결과는 1 seed 문헌 baseline screening이다.",
        "- 기존 locked full supervised matrix 결과는 수정하지 않았다.",
        "- 3 seed full extension 또는 hyperparameter tuning은 수행하지 않았다.",
        "- 1 seed screening과 3 seed locked result를 직접 동등 비교하면 안 된다.",
        "",
        "## 모델별 1 Seed Screening 결과",
        "",
        "| model | params | accuracy | macro F1 | weighted F1 | success runs | failed runs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate, key=lambda item: float(item.get("mean_macro_f1") or 0.0), reverse=True):
        p = params.get(row["model_name"], {})
        lines.append(
            "| {model} | {params} | {acc:.4f} | {macro:.4f} | {weighted:.4f} | {succ} | {fail} |".format(
                model=row["model_name"],
                params=p.get("total_params", ""),
                acc=float(row.get("mean_accuracy") or 0.0),
                macro=float(row.get("mean_macro_f1") or 0.0),
                weighted=float(row.get("mean_weighted_f1") or 0.0),
                succ=row.get("n_success_runs", ""),
                fail=row.get("n_failed_runs", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Skipped/Failed",
            "",
            f"- skipped rows: {len(skipped)}",
            f"- failed rows: {len(failed)}",
            "",
            "## 해석 주의",
            "",
            "screening 결과는 후보 선별용 수치이며 논문 main claim에 직접 사용하지 않는다. literature baseline 대비 우월성 주장은 3 seed full extension 승인 및 실행 후에만 검토한다.",
        ]
    )
    docs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _locked_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "model_name": row["model_name"],
        "result_type": "locked_3seed_full",
        "total_params": row.get("total_params", ""),
        "mean_accuracy": row.get("mean_accuracy", ""),
        "std_accuracy": row.get("std_accuracy", ""),
        "mean_macro_f1": row.get("mean_macro_f1", ""),
        "std_macro_f1": row.get("std_macro_f1", ""),
        "mean_weighted_f1": row.get("mean_weighted_f1", ""),
        "warning": "3 seed locked result; do not directly equate with 1 seed screening",
    }


def _screening_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "model_name": row["model_name"],
        "result_type": "literature_1seed_screening",
        "total_params": row.get("total_params", ""),
        "mean_accuracy": row.get("mean_accuracy", ""),
        "std_accuracy": row.get("std_accuracy", ""),
        "mean_macro_f1": row.get("mean_macro_f1", ""),
        "std_macro_f1": row.get("std_macro_f1", ""),
        "mean_weighted_f1": row.get("mean_weighted_f1", ""),
        "warning": "1 seed screening only; not a final paper result",
    }


def _screening_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Literature Baseline Screening Summary",
        "",
        "> Warning: locked matrix rows are 3 seed full results, while literature rows are 1 seed screening results. The mixed ranking is for triage only.",
        "",
        "| rank | model | result type | mean macro F1 | mean accuracy |",
        "|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank_by_macro_f1_mixed_reference_only']} | {row['model_name']} | {row['result_type']} | "
            f"{float(row.get('mean_macro_f1') or 0.0):.4f} | {float(row.get('mean_accuracy') or 0.0):.4f} |"
        )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
