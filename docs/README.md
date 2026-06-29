# Documentation Index

이 폴더는 clean-room squat IMU classification 프로젝트의 문서와 GitHub용 선별 결과 산출물을 모은다.

## 빠른 시작

- 실험 프로토콜: `experiment_protocol.md`
- 전체 결과 추적: `results_tracking.md`
- GitHub용 선별 결과 artifact: `results_artifacts/README.md`
- 논문 결과 초안: `paper_results_draft_v2.md`
- 논문 claim audit: `paper_claim_audit_v2.md`
- 논문 discussion 초안: `paper_discussion_draft_v2.md`
- 교수님 보고서 mirror v1: `professor_report_v1/Professor Report - Residual Channel-Shared Encoder.md`
- 교수님 보고서 mirror v2: `professor_report_v2/Professor Report v2 - Architecture and Protocol.md`

## 주요 주제별 문서

### Data

- `data_inventory.md`
- `target_dataset_decision.md`
- `target_conversion_v1_report.md`

### Models

- `model_architecture_diagrams.md`
- `model_capacity_correction_v1.md`
- `model_capacity_v3_report.md`
- `netron_model_visualization.md`

### Core Experiments

- `full_supervised_matrix_v1_report.md`
- `literature_baseline_full_extension_v1_report.md`
- `controlled_feature_extractor_comparison_v1_report.md`
- `v3_component_ablation_v1_report.md`
- `xgboost_only_completion_v1_report.md`

### Interpretation

- `controlled_feature_extractor_interpretation_notes_for_user.md`
- `literature_baseline_interpretation_notes_for_user.md`
- `v3_component_ablation_interpretation_notes_for_user.md`
- `paper_claim_audit_v2.md`
- `professor_report_v1/Figure and Table Index.md`

### Manuscript Drafts

- `paper_method_draft_v2.md`
- `paper_results_draft_v2.md`
- `paper_discussion_draft_v2.md`
- `paper_outline_kiee.md`

### Professor Report

- `professor_report_v1/Professor Report - Residual Channel-Shared Encoder.md`
- `professor_report_v1/Figure and Table Index.md`
- `professor_report_v1/figures/captions.md`
- `professor_report_v2/Professor Report v2 - Architecture and Protocol.md`
- `professor_report_v2/Figure and Table Index v2.md`
- `professor_report_v2/figures/captions.md`

`professor_report_v2/`는 architecture audit, normalization/scaler 순서, parameter count, confusion matrix, residual feature audit을 보강한 교수님 보고용 mirror다.

## 결과 산출물 관리 원칙

- 원본 `results/` 전체는 git에 commit하지 않는다.
- GitHub에서 검토해야 하는 핵심 CSV/MD/PNG는 `docs/results_artifacts/`에 선별 복사한다.
- raw processed data, checkpoint, dependency cache, full generated results는 제외한다.
- 기존 result directory는 read-only reference로 취급한다.
