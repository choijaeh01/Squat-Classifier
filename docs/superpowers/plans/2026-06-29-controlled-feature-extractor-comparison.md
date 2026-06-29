# Controlled Feature Extractor Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 공통 classifier head를 고정한 controlled feature extractor 비교 실험을 clean-room codebase에 추가하고 CAU에서 제한 없이 명시된 3 seed full comparison을 실행한다.

**Architecture:** Controlled neural model은 `FeatureExtractor -> CommonMLPClassifierHead` 형태로 분리한다. 기존 final LOSO split, train-only scaler, supervised trainer, classical trainer, full-matrix aggregation utilities를 재사용하고 새 결과는 `results/controlled_feature_extractor_comparison/`에만 저장한다.

**Tech Stack:** Python, PyTorch, NumPy, scikit-learn optional, xgboost optional, matplotlib, unittest.

---

### Task 1: Tests and Config

**Files:**
- Create: `configs/controlled_feature_extractor_comparison_v1.yaml`
- Create: `configs/controlled_feature_extractor_capacity_v1.yaml`
- Create: `tests/test_controlled_common_head.py`
- Create: `tests/test_controlled_feature_extractors.py`
- Create: `tests/test_controlled_model_forward.py`
- Create: `tests/test_controlled_comparison_config_safety.py`
- Create: `tests/test_controlled_comparison_dry_run.py`
- Create: `tests/test_controlled_feature_audit.py`
- Create: `tests/test_xgboost_availability_handling.py`

- [ ] Write tests before implementation.
- [ ] Run targeted controlled tests and verify they fail because modules or registry names are missing.

### Task 2: Controlled Models

**Files:**
- Create: `src/models/common_head.py`
- Create: `src/models/controlled_extractors.py`
- Create: `src/models/controlled_models.py`
- Modify: `src/models/registry.py`
- Modify: `src/models/classical_features.py`
- Modify: `src/analysis/feature_importance_analysis.py`

- [ ] Implement `CommonMLPClassifierHead`.
- [ ] Implement controlled extractors with representation_dim 64.
- [ ] Register all controlled models and `feature_xgboost_v1`.
- [ ] Add XGBoost availability and estimator handling without installing packages.

### Task 3: Capacity Audit and Runner

**Files:**
- Create: `scripts/audit_controlled_feature_extractor_capacity.py`
- Create: `scripts/run_controlled_feature_extractor_comparison.py`
- Create: `scripts/summarize_controlled_feature_extractor_comparison.py`
- Create: `src/training/controlled_comparison_runner.py`
- Create: `src/analysis/controlled_comparison_analysis.py`

- [ ] Implement safety validation and expected run plan count 252.
- [ ] Reuse final LOSO split and train-only scaler logic.
- [ ] Save required CSVs, analysis tables, figures, and skipped/failed runs.

### Task 4: Verification and CAU Execution

**Files:**
- Create/update docs under `docs/`.

- [ ] Run local unittest, capacity audit, and dry-run only.
- [ ] Commit code/docs before CAU training.
- [ ] Sync project and processed dataset to `/home/user3/workspace/Jae/squat_imu_experiments`.
- [ ] Run CAU tests, capacity audit, and confirmed controlled comparison.
- [ ] Retrieve results and write Korean reports.
