# XGBoost Only Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run only the previously skipped `feature_xgboost_v1` controlled baseline in a separate completion result directory without modifying locked or controlled comparison results.

**Architecture:** Add a dedicated config, runner, analysis module, and CLI scripts. Reuse the existing final LOSO split implementation, train-only scaler, classical feature extractor, and metric aggregation. Treat existing controlled comparison outputs as read-only reference inputs for merged summary and paired comparison.

**Tech Stack:** Python, NumPy, scikit-learn-compatible XGBoost API, matplotlib, unittest, existing clean-room `squat_imu_experiments` modules.

---

### Task 1: Safety Tests and Config

**Files:**
- Create: `configs/xgboost_only_completion_v1.yaml`
- Create: `tests/test_xgboost_completion_config_safety.py`
- Create: `tests/test_xgboost_feature_audit.py`
- Create: `tests/test_xgboost_dependency_handling.py`
- Create: `tests/test_xgboost_completion_dry_run.py`

- [ ] **Step 1: Write failing tests**

Tests must assert that only `feature_xgboost_v1` is configured, forbidden features are disabled, feature audit allows only signal-derived features, missing XGBoost produces `dependency_missing`, and dry-run writes an 18-run plan without training.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m unittest tests.test_xgboost_completion_config_safety tests.test_xgboost_feature_audit tests.test_xgboost_dependency_handling tests.test_xgboost_completion_dry_run -v
```

Expected: failures because runner/config modules do not exist yet.

### Task 2: Runner and Analysis

**Files:**
- Create: `src/training/xgboost_completion_runner.py`
- Create: `src/analysis/xgboost_completion_analysis.py`
- Modify: `src/models/classical_features.py`

- [ ] **Step 1: Implement config loader and safety validation**

Validate seeds, model list, split policy, feature policy, normalization, statistics, and safety flags.

- [ ] **Step 2: Implement dry-run**

Write preflight files, feature audit, split plan, validation policy summary, scaler audit, and a 3 seeds x 6 folds run plan. Do not import or fit XGBoost in dry-run.

- [ ] **Step 3: Implement confirmed run**

Require clean git, explicit confirmation, and available XGBoost dependency. Fit only `feature_xgboost_v1` on train indices, transform validation/test using train-only scaler, and save fold metrics, classwise metrics, confusion matrices, predictions, feature importance, aggregate metrics, bootstrap CI, and failed runs.

- [ ] **Step 4: Implement read-only merge**

Read `results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1/` without writing to it, then create `merged_with_controlled_comparison/` summaries and figures inside the new XGBoost output directory.

### Task 3: CLI Scripts, Docs, and Verification

**Files:**
- Create: `scripts/run_xgboost_only_completion.py`
- Create: `scripts/summarize_xgboost_only_completion.py`
- Create/update docs after CAU result

- [ ] **Step 1: Add CLI scripts**

Support `--dry-run`, `--confirm-xgboost-completion`, and optional `--resume`.

- [ ] **Step 2: Run local verification**

Run:

```bash
python -m unittest discover -s tests -v
python scripts/run_xgboost_only_completion.py --config configs/xgboost_only_completion_v1.yaml --dry-run
```

- [ ] **Step 3: Commit before CAU**

Commit code/config/tests, excluding `.deps/`, `results/`, and processed data.

- [ ] **Step 4: Run on CAU**

Check dependency, install into `.deps/xgboost` if needed, run tests, then execute:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.deps/xgboost:$PYTHONPATH python scripts/run_xgboost_only_completion.py \
  --config configs/xgboost_only_completion_v1.yaml \
  --confirm-xgboost-completion
```

- [ ] **Step 5: Retrieve results and document**

Copy only the new `results/xgboost_only_completion/<timestamp>_xgboost_only_completion_v1/` directory to local. Write Korean report and update tracking/protocol/paper draft documents.
