# XGBoost Only Completion v1 Report

## 실행 목적

`xgboost_only_completion_v1`은 `controlled_feature_extractor_comparison_v1`에서 dependency 부재로 skipped 처리된 `feature_xgboost_v1`만 별도로 완료한 실행이다. 기존 controlled comparison 결과 디렉터리는 수정하지 않았고, 동일 split, scaler, feature audit 정책을 사용해 새 output directory에만 결과를 저장했다.

이 실행은 hyperparameter tuning이 아니다. 결과를 보고 feature set, split, preprocessing, scaler policy, XGBoost parameter를 변경하지 않았다.

## 실행 개요

- 실행 위치: CAU 서버
- CAU hostname: `e8acec3a6d36`
- Python: `3.10.12`
- Torch: `2.8.0+cu129`
- sklearn: `1.7.2`
- XGBoost: `3.2.0`
- 실행 commit: `68905645b50e7178669e74e3a4e70e8b3d523da3`
- config: `configs/xgboost_only_completion_v1.yaml`
- 결과 경로: `results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1/`
- reference controlled result: `results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1/`

## XGBoost 설치 방식

CAU 기본 Python에서는 `xgboost`가 설치되어 있지 않았다. 시스템 Python site-packages, `sudo`, `pip install --user`는 사용하지 않았다.

Project-local dependency path를 사용했다.

```bash
mkdir -p .deps/xgboost
python -m pip install --target .deps/xgboost xgboost
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.deps/xgboost:$PYTHONPATH python scripts/run_xgboost_only_completion.py \
  --config configs/xgboost_only_completion_v1.yaml \
  --confirm-xgboost-completion
```

설치 후 `PYTHONPATH=.deps/xgboost:$PYTHONPATH`를 붙였을 때만 `xgboost 3.2.0`이 import되도록 확인했다. `.deps/`는 `.gitignore`에 포함되어 git 추적 대상이 아니다.

## 실행 범위

- model: `feature_xgboost_v1`
- seeds: `42`, `123`, `2025`
- folds: 6 LOSO folds
- expected runs: 18
- successful runs: 18
- failed runs: 0
- skipped runs: 0

Split policy는 final protocol과 동일하다.

- `loso_with_within_train_subject_stratified_validation`
- fold별 train: 400 windows
- fold별 validation: 100 windows
- fold별 test: 100 windows
- scaler fit: train indices only
- per-window z-score: false
- augmentation, focal loss, balanced sampling, SSL, external dataset: disabled

## 누수 및 Feature Audit

- test subject isolation: 통과
- train/validation index disjoint: 통과
- scaler leakage check: 통과
- feature audit: 162개 feature 모두 `allowed=True`
- metadata feature 사용: 없음
- subject ID feature 사용: 없음
- class label feature 사용: 없음
- window boundary/original length feature 사용: 없음

`feature_xgboost_v1`은 RandomForest baseline과 같은 IMU signal-derived hand-crafted feature set을 사용한다. feature에는 filename, subject, class, set, window index, original length, boundary 정보가 포함되지 않는다.

## XGBoost 결과

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| feature_xgboost_v1 | 0.8156 | 0.7961 | 0.7961 | [0.7567, 0.8328] |

Seed/fold 전체 18 run 평균이다. CI는 기존 pipeline의 bootstrap 요약 방식으로 계산했다.

## Class 3 Excessive Lean

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Class 3 Excessive Lean | 0.8288 | 0.6444 | 0.6614 | 360 |

Class 3에서는 precision은 높지만 recall은 0.6444로, 일부 Excessive Lean sample을 다른 class로 놓치는 경향이 남아 있다.

## Top XGBoost Features

| rank | feature | channel | statistic | mean importance |
|---:|---|---|---|---:|
| 1 | `s1_ax_std` | `s1_ax` | std | 0.0674 |
| 2 | `s1_gx_mean` | `s1_gx` | mean | 0.0656 |
| 3 | `s1_ax_energy` | `s1_ax` | energy | 0.0601 |
| 4 | `s0_ax_mean` | `s0_ax` | mean | 0.0321 |
| 5 | `s0_az_std` | `s0_az` | std | 0.0303 |
| 6 | `s0_ax_median` | `s0_ax` | median | 0.0225 |
| 7 | `s2_gy_max` | `s2_gy` | max | 0.0197 |
| 8 | `s2_az_median` | `s2_az` | median | 0.0195 |
| 9 | `s0_ay_median` | `s0_ay` | median | 0.0182 |
| 10 | `s1_az_mean` | `s1_az` | mean | 0.0170 |

RandomForest와 마찬가지로 오른쪽 허벅지 `s1` 관련 feature, 특히 `s1_ax`와 `s1_gx`가 상위에 나타났다. Feature importance는 model-specific importance이며 생체역학적 인과 설명으로 단정하지 않는다.

## Controlled Comparison과 병합한 참고 순위

| rank | model | result type | macro F1 |
|---:|---|---|---:|
| 1 | controlled_stats_mlp | controlled_feature_extractor_readonly | 0.8174 |
| 2 | controlled_shared_1d_residual | controlled_feature_extractor_readonly | 0.8004 |
| 3 | controlled_all_channel_1d_cnn | controlled_feature_extractor_readonly | 0.7994 |
| 4 | controlled_shared_1d_residual_identity | controlled_feature_extractor_readonly | 0.7973 |
| 5 | feature_xgboost_v1 | xgboost_only_completion_3seed_full | 0.7961 |
| 6 | feature_random_forest_v1 | controlled_feature_extractor_readonly | 0.7845 |
| 7 | rescnn_bigru_attention_lite_v1 | controlled_feature_extractor_readonly | 0.7691 |
| 8 | controlled_all_channel_1d_cnn_small | controlled_feature_extractor_readonly | 0.7562 |
| 9 | lee_style_cnn_lstm_2d_v1 | controlled_feature_extractor_readonly | 0.6269 |

기존 controlled result는 read-only로만 읽었고 수정하지 않았다.

## Paired Difference

부호는 `feature_xgboost_v1 - reference_model`이다.

| reference | delta macro F1 | CI | n pairs |
|---|---:|---|---:|
| feature_random_forest_v1 | +0.0116 | [-0.0145, 0.0398] | 18 |
| controlled_stats_mlp | -0.0212 | [-0.0594, 0.0171] | 18 |
| controlled_shared_1d_residual | -0.0043 | [-0.0385, 0.0304] | 18 |
| controlled_shared_1d_residual_identity | -0.0012 | [-0.0328, 0.0310] | 18 |
| controlled_all_channel_1d_cnn | -0.0033 | [-0.0571, 0.0533] | 18 |

모든 listed comparison에서 CI가 0을 포함한다. 따라서 XGBoost가 RF, controlled stats MLP, residual shared, all-channel CNN보다 통계적으로 우월하거나 열등하다고 단정하지 않는다.

## 산출물

- `aggregate_metrics_by_model.csv`
- `bootstrap_confidence_intervals.csv`
- `paired_model_differences.csv`
- `feature_definitions.csv`
- `feature_audit.csv`
- `xgboost_feature_importance_aggregate.csv`
- `xgboost_feature_importance_by_fold.csv`
- `merged_with_controlled_comparison/*.csv`
- `figures/xgboost_macro_f1_vs_baselines.png`
- `figures/xgboost_feature_importance_top20.png`
- `figures/xgboost_rf_feature_importance_comparison.png`
- `figures/xgboost_class3_f1_recall.png`
- `figures/xgboost_subjectwise_macro_f1.png`

## 해석 제한

- XGBoost는 separate completion으로 실행했지만 split, scaler, feature policy는 controlled comparison과 동일하다.
- 결과를 보고 XGBoost hyperparameter를 조정하지 않았다.
- feature importance는 classifier 내부 중요도이며 인과적 설명이 아니다.
- 논문 결과 표에 XGBoost를 포함할지, controlled comparison table에 보조 baseline으로 둘지는 사용자가 최종 판단해야 한다.
