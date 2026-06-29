# controlled_feature_extractor_comparison_v1 Report

## 실행 개요

- 목적: 같은 64차원 representation과 같은 MLP classifier head를 고정한 상태에서 feature extractor 자체의 기여를 비교한다.
- 실행 위치: CAU 서버
- CAU hostname: `e8acec3a6d36`
- Python: `3.10.12`
- Torch: `2.8.0+cu129`
- sklearn: `1.7.2`
- device: `cuda`, `NVIDIA RTX 6000 Ada Generation`
- 실행 commit: `e6128a5cc1f0b62cc300ac78e4adc102cd57b4e2`
- config: `configs/controlled_feature_extractor_comparison_v1.yaml`
- 결과 경로: `results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1/`

## 실행 범위

- seeds: `42`, `123`, `2025`
- folds: 6 LOSO folds
- configured runs: 252
- successful runs: 234
- failed runs: 0
- skipped runs: 18
- skipped reason: `feature_xgboost_v1`의 `xgboost is not installed`

XGBoost는 요구사항대로 의존성이 없어서 해당 baseline만 skipped 처리했다. scikit-learn 기반 RandomForest와 LinearSVM은 정상 실행되었다.

후속 단계에서 `feature_xgboost_v1`만 `xgboost_only_completion_v1`로 별도 실행했다. 기존 controlled result directory는 수정하지 않았고, completion 결과는 `results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1/`에 저장했다.

## Split 및 누수 검증

- validation policy: `loso_with_within_train_subject_stratified_validation`
- 각 fold: train 400, validation 100, test 100
- test subject isolation: 모두 통과
- train/validation index disjoint: 모두 통과
- scaler fit scope: train indices only
- scaler leakage check: 모두 통과
- augmentation, focal loss, balanced sampling, SSL, external dataset: 모두 비활성화

## 공통 Head 검증

controlled neural 모델 9개는 모두 다음 classifier head를 공유했다.

- representation dim: 64
- common head parameter count: 4,485
- common head structure: `linear:64->64`, `relu`, `dropout:0.1`, `linear:64->5`

따라서 controlled neural 모델 간 차이는 classifier head가 아니라 extractor, identity/residual branch, pooling 또는 feature 요약 방식의 차이로 해석해야 한다.

## Model Comparison

| model | params | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---:|---|
| controlled_stats_mlp | 9,157 | 0.8250 | 0.8174 | 0.8174 | [0.7744, 0.8584] |
| controlled_shared_1d_residual | 20,021 | 0.8094 | 0.8004 | 0.8004 | [0.7423, 0.8540] |
| controlled_all_channel_1d_cnn | 20,197 | 0.8250 | 0.7994 | 0.7994 | [0.7404, 0.8564] |
| controlled_shared_1d_residual_identity | 21,813 | 0.8072 | 0.7973 | 0.7973 | [0.7397, 0.8493] |
| feature_random_forest_v1 | n/a | 0.8056 | 0.7845 | 0.7845 | [0.7318, 0.8376] |
| rescnn_bigru_attention_lite_v1 | 38,934 | 0.7944 | 0.7691 | 0.7691 | [0.7076, 0.8270] |
| controlled_all_channel_1d_cnn_small | 11,317 | 0.7806 | 0.7562 | 0.7562 | [0.6786, 0.8242] |
| feature_linear_svm_v1 | n/a | 0.7461 | 0.7213 | 0.7213 | [0.6612, 0.7828] |
| controlled_flatten_mlp | 594,373 | 0.7344 | 0.7046 | 0.7046 | [0.5953, 0.8074] |
| lee_style_cnn_lstm_2d_v1 | 62,637 | 0.6622 | 0.6269 | 0.6269 | [0.5237, 0.7165] |
| controlled_shared_1d_identity | 8,885 | 0.5617 | 0.5182 | 0.5182 | [0.4335, 0.5968] |
| controlled_2d_cnn | 14,853 | 0.5056 | 0.4452 | 0.4452 | [0.3472, 0.5413] |
| controlled_shared_1d | 7,093 | 0.3500 | 0.2806 | 0.2806 | [0.2233, 0.3385] |

별도 completion run에서 `feature_xgboost_v1`은 accuracy 0.8156, Macro F1 0.7961, weighted F1 0.7961, Macro F1 CI [0.7567, 0.8328]을 기록했다. 이 값은 기존 controlled result를 수정하지 않고 read-only 병합 요약에만 추가했다.

## Component 관찰

- `controlled_shared_1d`는 Macro F1 0.2806으로 낮았다.
- `controlled_shared_1d_identity`는 0.5182로 상승했지만 충분하지 않았다.
- `controlled_shared_1d_residual`은 0.8004로 크게 회복했다.
- `controlled_shared_1d_residual_identity`는 0.7973으로 residual-only shared와 거의 같았다.
- 이 controlled setting에서는 residual branch가 shared 1D encoder의 성능 회복에 가장 큰 영향을 준 것으로 관찰된다.
- identity embedding만으로는 naive shared bottleneck을 충분히 해결하지 못했다.

## Paired Differences

기준 모델 `controlled_shared_1d_residual_identity` 대비 paired Macro F1 difference는 다음과 같다. 부호는 comparison minus reference이다.

| comparison | mean difference | CI | n pairs |
|---|---:|---|---:|
| controlled_all_channel_1d_cnn | +0.0021 | [-0.0671, 0.0671] | 18 |
| controlled_all_channel_1d_cnn_small | -0.0412 | [-0.1320, 0.0326] | 18 |
| feature_random_forest_v1 | -0.0128 | [-0.0438, 0.0182] | 18 |
| rescnn_bigru_attention_lite_v1 | -0.0282 | [-0.1025, 0.0436] | 18 |
| lee_style_cnn_lstm_2d_v1 | -0.1704 | [-0.2936, -0.0592] | 18 |

`controlled_shared_1d_residual_identity`와 all-channel controlled CNN, RandomForest, ResCNN-BiGRU-Attention-lite의 CI는 0을 포함한다. Lee-style CNN-LSTM과의 차이는 이 run에서 reference가 더 높은 방향으로 나타났다.

## Class 3 Excessive Lean

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| controlled_shared_1d_residual_identity | 0.7417 | 0.7509 |
| controlled_shared_1d_residual | 0.7222 | 0.7542 |
| controlled_stats_mlp | 0.6917 | 0.7150 |
| controlled_all_channel_1d_cnn | 0.6806 | 0.6692 |
| controlled_all_channel_1d_cnn_small | 0.6639 | 0.6835 |
| feature_random_forest_v1 | 0.6500 | 0.6848 |
| rescnn_bigru_attention_lite_v1 | 0.6833 | 0.6552 |
| lee_style_cnn_lstm_2d_v1 | 0.4861 | 0.4607 |
| controlled_shared_1d | 0.1556 | 0.1573 |

Class 3에서는 residual shared 계열이 naive shared 대비 큰 개선을 보였다. 이 관찰은 기존 v2 shared underfit 진단과 일관된다.

## Classical Feature Audit

`feature_definitions.csv`와 `feature_audit.csv`는 모든 feature가 IMU signal-derived feature임을 기록한다.

- metadata 사용: 없음
- subject ID 사용: 없음
- class label 사용: 없음
- set/window index 사용: 없음
- original boundary/window length 사용: 없음

RandomForest 상위 feature는 주로 오른쪽 허벅지 `s1_ax` 관련 통계에 집중되었다.

| rank | feature | mean importance |
|---:|---|---:|
| 1 | s1_ax_std | 0.0612 |
| 2 | s1_ax_energy | 0.0505 |
| 3 | s1_ax_rms | 0.0479 |
| 4 | s1_ax_max | 0.0432 |
| 5 | s1_gx_mean | 0.0409 |
| 6 | s1_ax_mean | 0.0374 |
| 7 | s1_ax_ptp | 0.0347 |

XGBoost completion에서도 feature audit은 162개 feature 모두 `allowed=True`였고, metadata, subject ID, label, boundary, original length feature는 없었다. XGBoost 상위 feature는 `s1_ax_std`, `s1_gx_mean`, `s1_ax_energy` 순으로 나타났다.

## Read-only Reference 병합

기존 locked full matrix와 literature extension 결과는 read-only로 읽어 `analysis/merged_readonly_reference_comparison.csv`에 병합했다. 기존 결과 디렉터리는 수정하지 않았다.

주의: controlled comparison의 목적은 extractor 통제 비교이며, 기존 locked matrix와 architecture/head 조건이 완전히 같지 않은 모델도 있다. 따라서 병합 표는 논문 claim을 직접 확정하는 표가 아니라 해석 보조 표로 사용한다.

## 산출물

- `aggregate_metrics_by_model.csv`
- `aggregate_metrics_by_model_seed.csv`
- `bootstrap_confidence_intervals.csv`
- `paired_model_differences.csv`
- `classwise_metrics_by_model.csv`
- `subjectwise_metrics_by_model.csv`
- `feature_definitions.csv`
- `feature_audit.csv`
- `random_forest_feature_importance_aggregate.csv`
- `linear_svm_coefficients_aggregate.csv`
- `analysis/*.csv`
- `figures/*.png`
- `figure_captions.md`

## 해석 제한

- 결과를 보고 hyperparameter, split, preprocessing, 모델 폭을 변경하지 않았다.
- 원래 controlled run에서 XGBoost는 dependency가 없어 skipped 처리되었다. 이후 별도 completion run으로 `feature_xgboost_v1`을 실행했으며, 기존 controlled result directory는 수정하지 않았다.
- `controlled_stats_mlp`가 가장 높은 Macro F1을 보였지만, 이는 hand-crafted summary statistic이 현재 데이터에서 매우 강한 signal을 담고 있음을 보여주는 관찰이다. 최종 논문 서사에서 main model 또는 baseline 배치를 어떻게 조정할지는 별도 판단이 필요하다.
- `controlled_shared_1d_residual`과 `controlled_shared_1d_residual_identity`는 all-channel controlled CNN과 유사한 수준으로 관찰되며, naive shared 1D의 병목을 residual branch가 크게 완화했다는 설명에는 사용할 수 있다.
