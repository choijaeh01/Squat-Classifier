# v3_component_ablation_v1 Report

## 실행 개요

- 목적: `channel_shared_posres_attention_v3`의 residual branch, identity embedding, attention pooling 기여를 분리해 확인
- 실행 위치: CAU 서버
- CAU hostname: `4d244d15d634`
- CAU project path: `/home/user3/workspace/Jae/squat_imu_experiments`
- 실행 commit: `28ee3a13aa6d2405258bc4195de57b1afc285bd9`
- 결과 경로: `results/v3_component_ablation/20260617_202714_v3_component_ablation_v1/`
- 기존 locked full matrix: read-only reference
- 기존 literature full extension: read-only reference
- 새 학습 범위: 4 ablation models x 3 seeds x 6 folds = 72 runs

## 고정 Protocol

- split: `loso_with_within_train_subject_stratified_validation`
- subjects: 1-6
- 각 fold: train 400, validation 100, test 100 windows
- scaler: train indices 400개에만 fit
- validation/test: train scaler로 transform만 수행
- per-window z-score: false
- augmentation/focal loss/balanced sampling/SSL/external dataset: disabled
- loss/optimizer: Cross Entropy, Adam
- max epochs: 120
- early stopping: validation Macro F1, patience 25

## 실행 무결성

| item | result |
|---|---:|
| expected runs | 72 |
| successful runs | 72 |
| failed runs | 0 |
| fold metric rows | 72 |
| split leakage passed | 72/72 |
| scaler leakage passed | 72/72 |

## Ablation 모델 정의

| model | 제거한 요소 | 남긴 요소 |
|---|---|---|
| channel_shared_posres_attention_v3_no_residual | raw-summary residual branch | shared encoder, channel/sensor/modality/axis embeddings, attention pooling |
| channel_shared_posres_attention_v3_residual_only_mlp | shared encoder, identity embeddings, attention pooling | original v3 raw summary statistics MLP |
| channel_shared_posres_attention_v3_no_identity | channel/sensor/modality/axis token embeddings | shared encoder, attention pooling, residual branch |
| channel_shared_posres_meanpool_v3_no_attention | attention pooling | shared encoder, identity embeddings, residual branch, mean pooling |

## Parameter Count

| model | total | encoder | identity | attention | residual | head |
|---|---:|---:|---:|---:|---:|---:|
| channel_shared_posres_attention_v3 | 14742 | 2064 | 832 | 545 | 6752 | 4485 |
| channel_shared_posres_attention_v3_no_residual | 5942 | 2064 | 832 | 545 | 0 | 2437 |
| channel_shared_posres_attention_v3_residual_only_mlp | 9189 | 0 | 0 | 0 | 6752 | 2437 |
| channel_shared_posres_attention_v3_no_identity | 13910 | 2064 | 0 | 545 | 6752 | 4485 |
| channel_shared_posres_meanpool_v3_no_attention | 14197 | 2064 | 832 | 0 | 6752 | 4485 |

## 3 Seed 결과

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| channel_shared_posres_meanpool_v3_no_attention | 0.8161 | 0.8082 | 0.8082 | [0.7471, 0.8603] |
| channel_shared_posres_attention_v3_residual_only_mlp | 0.8111 | 0.8036 | 0.8036 | [0.7488, 0.8530] |
| channel_shared_posres_attention_v3_no_identity | 0.7817 | 0.7675 | 0.7675 | [0.7112, 0.8195] |
| channel_shared_posres_attention_v3_no_residual | 0.6261 | 0.5937 | 0.5937 | [0.4998, 0.6842] |

## Existing Results와 병합한 Ranking

| rank | model | result type | macro F1 | macro F1 CI |
|---:|---|---|---:|---|
| 1 | channel_shared_posres_attention_v3 | locked_3seed_full | 0.8108 | [0.7644, 0.8537] |
| 2 | channel_shared_posres_meanpool_v3_no_attention | v3_component_ablation_3seed_full | 0.8082 | [0.7471, 0.8603] |
| 3 | channel_shared_posres_attention_v3_residual_only_mlp | v3_component_ablation_3seed_full | 0.8036 | [0.7488, 0.8530] |
| 4 | feature_random_forest_v1 | literature_extension_3seed_full | 0.7845 | [0.7318, 0.8376] |
| 5 | all_channel_conv1d_small | locked_3seed_full | 0.7740 | [0.7026, 0.8424] |
| 6 | rescnn_bigru_attention_lite_v1 | literature_extension_3seed_full | 0.7691 | [0.7076, 0.8270] |
| 7 | channel_shared_posres_attention_v3_no_identity | v3_component_ablation_3seed_full | 0.7675 | [0.7112, 0.8195] |
| 8 | all_channel_conv1d_v1 | locked_3seed_full | 0.7531 | [0.6502, 0.8429] |
| 9 | channel_shared_posres_attention_v3_no_residual | v3_component_ablation_3seed_full | 0.5937 | [0.4998, 0.6842] |

주의: 위 표는 같은 final validation policy의 3 seed 결과를 병합한 reference table이다. 기존 locked matrix와 literature extension 결과 디렉터리는 수정하지 않았다.

## Paired Difference vs Original v3

`mean_difference`는 comparison minus original v3이다.

| comparison model | mean difference | CI | n pairs |
|---|---:|---|---:|
| channel_shared_posres_attention_v3_no_residual | -0.2171 | [-0.2945, -0.1449] | 18 |
| channel_shared_posres_attention_v3_residual_only_mlp | -0.0072 | [-0.0520, 0.0421] | 18 |
| channel_shared_posres_attention_v3_no_identity | -0.0434 | [-0.0852, -0.0072] | 18 |
| channel_shared_posres_meanpool_v3_no_attention | -0.0026 | [-0.0402, 0.0343] | 18 |
| feature_random_forest_v1 | -0.0263 | [-0.0553, 0.0019] | 18 |
| rescnn_bigru_attention_lite_v1 | -0.0417 | [-0.1051, 0.0192] | 18 |

## Class 3 Excessive Lean

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| channel_shared_posres_attention_v3 | 0.6889 | 0.7184 |
| channel_shared_posres_attention_v3_no_residual | 0.5639 | 0.5321 |
| channel_shared_posres_attention_v3_residual_only_mlp | 0.6972 | 0.7241 |
| channel_shared_posres_attention_v3_no_identity | 0.7278 | 0.7266 |
| channel_shared_posres_meanpool_v3_no_attention | 0.6806 | 0.7196 |
| feature_random_forest_v1 | 0.6500 | 0.6848 |
| rescnn_bigru_attention_lite_v1 | 0.6833 | 0.6552 |

## Attention-RF Alignment

이 분석은 inference-only v3 attention weight와 RandomForest feature importance의 post-hoc 비교다. Attention weight와 feature importance는 같은 의미가 아니므로 정성적 alignment로만 해석한다.

- channel-level Pearson correlation: 0.6361
- channel-level Spearman correlation: 0.3870
- v3 attention 상위 채널: `s1_ax`, `s0_ay`, `s0_ax`, `s1_az`, `s2_ax`
- RF top feature는 `s1_ax` 관련 통계량이 다수 포함됨
- sensor group 기준 RF importance sum은 `s1_right_thigh`가 가장 큼

## Figure 후보

| file | 용도 |
|---|---|
| `figures/ablation_macro_f1_bar_ci.png` | ablation model별 Macro F1 CI |
| `figures/ablation_accuracy_bar_ci.png` | ablation model별 accuracy |
| `figures/component_contribution_delta_macro_f1.png` | component 제거에 따른 paired delta |
| `figures/class3_ablation_f1_recall.png` | Class 3 recall/F1 |
| `figures/subjectwise_ablation_heatmap.png` | subject-wise Macro F1 |
| `figures/residual_only_vs_rf_macro_f1.png` | residual-only MLP와 RandomForest 참고 비교 |
| `figures/v3_attention_vs_rf_importance_channel.png` | channel-level attention vs RF importance |
| `figures/v3_attention_top_channels_by_class.png` | class별 v3 top attention channel |

## 수치 중심 관찰

- `no_residual`은 original v3 대비 Macro F1이 크게 낮았다.
- `no_identity`도 original v3 대비 낮았고 paired CI가 0을 포함하지 않았다.
- `no_attention`은 original v3와 매우 근접했고 paired CI가 0을 포함했다.
- `residual_only_mlp`도 original v3와 근접했고 paired CI가 0을 포함했다.
- Class 3에서는 `no_identity`, `residual_only_mlp`, `no_attention`이 original v3와 유사하거나 약간 높게 관찰됐다.

## 해석 제한

- 이 ablation은 결과를 보고 모델 구조나 hyperparameter를 변경하지 않은 고정 protocol 실행이다.
- `residual_only_mlp`가 높게 나왔다는 사실만으로 proposed model을 바꿔야 한다고 결론내리지는 않는다.
- `no_attention`이 original v3와 유사하다는 사실은 attention pooling의 필요성을 재검토할 근거지만, 최종 학술 해석은 사용자 판단으로 남긴다.
- paired CI가 0을 포함하는 비교에서는 우월성 또는 열등성 표현을 피한다.
