# v3_component_ablation_v1 Report

## 실행 개요

- 목적: `channel_shared_posres_attention_v3`의 component별 기여를 확인하는 ablation
- 실행 위치: CAU
- CAU 경로: `/home/user3/workspace/Jae/squat_imu_experiments`
- 실행 commit: `28ee3a13aa6d2405258bc4195de57b1afc285bd9`
- config: `configs/v3_component_ablation_v1.yaml`
- 결과 경로: `results/v3_component_ablation/20260617_202714_v3_component_ablation_v1/`
- 총 run 수: 4 models x 3 seeds x 6 folds = 72
- 성공/실패: 72 성공, 0 실패
- split leakage check: 72/72 통과
- scaler leakage check: 72/72 통과
- 학습 조건: final LOSO validation policy, CrossEntropy, Adam, no augmentation, no focal loss, no balanced sampling, no SSL, no external dataset

## Ablation 구현 방식

| model | 제거 또는 분리한 component | 설명 |
|---|---|---|
| channel_shared_posres_attention_v3_no_residual | residual branch 제거 | shared encoder, identity embedding, attention pooling만 사용 |
| channel_shared_posres_attention_v3_residual_only_mlp | shared encoder와 attention 제거 | v3 residual branch의 channel별 mean/std/min/max summary만 MLP에 입력 |
| channel_shared_posres_attention_v3_no_identity | token identity embedding 제거 | shared encoder, attention pooling, residual branch는 유지. residual branch에는 channel order 기반 정보가 남아 있음 |
| channel_shared_posres_meanpool_v3_no_attention | attention pooling 제거 | identity embedding과 residual branch는 유지하고 token mean pooling 사용 |

## Parameter Count

| model | total | encoder | identity | attention | residual | head |
|---|---:|---:|---:|---:|---:|---:|
| channel_shared_posres_attention_v3 | 14742 | 2064 | 832 | 545 | 6752 | 4485 |
| channel_shared_posres_attention_v3_no_residual | 5942 | 2064 | 832 | 545 | 0 | 2437 |
| channel_shared_posres_attention_v3_residual_only_mlp | 9189 | 0 | 0 | 0 | 6752 | 2437 |
| channel_shared_posres_attention_v3_no_identity | 13910 | 2064 | 0 | 545 | 6752 | 4485 |
| channel_shared_posres_meanpool_v3_no_attention | 14197 | 2064 | 832 | 0 | 6752 | 4485 |

## Ablation 결과

| model | accuracy mean | macro F1 mean | macro F1 std | macro F1 CI | weighted F1 mean |
|---|---:|---:|---:|---|---:|
| channel_shared_posres_meanpool_v3_no_attention | 0.8161 | 0.8082 | 0.1226 | [0.7471, 0.8603] | 0.8082 |
| channel_shared_posres_attention_v3_residual_only_mlp | 0.8111 | 0.8036 | 0.1130 | [0.7488, 0.8530] | 0.8036 |
| channel_shared_posres_attention_v3_no_identity | 0.7817 | 0.7675 | 0.1191 | [0.7112, 0.8195] | 0.7675 |
| channel_shared_posres_attention_v3_no_residual | 0.6261 | 0.5937 | 0.1956 | [0.4998, 0.6842] | 0.5937 |

## 기존 결과와 병합한 참고 ranking

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

이 표는 기존 locked matrix와 literature full extension을 read-only로 읽어 병합한 참고 ranking이다. 기존 locked result directory는 수정하지 않았다.

## Paired Difference vs Original v3

값은 `comparison - channel_shared_posres_attention_v3`이다.

| comparison model | mean delta macro F1 | CI | n pairs |
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
| channel_shared_posres_attention_v3_no_identity | 0.7278 | 0.7266 |
| channel_shared_posres_attention_v3_no_residual | 0.5639 | 0.5321 |
| channel_shared_posres_attention_v3_residual_only_mlp | 0.6972 | 0.7241 |
| channel_shared_posres_meanpool_v3_no_attention | 0.6806 | 0.7196 |
| feature_random_forest_v1 | 0.6500 | 0.6848 |
| rescnn_bigru_attention_lite_v1 | 0.6833 | 0.6552 |

## Attention-RF Alignment

이 분석은 inference-only v3 attention weight와 RandomForest feature importance의 post-hoc 비교다. Attention weight와 RF importance는 같은 의미가 아니므로 정성적 alignment로만 본다.

- channel-level Pearson correlation: 0.6361
- channel-level Spearman correlation: 0.3870
- v3 attention 상위 채널: `s1_ax`, `s0_ay`, `s0_ax`, `s1_az`, `s2_ax`
- RF importance 상위 feature는 `s1_ax_std`, `s1_ax_energy`, `s1_ax_rms`, `s1_ax_max`, `s1_gx_mean` 순이었다.
- group aggregate에서 RF는 `s1_right_thigh`와 accelerometer feature에 큰 비중을 보였고, v3 attention도 `s1_ax`와 `s0_ay`에 집중되는 경향을 보였다.

## 생성된 주요 산출물

- `aggregate_metrics_by_model.csv`
- `bootstrap_confidence_intervals.csv`
- `merged_with_existing_results/merged_ablation_comparison.csv`
- `merged_with_existing_results/component_contribution_summary.csv`
- `merged_with_existing_results/paired_differences_vs_v3.csv`
- `merged_with_existing_results/paired_differences_vs_rf.csv`
- `merged_with_existing_results/paired_differences_vs_rescnn_bigru.csv`
- `merged_with_existing_results/class3_component_ablation_summary.csv`
- `merged_with_existing_results/subjectwise_component_ablation_summary.csv`
- `merged_with_existing_results/attention_rf_alignment_channel.csv`
- `merged_with_existing_results/attention_rf_alignment_group.csv`
- `merged_with_existing_results/attention_rf_alignment_summary.md`
- `figures/ablation_macro_f1_bar_ci.png`
- `figures/component_contribution_delta_macro_f1.png`
- `figures/v3_attention_vs_rf_importance_channel.png`
- `figures/v3_attention_top_channels_by_class.png`

## 해석 제한

- `no_residual`의 큰 하락은 residual branch가 관찰상 중요한 component임을 시사하지만, 최종 학술 해석은 사용자가 판단한다.
- `residual_only_mlp`와 `no_attention`이 original v3와 매우 가까운 값으로 나왔으므로, attention pooling의 필수성 또는 shared encoder 자체의 독립 기여를 강하게 주장하기는 어렵다.
- `no_identity`는 token identity embedding 제거 실험이지만 residual branch에 channel-order 정보가 남아 있으므로 모든 위치 정보를 제거한 실험은 아니다.
- 결과를 보고 learning rate, model width, split, preprocessing을 변경하지 않았다.
