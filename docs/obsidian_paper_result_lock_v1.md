# Paper Result Lock v1 상세 보고서

작성일: 2026-06-18  
프로젝트: `squat_imu_experiments`  
Locked result: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`

## 1. 요약

이번 단계에서는 새 학습 없이 `full_supervised_matrix_v1` 결과를 논문 후보 결과로 lock했다. 총 126 runs는 모두 성공했으며, split leakage와 scaler leakage check가 모두 통과했다.

핵심 결론은 다음과 같다.

- Main proposed model 후보: `channel_shared_posres_attention_v3`
- 평균 Macro F1: 0.8108
- all-channel v1 대비 paired delta: +0.0577, CI [-0.0251, 0.1511]
- all-channel small 대비 paired delta: +0.0368, CI [-0.0319, 0.1097]
- paired CI가 0을 포함하므로 통계적 우월성은 주장하지 않는다.
- 대신 “position-aware shared encoder가 all-channel baselines와 경쟁 가능한 성능을 보였고 naive shared v2보다 크게 개선됐다”는 claim이 안전하다.

## 2. Result Integrity Check

| 항목 | 결과 |
|---|---:|
| total runs | 126 |
| successful runs | 126 |
| failed runs | 0 |
| model-seed-fold combinations | complete |
| leakage checks | passed |
| scaler leakage checks | passed |
| fold size | train 400, val 100, test 100 |
| aggregate metric consistency | passed |

## 3. Main Model Comparison

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| channel_shared_posres_attention_v3 | 0.8217 | 0.8108 | 0.8108 | [0.7644, 0.8537] |
| all_channel_conv1d_small | 0.8044 | 0.7740 | 0.7740 | [0.7026, 0.8424] |
| all_channel_conv1d_v1 | 0.7839 | 0.7531 | 0.7531 | [0.6502, 0.8429] |
| modality_shared_sensorattn_v3 | 0.6444 | 0.6023 | 0.6023 | [0.5003, 0.6944] |
| cnn2d_baseline_v1 | 0.4472 | 0.3688 | 0.3688 | [0.2904, 0.4489] |
| channel_shared_attentionpool_v2 | 0.3794 | 0.3136 | 0.3136 | [0.2553, 0.3731] |
| channel_shared_meanpool_v2 | 0.3172 | 0.2356 | 0.2356 | [0.1780, 0.2955] |

![[figures/fig1_model_macro_f1_ci.png]]

Korean caption: 모델별 평균 Macro F1과 bootstrap confidence interval 비교.  
English caption: Mean Macro F1 with bootstrap confidence intervals across supervised LOSO runs.

## 4. Parameter Count vs Performance

`channel_shared_posres_attention_v3`는 14,742 parameters로 all-channel v1의 16,037 parameters와 유사한 범위에 있으며, full matrix에서 더 높은 평균 Macro F1을 기록했다.

![[figures/fig2_parameter_count_vs_macro_f1.png]]

Korean caption: 모델 parameter count와 평균 Macro F1의 관계.  
English caption: Parameter count versus mean Macro F1 for each model.

## 5. Class-wise Analysis

Class 3 Excessive Lean은 여러 모델에서 어려운 class다. v3도 class 3 recall이 최저 class였지만, naive shared v2 대비 크게 개선됐다.

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| channel_shared_posres_attention_v3 | 0.6889 | 0.7184 |
| all_channel_conv1d_v1 | 0.7000 | 0.6693 |
| all_channel_conv1d_small | 0.6444 | 0.6373 |
| modality_shared_sensorattn_v3 | 0.5250 | 0.4548 |
| cnn2d_baseline_v1 | 0.5028 | 0.3937 |
| channel_shared_attentionpool_v2 | 0.1917 | 0.1664 |
| channel_shared_meanpool_v2 | 0.0917 | 0.0840 |

![[figures/fig3_classwise_recall_heatmap.png]]

Korean caption: 모델별 class-wise recall heatmap.  
English caption: Class-wise recall heatmap across models.

![[figures/fig4_class3_recall_f1_by_model.png]]

Korean caption: Class 3 Excessive Lean의 recall과 F1 비교.  
English caption: Recall and F1 for Class 3 Excessive Lean.

## 6. Subject-wise Analysis

Subject 5는 여러 모델에서 어려운 held-out subject로 나타났다. 특히 all-channel v1과 v3 모두 subject 5가 낮은 fold에 포함된다.

![[figures/fig5_subjectwise_macro_f1_heatmap.png]]

Korean caption: subject-wise Macro F1 heatmap.  
English caption: Subject-wise Macro F1 heatmap.

## 7. Paired Difference

| comparison | paired delta Macro F1 | CI |
|---|---:|---|
| v3 minus all_channel_conv1d_v1 | +0.0577 | [-0.0251, 0.1511] |
| v3 minus all_channel_conv1d_small | +0.0368 | [-0.0319, 0.1097] |

![[figures/fig6_v3_vs_baseline_paired_delta.png]]

Korean caption: v3와 all-channel baselines의 paired Macro F1 차이.  
English caption: Paired Macro F1 differences between the proposed v3 model and all-channel baselines.

## 8. Confusion Matrix

![[figures/fig7_aggregate_confusion_matrix_v3.png]]

Korean caption: v3 모델의 aggregate confusion matrix.  
English caption: Aggregate confusion matrix for the proposed v3 model.

![[figures/fig8_aggregate_confusion_matrix_all_channel_v1.png]]

Korean caption: all-channel Conv1D v1의 aggregate confusion matrix.  
English caption: Aggregate confusion matrix for the all-channel Conv1D v1 baseline.

## 9. 안전한 Claim

- Full supervised LOSO matrix에서 v3가 가장 높은 평균 Macro F1을 기록했다.
- v3는 all-channel baselines와 경쟁 가능한 성능을 보였다.
- naive shared v2의 낮은 결과는 channel identity 손실 및 pooling bottleneck 가능성을 시사한다.
- v3의 channel/sensor/modality/axis identity와 residual branch는 shared encoder 구조를 안정화하는 데 기여한 것으로 해석할 수 있다.

## 10. 피해야 할 Claim

- v3가 all-channel Conv1D보다 통계적으로 유의하게 우수하다는 주장
- SSL 또는 external transfer 효과에 대한 주장
- augmentation, focal loss, balanced sampling이 효과적이라는 주장
- 현재 결과가 larger population에 일반화된다는 강한 주장

## 11. Ablation 판단

지금은 추가 ablation보다 논문 초안을 먼저 작성하고 교수님 피드백을 받는 것을 추천한다. 최소 ablation이 필요하다면 `v3 without residual branch` 또는 `v3 without identity embeddings`가 가장 직접적으로 claim을 방어한다.

## 12. 다음 단계 추천

다음 단계는 교수님 보고자료 또는 KIEE 논문 초안 작성이다. 현재 결과는 논문 후보 결과로 lock했으므로, 새 실험보다 Method, Results, Discussion 문장을 정리하는 것이 우선이다.
