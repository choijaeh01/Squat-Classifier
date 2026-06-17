# Pilot LOSO Diagnostics v1

## 목적

이 문서는 `pilot_loso_v1` 결과를 진단하기 위한 분석 기록이다. 이번 단계는 full experiment가 아니며, 1 seed pilot 결과를 최종 논문 성능이나 모델 우열의 최종 근거로 해석하지 않는다.

- 입력 결과: `results/pilot_loso/20260617_120216_pilot_loso_v1/`
- 진단 출력: `results/pilot_loso/20260617_120216_pilot_loso_v1/diagnostics/`
- 분석 스크립트: `scripts/analyze_pilot_loso.py`
- CAU에서 figure 생성 완료

## 재계산된 Model Ranking

| rank | model | mean accuracy | mean macro F1 |
|---:|---|---:|---:|
| 1 | all_channel_conv1d_v1 | 0.6817 | 0.6457 |
| 2 | all_channel_conv1d_small | 0.6417 | 0.5902 |
| 3 | modality_shared_meanpool_v2 | 0.3467 | 0.2485 |
| 4 | channel_shared_attentionpool_v2 | 0.3000 | 0.2133 |
| 5 | cnn2d_baseline_v1 | 0.3133 | 0.2078 |
| 6 | channel_shared_meanpool_v2 | 0.2550 | 0.1400 |

All-channel Conv1D 계열이 shared 계열보다 높았다. 이 결과는 1 seed pilot이므로 최종 결론이 아니라 진단 신호로만 사용한다.

## Prediction Collapse 진단

정의: dominant predicted class 비율이 70% 이상이거나 예측 class 수가 2개 이하이면 collapse로 표시했다.

| model | unique predicted classes | dominant class | dominant pct | collapse |
|---|---:|---|---:|---|
| all_channel_conv1d_v1 | 5 | Knee Valgus | 0.2867 | false |
| all_channel_conv1d_small | 5 | Knee Valgus | 0.4033 | false |
| channel_shared_meanpool_v2 | 5 | Partial Squat | 0.4100 | false |
| channel_shared_attentionpool_v2 | 5 | Partial Squat | 0.3883 | false |
| modality_shared_meanpool_v2 | 5 | Butt Wink | 0.3717 | false |
| cnn2d_baseline_v1 | 5 | Knee Valgus | 0.3100 | false |

결론: shared 모델은 단일 class만 예측하는 완전 collapse는 아니다. 다만 channel-shared mean/attention 모델은 class 3을 거의 예측하지 않고 class 2/4로 많이 밀리는 skew가 있다.

## Train/Val/Test Gap

| model | train macro F1 at best | val macro F1 at best | test macro F1 | train-test gap |
|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 0.9565 | 0.7284 | 0.6457 | 0.3108 |
| all_channel_conv1d_small | 0.9181 | 0.6825 | 0.5902 | 0.3279 |
| channel_shared_meanpool_v2 | 0.3185 | 0.3029 | 0.1400 | 0.1785 |
| channel_shared_attentionpool_v2 | 0.3860 | 0.3136 | 0.2133 | 0.1726 |
| modality_shared_meanpool_v2 | 0.5629 | 0.4701 | 0.2485 | 0.3144 |
| cnn2d_baseline_v1 | 0.4369 | 0.4322 | 0.2078 | 0.2290 |

All-channel 모델은 train macro F1이 높고 generalization gap도 있다. 반면 channel-shared 모델은 train macro F1 자체가 낮다. 이는 shared 계열의 낮은 test 성능이 단순 overfitting보다 underfitting, pooling bottleneck, 또는 channel identity 손실과 더 관련 있을 가능성을 시사한다.

## Class 3 Excessive Lean 오분류

Class 3의 정답 120개에 대한 모델별 예측 분포는 다음이다.

| model | class 3 recall | 가장 많은 예측 class | 해당 비율 |
|---|---:|---|---:|
| all_channel_conv1d_v1 | 0.5083 | Excessive Lean | 0.5083 |
| all_channel_conv1d_small | 0.4750 | Excessive Lean | 0.4750 |
| channel_shared_meanpool_v2 | 0.0417 | Butt Wink | 0.4417 |
| channel_shared_attentionpool_v2 | 0.0417 | Butt Wink | 0.4083 |
| modality_shared_meanpool_v2 | 0.1417 | Butt Wink | 0.3917 |
| cnn2d_baseline_v1 | 0.1333 | Butt Wink | 0.3417 |

Class 3은 shared 계열에서 특히 class 2 Butt Wink로 많이 오분류된다. channel-shared mean pooling은 class 3을 120개 중 5개만 맞췄고, attention pooling도 동일하게 5개만 맞췄다.

## Shared 계열 비교

Attention pooling은 mean pooling보다 pilot macro F1이 높았다.

- `channel_shared_meanpool_v2`: 0.1400
- `channel_shared_attentionpool_v2`: 0.2133

하지만 class 3 recall은 둘 다 0.0417로 동일하게 낮았다. Attention은 전체 class skew를 일부 완화했지만, class 3 구분 문제는 해결하지 못했다.

Modality-shared는 channel-shared보다 개선됐다.

- `channel_shared_meanpool_v2`: 0.1400
- `modality_shared_meanpool_v2`: 0.2485

Acc/Gyro를 나눈 공유 구조가 단일 channel-shared mean pooling보다 낫지만, all-channel baseline과의 차이는 여전히 크다.

## All-channel Small과 Channel-shared Mean 비교

`all_channel_conv1d_small`은 4,181 params, `channel_shared_meanpool_v2`는 2,677 params다. 작은 all-channel baseline이 parameter 수가 크게 많지 않은데도 macro F1 0.5902로 channel-shared meanpool의 0.1400보다 훨씬 높다. 따라서 pilot 기준 문제는 단순 parameter count가 아니라, 18개 채널의 sensor position identity와 cross-channel 관계를 mean pooling이 지워버리는 구조적 병목일 가능성이 크다.

## 산출물

CSV:

- `model_ranking_recomputed.csv`
- `fold_metric_diagnostics.csv`
- `prediction_distribution_by_model.csv`
- `class3_error_analysis.csv`
- `train_val_test_gap_by_model.csv`
- `best_epoch_distribution.csv`
- `collapse_diagnosis.csv`
- `diagnostic_summary.json`

Figures:

- `prediction_distribution_by_model.png`
- `train_val_test_gap_by_model.png`
- `class3_confusion_by_model.png`
- `best_epoch_distribution.png`
- `model_vs_class_recall_heatmap.png`
- `subject_vs_model_macro_f1_heatmap.png`

## 다음 단계 제안

새 모델은 아직 구현하지 않는다. 다음 단계에서 승인 후 검토할 v3 후보는 position-aware shared encoder다.

- shared encoder는 유지하되 sensor/channel identity embedding을 추가
- channel embedding을 단순 평균하지 않고 sensor location-aware pooling 사용
- acc/gyro modality와 sensor position을 함께 encode
- mean pooling 대신 작은 gated pooling 또는 per-sensor pooling 후 fusion 검토

이 수정은 full experiment 전에 별도 architecture correction 단계로 다루어야 한다.
