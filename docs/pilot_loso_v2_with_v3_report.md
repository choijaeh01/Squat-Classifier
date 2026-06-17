# Pilot LOSO v2 With v3 Report

## 실행 개요

v3 overfit diagnostic에서 두 v3 모델이 95% train accuracy에 도달했기 때문에, 1 seed 제한 pilot LOSO v2를 실행했다. 이 실행은 final full supervised matrix가 아니다.

- 실행 위치: CAU
- 결과 경로: `results/pilot_loso/20260617_130150_pilot_loso_v2_with_v3/`
- config: `configs/pilot_loso_v2_with_v3.yaml`
- seed: 42
- folds: 6
- validation policy: next-subject cyclic
- max epochs: 30
- augmentation, focal loss, SSL, external dataset: disabled
- 실패 fold: 0

## Aggregate Metrics

| model | mean accuracy | mean macro F1 | std macro F1 | min macro F1 | max macro F1 |
|---|---:|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 0.6817 | 0.6457 | 0.2602 | 0.2476 | 0.9484 |
| channel_shared_posres_attention_v3 | 0.6850 | 0.6439 | 0.1893 | 0.3575 | 0.7933 |
| all_channel_conv1d_small | 0.6417 | 0.5902 | 0.3306 | 0.0667 | 0.9900 |
| modality_shared_sensorattn_v3 | 0.4550 | 0.3852 | 0.2797 | 0.1015 | 0.7912 |
| modality_shared_meanpool_v2 | 0.3467 | 0.2485 | 0.1553 | 0.0784 | 0.5377 |
| channel_shared_attentionpool_v2 | 0.3000 | 0.2133 | 0.1393 | 0.0672 | 0.4056 |
| channel_shared_meanpool_v2 | 0.2550 | 0.1400 | 0.0490 | 0.0678 | 0.1957 |

## Class 3 Excessive Lean

| model | class 3 recall | dominant prediction for true class 3 |
|---|---:|---|
| all_channel_conv1d_v1 | 0.5083 | Excessive Lean |
| all_channel_conv1d_small | 0.4750 | Excessive Lean |
| channel_shared_posres_attention_v3 | 0.5167 | Excessive Lean |
| modality_shared_sensorattn_v3 | 0.2667 | Knee Valgus |
| modality_shared_meanpool_v2 | 0.1417 | Butt Wink |
| channel_shared_attentionpool_v2 | 0.0417 | Butt Wink |
| channel_shared_meanpool_v2 | 0.0417 | Butt Wink |

`channel_shared_posres_attention_v3`는 class 3 recall을 v2 channel-shared 모델의 0.0417에서 0.5167로 크게 개선했다.

## Collapse 진단

모든 모델은 5개 class를 모두 예측했다. dominant predicted class 비율은 모두 70% 미만이므로 collapse로 표시된 모델은 없다.

## 해석

`channel_shared_posres_attention_v3`는 1 seed pilot에서 all-channel v1과 거의 같은 mean macro F1을 보였다. 이 모델은 shared encoder 구조를 유지하면서 channel identity와 residual branch를 추가했기 때문에, shared 구조 논문 claim을 살릴 수 있는 가장 강한 후보이다.

`modality_shared_sensorattn_v3`는 기존 `modality_shared_meanpool_v2`보다 개선됐지만 all-channel 및 posres v3보다는 낮다. full matrix에는 보조 shared 후보로 포함할 수 있으나, 주 shared 모델 후보는 `channel_shared_posres_attention_v3`가 더 적합하다.

주의: 이 결과는 1 seed pilot이다. 최종 논문 성능으로 해석하지 않는다.
