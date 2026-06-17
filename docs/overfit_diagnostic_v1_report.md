# Overfit Diagnostic v1 Report

## 목적

이 실험은 일반화 성능 평가가 아니다. 각 모델이 작은 balanced training subset을 외울 수 있는지 확인하는 구현 및 capacity sanity check이다. 결과를 논문 성능으로 해석하지 않는다.

- 실행 위치: CAU 서버
- 원격 경로: `/home/user3/workspace/Jae/squat_imu_experiments`
- 결과 경로: `results/overfit_diagnostics/20260617_122833_overfit_diagnostic_v1/`
- config: `configs/overfit_diagnostic_v1.yaml`
- device: `cuda`
- GPU: `NVIDIA RTX 6000 Ada Generation`
- PyTorch: `2.8.0+cu129`

## Subset 구성

- subject IDs: 3, 4, 5, 6
- sampling: class-balanced
- samples per class: 8
- total samples: 40
- scaler fit: selected subset only
- augmentation: disabled
- per-window z-score: disabled

선택된 subset은 `selected_subset.csv`에 저장했다. class별 sample 수는 모두 8개다.

## 결과

Target은 train accuracy 0.95 도달 여부다.

| model | reached 95% | epoch reached | final train acc | final train macro F1 | final loss |
|---|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | true | 10 | 0.9500 | 0.9479 | 0.6915 |
| all_channel_conv1d_small | true | 18 | 0.9500 | 0.9479 | 0.8825 |
| channel_shared_meanpool_v2 | false | - | 0.6250 | 0.5716 | 0.9651 |
| channel_shared_attentionpool_v2 | false | - | 0.7000 | 0.6946 | 0.8014 |
| modality_shared_meanpool_v2 | false | - | 0.7500 | 0.7248 | 0.6332 |
| cnn2d_baseline_v1 | false | - | 0.6250 | 0.5580 | 1.1330 |

## 해석

All-channel Conv1D 계열은 40개 subset을 95%까지 외울 수 있었다. 반면 channel-shared mean/attention 및 modality-shared 모델은 150 epoch 내 95% train accuracy에 도달하지 못했다.

이는 shared 모델의 pilot LOSO 부진이 단순히 held-out subject 일반화 실패만은 아니라는 신호다. 특히 `channel_shared_meanpool_v2`는 parameter 수가 가장 적고 channel identity를 평균 pooling으로 제거하기 때문에, 작은 subset에서도 충분히 분리하지 못하는 capacity 또는 representation bottleneck이 있다.

Attention pooling은 mean pooling보다 overfit 성능이 높았다.

- meanpool: final train accuracy 0.6250
- attentionpool: final train accuracy 0.7000

Modality-shared는 channel-shared보다 더 높았다.

- channel_shared_meanpool_v2: 0.6250
- modality_shared_meanpool_v2: 0.7500

하지만 둘 다 95% target에는 도달하지 못했다.

## 결론

현재 v2 shared 모델은 완전 prediction collapse는 아니지만, 작은 subset memorization에서도 all-channel baseline보다 약하다. full supervised experiment 전에 position-aware shared encoder 또는 sensor-aware aggregation을 검토할 필요가 있다.

새 모델은 이번 단계에서 구현하지 않았다.
