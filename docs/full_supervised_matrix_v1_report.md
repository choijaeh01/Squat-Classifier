# Full Supervised Matrix v1 Report

## 실행 개요

- 목적: 논문 결과 후보 supervised LOSO full matrix
- 실행 위치: CAU `/home/user3/workspace/Jae/squat_imu_experiments`
- CAU hostname: `4d244d15d634`
- Python: 3.10.12
- Device: `cuda`, NVIDIA RTX 6000 Ada Generation
- Experiment commit: `9e49ddcb64b0cfef1d77a05c83fa9d68de961ca7`
- Config: `configs/full_supervised_matrix_v1.yaml`
- 결과 경로: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`

이번 실행은 3 seeds x 6 folds x 7 models = 126 runs이다. Hyperparameter tuning, SSL, external adapter, focal loss, balanced sampling, mixup, Time-CutMix, augmentation은 사용하지 않았다.

## Final Validation Policy

Split type은 `loso_with_within_train_subject_stratified_validation`이다. 각 LOSO fold에서 test subject 1명은 완전히 held-out이다. 나머지 5명 subject의 각 class 20 windows 중 16개는 train, 4개는 validation에 배정한다.

- fold별 train/validation/test: 400/100/100
- scaler fit: train indices 400개만 사용
- validation/test: train scaler transform만 적용
- per-window z-score: false
- augmentation: false

## 실행 및 누수 검증

- 총 run 수: 126
- 성공 run 수: 126
- 실패 run 수: 0
- split summary row 수: 18 seeds-folds
- 모든 split: train 400, validation 100, test 100
- test subject isolation: 모두 `True`
- train/validation index disjoint: 모두 `True`
- scaler fit windows: 모든 row 400
- validation/test indices used in scaler: 모두 `False`
- scaler leakage check: 모두 `True`

## Parameter Count

| model | total params | encoder | aggregation | head |
|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 16037 | 15712 | 0 | 325 |
| all_channel_conv1d_small | 4181 | 4056 | 0 | 125 |
| cnn2d_baseline_v1 | 8421 | 8256 | 0 | 165 |
| channel_shared_meanpool_v2 | 2677 | 2064 | 0 | 613 |
| channel_shared_attentionpool_v2 | 2950 | 2064 | 273 | 613 |
| channel_shared_posres_attention_v3 | 14742 | 2064 | 8193 | 4485 |
| modality_shared_sensorattn_v3 | 16103 | 4128 | 9730 | 2245 |

## Overall Results

Weighted F1은 macro F1과 동일하게 나왔다. 각 test fold가 class-balanced이기 때문이다.

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| channel_shared_posres_attention_v3 | 0.8217 | 0.8108 | 0.8108 | [0.7644, 0.8537] |
| all_channel_conv1d_small | 0.8044 | 0.7740 | 0.7740 | [0.7026, 0.8424] |
| all_channel_conv1d_v1 | 0.7839 | 0.7531 | 0.7531 | [0.6502, 0.8429] |
| modality_shared_sensorattn_v3 | 0.6444 | 0.6023 | 0.6023 | [0.5003, 0.6944] |
| cnn2d_baseline_v1 | 0.4472 | 0.3688 | 0.3688 | [0.2904, 0.4489] |
| channel_shared_attentionpool_v2 | 0.3794 | 0.3136 | 0.3136 | [0.2553, 0.3731] |
| channel_shared_meanpool_v2 | 0.3172 | 0.2356 | 0.2356 | [0.1780, 0.2955] |

## Seed별 결과

| model | seed 42 macro F1 | seed 123 macro F1 | seed 2025 macro F1 |
|---|---:|---:|---:|
| all_channel_conv1d_v1 | 0.7893 | 0.7484 | 0.7217 |
| all_channel_conv1d_small | 0.8199 | 0.7551 | 0.7471 |
| cnn2d_baseline_v1 | 0.3797 | 0.3256 | 0.4012 |
| channel_shared_meanpool_v2 | 0.2357 | 0.2314 | 0.2398 |
| channel_shared_attentionpool_v2 | 0.2933 | 0.3494 | 0.2981 |
| channel_shared_posres_attention_v3 | 0.7952 | 0.8156 | 0.8216 |
| modality_shared_sensorattn_v3 | 0.6038 | 0.5705 | 0.6327 |

## Paired Differences

Paired difference는 동일 seed-fold 쌍 18개에서 comparison minus reference로 계산했다.

| reference | comparison | mean delta macro F1 | CI | n pairs |
|---|---|---:|---|---:|
| all_channel_conv1d_v1 | channel_shared_posres_attention_v3 | +0.0577 | [-0.0251, 0.1511] | 18 |
| all_channel_conv1d_small | channel_shared_posres_attention_v3 | +0.0368 | [-0.0319, 0.1097] | 18 |
| channel_shared_posres_attention_v3 | channel_shared_attentionpool_v2 | -0.4972 | [-0.5607, -0.4274] | 18 |
| channel_shared_posres_attention_v3 | channel_shared_meanpool_v2 | -0.5752 | [-0.6324, -0.5173] | 18 |

`channel_shared_posres_attention_v3`는 두 all-channel 기준보다 평균 macro F1이 높지만, paired CI가 0을 포함한다. 따라서 “명확히 우수하다”는 통계적 주장은 아직 하지 않는다. 반면 naive shared v2 대비 개선은 매우 크고 CI도 0에서 멀다.

## Subject-wise 결과

모델별 최저 subject macro F1:

| model | lowest subject | macro F1 |
|---|---:|---:|
| all_channel_conv1d_v1 | 5 | 0.4198 |
| all_channel_conv1d_small | 1 | 0.5365 |
| cnn2d_baseline_v1 | 1 | 0.1896 |
| channel_shared_meanpool_v2 | 1 | 0.1155 |
| channel_shared_attentionpool_v2 | 5 | 0.1033 |
| channel_shared_posres_attention_v3 | 5 | 0.6853 |
| modality_shared_sensorattn_v3 | 5 | 0.2535 |

Subject 5가 여러 모델에서 어려운 fold로 나타났다.

## Class-wise 및 Class 3 결과

| model | lowest recall class | lowest recall | class 3 recall | class 3 F1 |
|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 0 | 0.6861 | 0.7000 | 0.6693 |
| all_channel_conv1d_small | 3 | 0.6444 | 0.6444 | 0.6373 |
| cnn2d_baseline_v1 | 0 | 0.1889 | 0.5028 | 0.3937 |
| channel_shared_meanpool_v2 | 3 | 0.0917 | 0.0917 | 0.0840 |
| channel_shared_attentionpool_v2 | 3 | 0.1917 | 0.1917 | 0.1664 |
| channel_shared_posres_attention_v3 | 3 | 0.6889 | 0.6889 | 0.7184 |
| modality_shared_sensorattn_v3 | 0 | 0.4167 | 0.5250 | 0.4548 |

Class 3 Excessive Lean은 v3 shared 모델에서도 가장 낮은 recall class였지만, v2 shared meanpool/attention 대비 큰 개선이 확인됐다.

Aggregate confusion 기준 class 3 true windows 360개 중 `channel_shared_posres_attention_v3`는 248개를 class 3으로 맞췄고, 주요 오분류는 class 0으로 54개였다. `all_channel_conv1d_v1`은 252개를 class 3으로 맞췄고, 주요 오분류는 class 2로 55개였다.

## 해석

논문에서 주장 가능한 문장:

- Final validation policy에서 scaler leakage 없이 3-seed subject-independent LOSO 평가를 수행했다.
- `channel_shared_posres_attention_v3`는 naive channel-shared v2 모델보다 큰 폭으로 높은 macro F1을 보였다.
- Channel identity embedding과 residual branch를 포함한 position-aware shared encoder는 all-channel Conv1D와 유사하거나 더 높은 평균 macro F1을 보였다.
- 소규모 IMU squat dataset에서 단순 mean pooling shared encoder는 underfit 또는 channel identity 손실 문제가 크다.

아직 주장하면 안 되는 문장:

- `channel_shared_posres_attention_v3`가 all-channel Conv1D보다 통계적으로 유의하게 우수하다는 주장
- external IMU dataset transfer 성능에 대한 주장
- SSL 효과에 대한 주장
- augmentation, focal loss, balanced sampling의 효과에 대한 주장

## Main Model 추천 및 다음 Ablation

논문 main shared model은 `channel_shared_posres_attention_v3`를 추천한다. Baseline comparison에는 `all_channel_conv1d_v1`과 `all_channel_conv1d_small`을 함께 제시해야 한다.

다음 ablation 후보:

- v3에서 channel identity embedding 제거
- v3에서 residual branch 제거
- v3 attention pooling을 mean pooling으로 대체
- modality-shared sensor attention 구조의 fusion 방식 비교

이 ablation은 별도 승인 후 수행한다. Full matrix 결과를 보고 hyperparameter, split, preprocessing, model 구조를 임의 변경하지 않는다.
