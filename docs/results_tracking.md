# Results Tracking

이 문서는 clean-room squat IMU 실험의 실행 이력을 추적한다. `results/` 아래 산출물은 git에 commit하지 않으며, 문서에는 실행 조건과 해석 범위만 기록한다.

## real_smoke_training_v1

- 목적: 실제 processed target dataset으로 제한된 학습 루프가 CAU에서 동작하는지 확인
- 실행 위치: CAU
- 결과 경로: `results/smoke_training/20260617_094334_real_smoke_training_v1/`
- 범위: test subject 1, validation subject 2, train subjects 3/4/5/6
- 제한: max epoch 1, max train batches 3
- 상태: 6개 모델 모두 성공
- 해석: smoke metric은 성능 해석에 사용하지 않음

## pilot_loso_v1

- 목적: 1 seed 전체 6-fold LOSO runner 검증
- 실행 위치: CAU
- 원격 경로: `/home/user3/workspace/Jae/squat_imu_experiments`
- 결과 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/`
- seed: 42
- folds: 6
- models: 6
- 성공/실패: 36 성공, 0 실패
- leakage: 모든 fold에서 `leakage_check_passed=True`
- scaler: fold별 train subjects only, fold별 400 windows fit

Pilot aggregate metrics:

| model | mean accuracy | mean macro F1 |
|---|---:|---:|
| all_channel_conv1d_v1 | 0.6817 | 0.6457 |
| all_channel_conv1d_small | 0.6417 | 0.5902 |
| modality_shared_meanpool_v2 | 0.3467 | 0.2485 |
| channel_shared_attentionpool_v2 | 0.3000 | 0.2133 |
| cnn2d_baseline_v1 | 0.3133 | 0.2078 |
| channel_shared_meanpool_v2 | 0.2550 | 0.1400 |

주의: 이 표는 pipeline 검증용 1 seed pilot 결과이다. 최종 논문 성능이나 모델 우열로 해석하지 않는다.

## pilot_loso_diagnostics_v1

- 목적: `pilot_loso_v1`의 shared 모델 부진 원인 진단
- 입력 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/`
- 출력 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/diagnostics/`
- 실행 위치: 로컬 분석 및 CAU figure 생성
- collapse 기준: dominant predicted class 70% 이상 또는 unique predicted class 2개 이하
- 결론: 모든 모델이 5개 class를 예측했으며 완전 collapse는 아님
- 주요 문제: channel-shared 모델이 class 3 Excessive Lean을 거의 맞추지 못하고 class 2 Butt Wink 및 class 4 Partial Squat으로 밀림

Class 3 recall:

| model | class 3 recall |
|---|---:|
| all_channel_conv1d_v1 | 0.5083 |
| all_channel_conv1d_small | 0.4750 |
| modality_shared_meanpool_v2 | 0.1417 |
| cnn2d_baseline_v1 | 0.1333 |
| channel_shared_meanpool_v2 | 0.0417 |
| channel_shared_attentionpool_v2 | 0.0417 |

## overfit_diagnostic_v1

- 목적: 40개 balanced subset memorization sanity check
- 실행 위치: CAU
- 결과 경로: `results/overfit_diagnostics/20260617_122833_overfit_diagnostic_v1/`
- subset: subjects 3, 4, 5, 6에서 class별 8개, 총 40개
- scaler: selected subset only
- 일반화 성능 해석 금지

| model | reached 95% train acc | epoch | final train acc |
|---|---|---:|---:|
| all_channel_conv1d_v1 | true | 10 | 0.9500 |
| all_channel_conv1d_small | true | 18 | 0.9500 |
| channel_shared_meanpool_v2 | false | - | 0.6250 |
| channel_shared_attentionpool_v2 | false | - | 0.7000 |
| modality_shared_meanpool_v2 | false | - | 0.7500 |
| cnn2d_baseline_v1 | false | - | 0.6250 |

해석: shared 계열은 full generalization 이전 단계인 small subset memorization에서도 제한이 보인다. 다음 architecture correction에서는 position-aware shared encoder를 검토한다.
