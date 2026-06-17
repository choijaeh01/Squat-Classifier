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
