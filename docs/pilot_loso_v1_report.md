# Pilot LOSO v1 Report

## 실행 개요

이번 실행은 final full experiment가 아니라 1 seed pilot LOSO이다. 목적은 6개 subject 전체 fold runner, cyclic validation subject 정책, fold별 train-only scaler, best checkpoint 복원, 결과 aggregation, figure 저장이 CAU 서버에서 끝까지 동작하는지 확인하는 것이다.

- 실행 위치: CAU 서버
- CAU hostname: `4d244d15d634`
- 원격 경로: `/home/user3/workspace/Jae/squat_imu_experiments`
- Python: `3.10.12`
- PyTorch: `2.8.0+cu129`
- device: `cuda`
- GPU: `NVIDIA RTX 6000 Ada Generation`
- config: `configs/pilot_loso_v1.yaml`
- 결과 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/`
- 로컬 회수 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/`

## Config 요약

- seed: `42`
- dataset: `data/target/processed/v1_manual_windows_resample512`
- models: 6개
- max epochs: `30`
- early stopping: `val_macro_f1`, mode `max`, patience `8`, min delta `0.0001`, restore best
- loss: `cross_entropy`
- optimizer: `adam`, learning rate `0.001`, weight decay `0.0`
- augmentation: disabled
- focal loss: disabled
- mixup, Time-CutMix: disabled
- SSL: disabled
- external dataset: disabled
- scaler: global StandardScaler, fold별 train subjects only
- per-window z-score: disabled

## Validation Subject 정책

Validation subject는 next-subject cyclic으로 고정했다.

| fold | test subject | validation subject | train subjects |
|---:|---:|---:|---|
| 1 | 1 | 2 | 3, 4, 5, 6 |
| 2 | 2 | 3 | 1, 4, 5, 6 |
| 3 | 3 | 4 | 1, 2, 5, 6 |
| 4 | 4 | 5 | 1, 2, 3, 6 |
| 5 | 5 | 6 | 1, 2, 3, 4 |
| 6 | 6 | 1 | 2, 3, 4, 5 |

각 fold는 `n_train=400`, `n_val=100`, `n_test=100`이다. 모든 fold에서 `leakage_check_passed=True`였다.

## Scaler 누수 확인

`scaler_stats_by_fold.csv`에 fold별 scaler fit 범위를 저장했다.

- fit scope: `train_subjects_only`
- fold별 fit sample count: `400`
- fold별 fit window count: `204800`
- validation/test subject는 scaler fit에 포함되지 않음
- per-window z-score는 적용하지 않음

## 모델별 Parameter Count

| model | total params | encoder | aggregation | head |
|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 16037 | 15712 | 0 | 325 |
| all_channel_conv1d_small | 4181 | 4056 | 0 | 125 |
| channel_shared_meanpool_v2 | 2677 | 2064 | 0 | 613 |
| channel_shared_attentionpool_v2 | 2950 | 2064 | 273 | 613 |
| modality_shared_meanpool_v2 | 6373 | 4128 | 0 | 2245 |
| cnn2d_baseline_v1 | 8421 | 8256 | 0 | 165 |

## Fold 성공 여부

모든 모델과 모든 fold가 성공했다.

- 총 실행: 6 models x 6 folds = 36
- 성공 folds: 36
- 실패 folds: 0
- `failed_runs.csv`: 빈 파일로 저장됨

## Pilot Aggregate Metrics

아래 수치는 1 seed pilot 결과이며 최종 성능으로 해석하지 않는다.

| model | mean accuracy | std accuracy | mean macro F1 | std macro F1 | success folds |
|---|---:|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 0.6817 | 0.2247 | 0.6457 | 0.2602 | 6 |
| all_channel_conv1d_small | 0.6417 | 0.2803 | 0.5902 | 0.3306 | 6 |
| modality_shared_meanpool_v2 | 0.3467 | 0.1408 | 0.2485 | 0.1553 | 6 |
| channel_shared_attentionpool_v2 | 0.3000 | 0.0952 | 0.2133 | 0.1393 | 6 |
| cnn2d_baseline_v1 | 0.3133 | 0.0862 | 0.2078 | 0.0758 | 6 |
| channel_shared_meanpool_v2 | 0.2550 | 0.0479 | 0.1400 | 0.0490 | 6 |

## Subject-wise 결과

모델별 최저 test macro F1 fold는 다음과 같다.

| model | lowest subject | accuracy | macro F1 |
|---|---:|---:|---:|
| all_channel_conv1d_v1 | 1 | 0.3500 | 0.2476 |
| all_channel_conv1d_small | 1 | 0.2000 | 0.0667 |
| channel_shared_meanpool_v2 | 5 | 0.2000 | 0.0678 |
| channel_shared_attentionpool_v2 | 5 | 0.2000 | 0.0672 |
| modality_shared_meanpool_v2 | 5 | 0.2000 | 0.0784 |
| cnn2d_baseline_v1 | 2 | 0.2300 | 0.1298 |

전체 model-fold 조합에서 가장 낮은 fold는 `all_channel_conv1d_small`, test subject `1`, macro F1 `0.0667`이었다.

## Class-wise 결과

모든 model-fold 조합 평균 기준 class별 recall/F1은 다음이다.

| class_id | class name | mean recall | mean F1 |
|---:|---|---:|---:|
| 0 | Correct | 0.2417 | 0.2268 |
| 1 | Knee Valgus | 0.4486 | 0.3329 |
| 2 | Butt Wink | 0.5750 | 0.4320 |
| 3 | Excessive Lean | 0.2236 | 0.2152 |
| 4 | Partial Squat | 0.6264 | 0.4977 |

이 pilot 기준으로 가장 어려운 class는 class `3` Excessive Lean이었다. mean recall과 mean F1 모두 가장 낮았다.

## Confusion Matrix 요약

`confusion_matrices.csv`에는 fold별 confusion matrix와 model별 aggregated confusion matrix를 함께 저장했다. Figure `aggregated_confusion_matrix_per_model.png`도 생성됐다. Pilot에서는 일부 모델이 특정 class로 예측이 몰리는 fold가 있으며, 특히 shared mean/attention 계열은 subject 5 등에서 낮은 macro F1을 보였다.

## 생성된 Figure

다음 figure가 생성됐다.

- `figures/model_macro_f1_bar.png`
- `figures/model_accuracy_bar.png`
- `figures/subjectwise_macro_f1_heatmap.png`
- `figures/classwise_recall_heatmap.png`
- `figures/aggregated_confusion_matrix_per_model.png`
- `figures/parameter_count_vs_macro_f1.png`
- `figures/training_curves_per_model_fold.png`

## 해석 주의

이 결과는 1 seed pilot이다. 최종 논문 성능, 모델 간 통계적 우열, shared encoder 가설의 최종 결론으로 사용하지 않는다. 성능이 낮은 모델이나 fold가 있어도 이번 단계에서는 hyperparameter, preprocessing, label, split, 모델 구조를 바꾸지 않았다.

## Full Supervised Experiment 전 수정 필요 사항

- CAU 실행 전 code commit hash가 dirty 상태가 되지 않도록 실행 직전 commit 절차를 고정한다.
- full supervised runner에서는 3 seed 전체 matrix를 별도 승인 후 실행한다.
- fold별 checkpoint 용량 관리 정책을 정한다.
- validation subject cyclic 정책을 final protocol로 유지할지 검토한다.
- 결과 table에 confidence interval 또는 seed별 분산 요약을 추가한다.
- tuning, augmentation, focal loss, SSL, external transfer는 supervised baseline 확정 뒤 별도 ablation으로만 다룬다.
