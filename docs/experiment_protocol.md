# IMU-only Squat Posture Classification Experiment Protocol

## 목적

이 문서는 3개 MPU-6050 IMU로 수집한 18채널 squat 자세 데이터를 대상으로, 작은 subject 수에서 재현 가능한 subject-independent 실험을 수행하기 위한 clean-room 프로토콜을 정의한다. 기존 저장소 코드는 재사용하거나 import하지 않는다. 논문에 사용할 공식 결과는 이 `squat_imu_experiments` 코드베이스에서 생성한다.

## 데이터 조건

Target dataset 조건은 다음으로 고정한다.

- 센서: lower back, right thigh, right calf의 3개 MPU-6050 IMU
- 각 센서 채널: 3축 accelerometer, 3축 gyroscope
- 전체 입력: window 하나당 `(512, 18)`
- 클래스 수: 5
- 클래스 매핑:
  - 0 Correct
  - 1 Knee Valgus
  - 2 Butt Wink
  - 3 Excessive Lean
  - 4 Partial Squat
- 평가: Leave-One-Subject-Out cross validation

현재 실제 데이터 위치는 상위 작업공간의 `datasets/` 폴더이다. `target dataset v1`은 `datasets/raw/labeled`의 raw CSV와 `datasets/manually_labeled/class0~4`의 metadata boundary를 사용해 `data/target/processed/v1_manual_windows_resample512/`에 생성했다. real target training은 아직 실행하지 않는다.

## Clean-room 원칙

- 기존 squat repository의 모듈을 import하지 않는다.
- 기존 코드는 명시적으로 필요할 때 historical reference로만 본다.
- 새 실험의 manifest, preprocessing, model registry, metrics, result saving은 이 프로젝트 안에서 정의한다.
- 결과 파일에는 config snapshot, seed, git commit hash, data manifest checksum을 함께 저장한다.

## Manifest 시스템

모든 sample은 manifest record를 가진다.

- `sample_id`
- `path`
- `subject_id`
- `label`
- `source`
- `window_start`
- `window_length`
- `channels`
- `file_checksum`
- `metadata`

Manifest checksum은 canonical JSON을 SHA-256으로 해시한다. 향후 real dataset manifest를 만들 때는 모든 window에 대해 subject ID와 class label을 반드시 저장한다.

## 전처리 및 누수 방지

LOSO fold마다 다음 규칙을 적용한다.

- conversion 단계에서는 z-score, StandardScaler, clipping, augmentation을 적용하지 않는다.
- scaler는 held-out subject를 제외한 training subjects로만 fit한다.
- held-out subject의 window는 scaler fit에 절대 사용하지 않는다.
- augmentation은 training windows에만 적용한다.
- per-window normalization은 optional이며 config와 result payload에 명시적으로 기록한다.
- fold별 scaler 통계, fit 대상 subject 목록, held-out subject를 저장한다.

## Target Dataset v1 Conversion

공식 supervised target source는 raw+metadata 조합이다.

- raw source: `datasets/raw/labeled/class0~4/subject*_set*.csv`
- metadata boundary: `datasets/manually_labeled/class0~4/subject*_set*_metadata.json`
- reference only: `datasets/manually_labeled/class0~4/subject*_set*_window*.csv`
- output: `data/target/processed/v1_manual_windows_resample512/`

기존 `*_window*.csv`는 이미 잘린 파생 파일이며 row length가 184~461로 다양하다. 공식 입력은 raw CSV에 metadata boundary를 적용해 재구성하고, reference window는 endpoint 검증용으로만 사용한다.

승인된 sensor mapping은 다음이다.

- `s0`: lower back / waist / 허리
- `s1`: right thigh / 오른쪽 허벅지
- `s2`: right calf / 오른쪽 종아리

승인된 channel order는 다음이다.

`s0_ax`, `s0_ay`, `s0_az`, `s0_gx`, `s0_gy`, `s0_gz`, `s1_ax`, `s1_ay`, `s1_az`, `s1_gx`, `s1_gy`, `s1_gz`, `s2_ax`, `s2_ay`, `s2_az`, `s2_gx`, `s2_gy`, `s2_gz`

각 variable-length window는 phase-normalized linear interpolation으로 512 time steps로 변환한다. timestamp가 엄격히 증가하면 timestamp 기준 상대 phase를 사용하고, 그렇지 않으면 sample index 기준 phase를 사용한다. Padding과 단순 truncate는 사용하지 않는다.

## 모델 Registry

초기 registry는 다음 이름을 고정한다.

- `classical_feature_baseline`
- `all_channel_conv1d`
- `channel_shared_1d_encoder`
- `modality_shared_acc_gyro_encoder`
- `cnn2d_baseline`

`channel_shared_1d_encoder`는 모든 입력 채널에 같은 encoder object 하나를 재사용한다. 채널마다 별도 encoder를 만들지 않는다.

`modality_shared_acc_gyro_encoder`는 accelerometer 채널 전체에 하나의 shared encoder를 쓰고, gyroscope 채널 전체에 별도의 shared encoder 하나를 쓴다.

## 필수 결과 산출물

향후 full experiment에서는 다음을 저장한다.

- accuracy
- macro F1
- class-wise precision, recall, F1
- subject-wise macro F1
- confusion matrix per fold
- aggregated confusion matrix
- parameter count
- training history
- config snapshot
- git commit hash
- data manifest checksum

## Smoke Test

첫 단계 smoke test는 synthetic data만 사용한다.

- synthetic input shape: `(N, 512, 18)`
- synthetic subject IDs 생성
- 5-class labels 생성
- 모든 registry model에 대해 input shape `(batch, 512, 18)` forward 확인
- output shape `(batch, 5)` 확인
- parameter count 출력
- dummy metrics를 `results/smoke_test/metrics.json`에 저장

Smoke test는 real target training이나 external dataset adapter를 실행하지 않는다.

## 다음 단계

model capacity correction은 v2 모델 variant로 반영했다. `channel_shared_meanpool_v2`, `channel_shared_attentionpool_v2`, `modality_shared_meanpool_v2`, `all_channel_conv1d_small`을 추가해 parameter-reduced 비교가 가능하다.

## Real-data Smoke Training

real-data smoke training은 full training이 아니다. 목적은 processed dataset, subject split, train-only scaler, 제한된 optimizer loop, 결과 저장 포맷이 실제 CAU 환경에서 작동하는지 확인하는 것이다.

현재 smoke config는 `configs/real_smoke_training_v1.yaml`이다.

- split type: `loso_smoke`
- test subject: `1`
- validation subject: `2`
- train subjects: all remaining subjects
- max epochs: `1`
- max train batches: `3`
- max validation batches: `3`
- max test batches: `3`
- augmentation: disabled
- per-window z-score: disabled
- global StandardScaler: train subjects only

실제 optimizer/backward가 들어가는 smoke training은 CAU 서버에서만 실행한다. 로컬에서는 unit test, model capacity audit, dry-run forward-only까지만 허용한다.

금지 사항:

- full LOSO training
- 모든 subject fold 반복
- 2 epoch 초과
- 5 train batches 초과
- hyperparameter tuning
- focal loss, mixup, Time-CutMix
- SSL
- external dataset adapter

Smoke metric은 성능 해석에 사용하지 않는다. full LOSO로 넘어가기 전에는 fold별 저장 규칙, validation subject 정책, checkpoint 규칙, scaler stats 검증을 별도로 확장해야 한다.

## Pilot LOSO v1

`pilot_loso_v1`은 final full experiment가 아니라, 전체 6-fold LOSO runner가 CAU 서버에서 끝까지 동작하는지 확인하기 위한 1 seed 제한 실행이다.

Config:

- `configs/pilot_loso_v1.yaml`
- seed: `42`
- models: `all_channel_conv1d_v1`, `all_channel_conv1d_small`, `channel_shared_meanpool_v2`, `channel_shared_attentionpool_v2`, `modality_shared_meanpool_v2`, `cnn2d_baseline_v1`
- max epochs: `30`
- early stopping: `val_macro_f1`, patience `8`, restore best checkpoint
- loss: `cross_entropy`
- optimizer: `adam`
- augmentation, focal loss, mixup, Time-CutMix, SSL, external dataset: disabled

Validation subject policy는 next-subject cyclic으로 고정한다.

| fold | test subject | validation subject | train subjects |
|---:|---:|---:|---|
| 1 | 1 | 2 | 3, 4, 5, 6 |
| 2 | 2 | 3 | 1, 4, 5, 6 |
| 3 | 3 | 4 | 1, 2, 5, 6 |
| 4 | 4 | 5 | 1, 2, 3, 6 |
| 5 | 5 | 6 | 1, 2, 3, 4 |
| 6 | 6 | 1 | 2, 3, 4, 5 |

각 fold에서 scaler는 해당 fold의 train subjects 4명, 즉 400개 window에만 fit한다. validation/test subject는 scaler fit, early stopping 이외의 tuning, augmentation, SSL pretraining에 사용하지 않는다.

Pilot output은 `results/pilot_loso/<timestamp>_pilot_loso_v1/`에 저장한다.

- `split_plan.csv`
- `scaler_stats_by_fold.csv`
- `fold_metrics.csv`
- `classwise_metrics.csv`
- `training_history.csv`
- `confusion_matrices.csv`
- `predictions.csv`
- `aggregate_metrics_by_model.csv`
- `subjectwise_metrics_by_model.csv`
- `failed_runs.csv`
- `figures/*.png`

Pilot metric은 pipeline 검증용이다. 1 seed 결과이므로 논문 결론이나 모델 우열의 최종 근거로 사용하지 않는다.

## Pilot Diagnostics And Overfit Sanity Check

`pilot_loso_diagnostics_v1`은 기존 pilot 결과 CSV만 분석한다. 새 학습, hyperparameter tuning, split 변경, model 변경은 수행하지 않는다.

분석 항목:

- model ranking 재계산
- fold별 metric table
- best epoch 분포
- train/val/test gap
- prediction class distribution
- class-wise metric
- subject-wise heatmap
- class 3 Excessive Lean 오분류
- prediction collapse 여부

`overfit_diagnostic_v1`은 일반화 성능 평가가 아니라 작은 balanced subset memorization sanity check다.

- subjects: 3, 4, 5, 6
- samples per class: 8
- total samples: 40
- scaler fit: selected subset only
- target train accuracy: 0.95
- max epochs: 150
- augmentation/focal loss/SSL/external dataset: disabled

Overfit diagnostic에서 train과 eval은 같은 subset을 사용해도 된다. 이 결과는 모델 구현 및 capacity sanity check로만 사용하며 논문 성능으로 해석하지 않는다.

Pilot diagnostics 결과, 현재 v2 shared 모델은 완전한 단일-class collapse는 아니지만 channel-shared mean/attention 구조가 class 3을 거의 맞추지 못하고 class 2/4로 밀리는 경향이 있다. Overfit diagnostic에서도 all-channel Conv1D 계열만 95% train accuracy에 도달했다. 다음 architecture correction에서는 새 모델 구현 전 position-aware shared encoder를 설계 후보로 검토한다.

## Shared Encoder v3 제한 검증

v3 모델은 shared encoder 논리를 유지하되 channel identity를 명시적으로 보존한다.

추가 registry:

- `channel_shared_posres_attention_v3`
- `modality_shared_sensorattn_v3`

`channel_shared_posres_attention_v3`는 18개 channel에 동일한 shared encoder object 하나를 적용한다. 각 token에는 channel ID, sensor ID, modality ID, axis ID embedding을 더한다. 이후 attention pooling을 사용하고, raw summary statistics residual branch를 concat해 classifier에 전달한다.

`modality_shared_sensorattn_v3`는 acc encoder object 하나와 gyro encoder object 하나를 사용한다. 각 modality 내부에서 sensor-aware attention pooling을 수행하고, acc/gyro representation을 gated fusion으로 결합한다.

제한 검증 결과:

- 두 v3 모델 모두 synthetic 및 real processed batch forward pass를 통과했다.
- 두 v3 모델 모두 parameter count 8k-32k 목표 범위에 들어왔다.
- overfit diagnostic v2에서 두 v3 모델 모두 95% train accuracy에 도달했다.
- pilot LOSO v2에서 `channel_shared_posres_attention_v3`는 mean macro F1 0.6439로 all-channel v1 0.6457과 거의 같았다.

이 결과는 full supervised matrix 실행 전 후보 선정을 위한 1 seed 제한 검증이다. 최종 논문 성능으로 해석하지 않는다.

## Final Protocol Pilot v1

논문용 primary validation policy 후보는 `loso_with_within_train_subject_stratified_validation`으로 둔다.

각 fold:

- test subject: 현재 LOSO held-out subject 1명, 총 100 windows
- candidate train subjects: 나머지 5명
- 각 candidate train subject-class 조합: 20 windows 중 16 train, 4 validation
- fold별 train/validation/test 크기: 400/100/100
- scaler fit: train indices 400개만 사용
- per-window z-score, augmentation, focal loss, SSL, external dataset: disabled

이 policy는 기존 next-subject cyclic validation과 다르다. Cyclic validation은 validation subject 1명을 통째로 제외했지만, final protocol은 test subject만 완전히 제외하고 나머지 subject 내부에서 class-stratified validation을 만든다. 따라서 validation은 train 후보 subject 분포를 더 직접적으로 반영한다.

`final_protocol_pilot_v1` 실행 결과:

- CAU 결과 경로: `results/final_protocol_pilot/20260617_140012_final_protocol_pilot_v1/`
- split leakage: 모든 fold 통과
- scaler leakage: 모든 fold 및 모델 통과
- fold 성공/실패: 24 성공, 0 실패
- `channel_shared_posres_attention_v3`: mean accuracy 0.8067, mean macro F1 0.7950
- `all_channel_conv1d_v1`: mean accuracy 0.8150, mean macro F1 0.7893

이 결과는 1 seed final-protocol pilot이며 최종 논문 성능으로 해석하지 않는다. 다만 split/scaler policy가 요구 조건을 만족했으므로, 다음 단계에서 3 seed full supervised matrix로 넘어갈 수 있다.

## Full Supervised Matrix v1

Full supervised matrix v1은 final validation policy를 고정한 논문 결과 후보 실행이다.

- seeds: 42, 123, 2025
- folds: 6 LOSO folds
- models: 7
- total runs: 126
- train/validation/test per fold: 400/100/100
- scaler fit: train indices only
- optimizer/loss: Adam, cross entropy
- augmentation/focal loss/balanced sampling/SSL/external dataset: disabled

실행 결과:

- CAU 결과 경로: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`
- 성공/실패: 126 성공, 0 실패
- split leakage: 모두 통과
- scaler leakage: 모두 통과
- best mean macro F1: `channel_shared_posres_attention_v3`, 0.8108

Full matrix 결과는 논문 결과 후보로 사용할 수 있다. 다만 paired CI가 all-channel baseline 대비 0을 포함하므로 `channel_shared_posres_attention_v3`의 통계적 우월성은 주장하지 않는다.
