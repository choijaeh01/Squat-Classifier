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

다음 단계는 model capacity correction이다. 현재 shared encoder 계열은 encoder object는 공유하지만 flattened aggregation head가 커서 `all_channel_conv1d`보다 total parameter count가 크다. capacity-matched 비교를 위해 channel pooling, embedding dimension 축소, small variant registry 추가 중 하나를 승인한 뒤 진행한다.

그 다음 real-data smoke training은 full LOSO training과 분리한다. 허용되는 것은 processed loader, LOSO split, scaler 누수 방지, 한 batch forward, 아주 작은 overfit sanity check 설계까지이며, full LOSO epoch training과 hyperparameter tuning은 별도 승인 전에는 실행하지 않는다.
