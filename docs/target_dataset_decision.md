# Target Dataset Decision

## 1순위 공식 데이터 후보

논문 공식 supervised target squat classification 실험에는 다음 조합을 1순위 후보로 사용한다.

- 원천 raw session: `datasets/raw/labeled/class0~4/subject*_set*.csv`
- 수동 window boundary: `datasets/manually_labeled/class0~4/subject*_set*_metadata.json`
- 검증용 sliced window reference: `datasets/manually_labeled/class0~4/subject*_set*_window*.csv`

이 후보는 5개 class, 6명 subject, subject-class마다 2개 set, set마다 10개 manually labeled window를 제공한다. 총 supervised 후보 sample 수는 600개이다. class label은 폴더명 `class0~class4`, subject ID와 set ID는 파일명에서 복원한다.

공식 class mapping은 다음으로 고정한다.

| class_id | label |
|---:|---|
| 0 | Correct |
| 1 | Knee Valgus |
| 2 | Butt Wink |
| 3 | Excessive Lean |
| 4 | Partial Squat |

## 사용하지 말아야 할 파일 또는 폴더

공식 supervised target classification conversion에서는 다음을 제외한다.

- `datasets/raw/unlabeled`: class label이 없으므로 supervised target 평가용이 아니다.
- `datasets/manually_labeled/ssl`: SSL용 unlabeled window 후보이며 5-class label이 없다.
- `*.png`: plot/visualization sidecar이며 모델 입력이 아니다.
- `.DS_Store`: OS metadata이다.
- 기존 old repository 코드 또는 old repository의 preprocessing output: clean-room 조건에 맞지 않는다.

## Raw Data를 사용할 수 있는 경우의 장점

raw session과 metadata boundary를 함께 사용하면 다음 장점이 있다.

- 논문에 데이터 생성 과정을 raw continuous recording에서 manually labeled repetition window로 설명할 수 있다.
- conversion을 새 코드베이스에서 재현 가능하게 구현할 수 있다.
- `timestamp`, `seq`, `millis`를 QA에 활용할 수 있다.
- 파생 window CSV에 의존하는 것보다 source-of-record가 명확하다.
- window CSV와 raw slicing 결과를 비교해 boundary 적용 오류를 잡을 수 있다.

## Processed Window만 사용할 수 있는 경우의 한계

`datasets/manually_labeled/class0~4/*_window*.csv`만 사용하는 경우 다음 한계가 있다.

- 이미 잘린 파생 파일이므로 raw에서 어떻게 추출됐는지 설명력이 약해진다.
- 현재 row length가 184~461로 다양하며 target 입력인 512 time steps가 아니다.
- 512x18 변환 정책이 별도로 필요하다.
- raw와 metadata를 쓰지 않으면 window boundary QA가 어려워진다.

## 최종 포맷

v1 conversion output은 다음 파일로 고정했다.

| 파일 | shape | 내용 |
|---|---|---|
| `X.npy` | `(600, 512, 18)` | 18개 IMU 채널, 512 time-step resampled window, `float32` |
| `y.npy` | `(600,)` | integer class ID 0~4, `int64` |
| `subject_id.npy` | `(600,)` | subject ID 1~6, `int64` |
| `metadata.csv` | `N rows` | sample_id, source_raw_path, source_metadata_path, class_id, class_name, subject_id, set_id, window_id, start_sample, end_sample, original_num_samples, resampling_method, channel_order |

output directory는 `data/target/processed/v1_manual_windows_resample512/`이다.

## 제안 Channel Order

현재 CSV header에서 확인되는 18개 IMU 채널 순서는 다음과 같다.

`s0_ax`, `s0_ay`, `s0_az`, `s0_gx`, `s0_gy`, `s0_gz`, `s1_ax`, `s1_ay`, `s1_az`, `s1_gx`, `s1_gy`, `s1_gz`, `s2_ax`, `s2_ay`, `s2_az`, `s2_gx`, `s2_gy`, `s2_gz`

승인된 sensor mapping은 다음과 같다.

| sensor | location |
|---|---|
| `s0` | lower back / waist / 허리 |
| `s1` | right thigh / 오른쪽 허벅지 |
| `s2` | right calf / 오른쪽 종아리 |

원본 CSV 자체에는 위치명이 없으므로, 이 mapping은 사용자 승인값으로 config snapshot에 기록했다.

## LOSO Split 원칙

- fold key는 `subject_id` 하나만 사용한다.
- held-out subject의 모든 class, set, window는 test fold에만 둔다.
- scaler fit은 training subjects에 대해서만 수행한다.
- conversion 단계에서는 전체 데이터 통계 기반 normalization을 수행하지 않는다.
- per-window normalization을 사용할 경우 config에 명시하고 결과에 기록한다.

## Resampling 결정

각 manually labeled window는 원래 길이가 184~461 sample로 다양하다. v1에서는 padding이나 단순 truncate를 쓰지 않고 phase-normalized linear interpolation으로 모든 window를 512 time steps로 변환했다. timestamp가 엄격히 증가하는 window는 timestamp 기준 상대 phase를 사용하고, 그렇지 않으면 sample index 기준 phase를 사용한다.

conversion 단계에서는 z-score, StandardScaler, clipping, augmentation을 적용하지 않았다. normalization은 향후 LOSO fold 내부에서 train subject만 사용해 fit한다.

## Reference Window 사용 방식

`datasets/manually_labeled/class0~4/*_window*.csv`는 공식 입력으로 사용하지 않았다. 대신 raw+metadata slicing 결과와 reference window 600개의 row count, column presence, first/last timestamp, first/last 18채널 값을 비교했다. 결과는 `reference_validation.csv`에 저장했으며 600개 모두 `ok`였다.

## 현재 결정

`target dataset v1`은 생성 완료된 공식 supervised target dataset 후보이다. 이 데이터셋은 real-data training에 바로 사용할 수 있지만, 아직 full LOSO training, epoch 단위 training, hyperparameter tuning은 실행하지 않는다. 다음 단계에서는 먼저 model capacity correction 계획을 승인하고, 그 뒤 real-data smoke training을 full training과 분리해 설계한다.
