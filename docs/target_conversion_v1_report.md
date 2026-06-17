# Target Conversion v1 Report

## 요약

- output_dir: `data/target/processed/v1_manual_windows_resample512`
- X shape: `(600, 512, 18)`, dtype `float32`
- y shape: `(600,)`, dtype `int64`
- subject_id shape: `(600,)`, dtype `int64`
- 전체 window 수: 600
- subject 수: 6
- class 수: 5
- NaN count: 0
- Inf count: 0
- 제외 sample 수: 0

## 공식 Source 선택

`datasets/raw/labeled`의 raw continuous CSV를 공식 source로 사용하고, `datasets/manually_labeled/class0~4`의 metadata JSON boundary를 적용했다. 기존 `*_window*.csv`는 공식 입력이 아니라 raw slicing 검증용 reference로만 사용했다.

## 센서 매핑

| sensor | location |
|---|---|
| `s0` | lower back / waist / 허리 |
| `s1` | right thigh / 오른쪽 허벅지 |
| `s2` | right calf / 오른쪽 종아리 |

## Channel Order

`s0_ax, s0_ay, s0_az, s0_gx, s0_gy, s0_gz, s1_ax, s1_ay, s1_az, s1_gx, s1_gy, s1_gz, s2_ax, s2_ay, s2_az, s2_gx, s2_gy, s2_gz`

## Resampling

각 manually labeled variable-length window를 phase-normalized linear interpolation으로 512 time steps로 변환했다. timestamp가 엄격히 증가하는 window는 timestamp 기준 상대 시간축을 사용하고, 그렇지 않으면 sample index 기준 시간축을 사용한다. Padding, 단순 truncate, z-score, StandardScaler, clipping, augmentation은 conversion 단계에서 적용하지 않았다.

## LOSO Leakage 방지

conversion 산출물에는 normalization을 적용하지 않았다. 향후 LOSO 학습에서는 held-out subject를 제외한 training subject에 대해서만 scaler를 fit해야 하며, held-out subject는 scaler, augmentation, validation tuning, SSL pretraining에 사용하지 않는다.

## Subject-Class Matrix

| subject | class0 | class1 | class2 | class3 | class4 | total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 20 | 20 | 20 | 20 | 100 |
| 2 | 20 | 20 | 20 | 20 | 20 | 100 |
| 3 | 20 | 20 | 20 | 20 | 20 | 100 |
| 4 | 20 | 20 | 20 | 20 | 20 | 100 |
| 5 | 20 | 20 | 20 | 20 | 20 | 100 |
| 6 | 20 | 20 | 20 | 20 | 20 | 100 |

## Original Length

- min: 184
- mean: 294.51
- max: 461
- std: 57.58

## Reference Window 검증

reference window CSV는 raw+metadata 추출 window와 row count, column presence, first/last timestamp, first/last 18채널 값을 비교했다. 상세 결과는 `reference_validation.csv`에 저장했다.

- reference row 수: 600
- reference status: `{"ok": 600}`
- first timestamp match: `{"True": 600}`
- last timestamp match: `{"True": 600}`

## 남은 불확실성

- sensor 위치 매핑은 사용자 승인값을 config에 기록했으며, 원본 CSV 자체에는 위치명이 없다.
- 512 변환은 interpolation 정책의 결과이므로 원본 sample 수가 512였던 데이터가 아니다.
- git initial commit은 사용자 identity 미설정으로 실패했으며, conversion에는 현재 git 상태를 함께 기록했다.

## 다음 단계

다음 단계에서는 shared encoder 계열의 aggregation head를 줄이는 model capacity correction 계획을 승인한 뒤, real-data smoke training은 full LOSO training과 분리해 아주 작은 forward/overfit 점검으로만 설계해야 한다.
