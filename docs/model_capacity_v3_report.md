# Model Capacity v3 Report

## 목적

v2 shared 모델은 encoder object를 공유했지만, channel identity를 거의 보존하지 못했다. 특히 mean pooling과 작은 attention pooling은 18개 channel token을 sensor 위치와 축 의미 없이 집계했고, pilot 및 overfit diagnostic에서 class 3 Excessive Lean을 거의 맞추지 못했다.

v3는 shared encoder 논리는 유지하되 channel identity를 명시적으로 주입한다.

## 추가 모델

### `channel_shared_posres_attention_v3`

- 18개 단일 channel sequence에 동일한 `TemporalEncoder1D` object 하나를 재사용한다.
- channel token에 channel ID, sensor ID, modality ID, axis ID embedding을 더한다.
- mean pooling 대신 작은 attention pooling으로 channel token을 집계한다.
- raw channel summary statistics 기반 residual branch를 추가한다.
- shared temporal feature와 residual feature를 concat한 뒤 작은 classifier head를 사용한다.

Residual branch의 목적은 shared encoder가 공통 temporal feature를 학습하는 동안, 각 channel의 위치별 scale/summary 정보를 완전히 잃지 않도록 하는 것이다.

### `modality_shared_sensorattn_v3`

- accelerometer 9개 channel은 acc encoder object 하나를 공유한다.
- gyroscope 9개 channel은 gyro encoder object 하나를 공유한다.
- acc encoder와 gyro encoder는 서로 다른 object이다.
- sensor ID, modality ID, axis ID embedding을 token에 더한다.
- acc token과 gyro token을 각각 attention pooling한다.
- acc/gyro representation을 concat한 뒤 gated fusion과 작은 classifier head를 적용한다.

## Parameter Count

목표 범위는 all-channel Conv1D v1의 0.5배에서 2배, 즉 약 8k-32k였다.

| model | total params | encoder | aggregation | head |
|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 16037 | 15712 | 0 | 325 |
| all_channel_conv1d_small | 4181 | 4056 | 0 | 125 |
| channel_shared_meanpool_v2 | 2677 | 2064 | 0 | 613 |
| channel_shared_attentionpool_v2 | 2950 | 2064 | 273 | 613 |
| modality_shared_meanpool_v2 | 6373 | 4128 | 0 | 2245 |
| channel_shared_posres_attention_v3 | 14742 | 2064 | 8193 | 4485 |
| modality_shared_sensorattn_v3 | 16103 | 4128 | 9730 | 2245 |

두 v3 모델 모두 목표 parameter range 안에 있다.

## 해석

`channel_shared_posres_attention_v3`는 shared temporal encoder의 parameter 수를 작게 유지하면서, channel identity embedding과 residual branch가 대부분의 추가 capacity를 담당한다. 이는 논문 가설의 핵심인 weight sharing은 보존하되, v2에서 사라진 sensor/channel identity를 되살리는 방향이다.

`modality_shared_sensorattn_v3`는 acc/gyro encoder를 분리해 modality-specific temporal pattern을 허용한다. 기존 `modality_shared_meanpool_v2`와 달리 sensor-aware attention과 gated fusion을 사용하므로 mean pooling bottleneck이 완화된다.

## 다음 검증 계획

- overfit diagnostic v2: 40개 balanced subset memorization 가능 여부 확인
- pilot LOSO v2: 1 seed, cyclic validation, 6-fold 제한 실행
- full supervised matrix는 아직 실행하지 않는다.
