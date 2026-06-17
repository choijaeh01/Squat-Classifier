# Model Capacity Correction v1

## 목적

기존 shared encoder 모델은 encoder object를 실제로 공유했지만, channel별 embedding을 모두 flatten한 뒤 큰 Linear head로 넘겼다. 그 결과 shared encoder의 장점이 head parameter 증가로 상쇄되었다. 이번 correction은 기존 모델을 삭제하지 않고 v2 variant를 추가해 capacity-reduced 비교를 가능하게 만드는 것이다.

## 기존 Shared 모델이 Baseline보다 컸던 이유

기존 `channel_shared_1d_encoder`의 encoder 자체는 2,064 parameters로 작다. 하지만 18개 channel 각각에서 나온 32차원 embedding을 flatten하면 classifier 입력이 `18 x 32 = 576` 차원이 된다. 이후 `Linear(576, 64)`가 들어가면서 이 layer만 36,928 parameters가 된다. 최종 head 총합은 37,253 parameters이고, 전체 모델은 39,317 parameters가 된다.

`modality_shared_acc_gyro_encoder`도 같은 문제가 있다. acc encoder와 gyro encoder는 각각 공유되지만, 최종적으로 18개 channel embedding을 flatten해 동일한 큰 head로 넘기므로 전체 41,381 parameters가 된다.

따라서 문제는 weight sharing 실패가 아니라 flatten aggregation head 설계였다.

## v2에서 Flatten Head를 제거한 이유

v2 shared 모델은 channel별 embedding을 모두 classifier에 직접 넘기지 않는다. 대신 channel dimension을 pooling으로 먼저 줄인다.

- `channel_shared_meanpool_v2`: 18개 channel embedding의 평균을 사용한다.
- `channel_shared_attentionpool_v2`: 작은 attention scorer로 18개 channel embedding의 가중합을 만든다.
- `modality_shared_meanpool_v2`: acc 9개 channel을 mean pooling하고, gyro 9개 channel을 mean pooling한 뒤 두 embedding만 concat한다.

이 구조는 shared encoder가 학습한 일반 단일-channel IMU 특징을 유지하면서, head가 channel 수에 비례해 커지는 문제를 막는다.

## Mean Pooling과 Attention Pooling의 차이

Mean pooling은 parameter가 없고 가장 보수적인 aggregation이다. 모든 channel embedding을 같은 비중으로 평균내므로 과적합 위험이 작고 해석도 단순하다. 대신 특정 channel이 더 중요한 상황을 표현하기 어렵다.

Attention pooling은 작은 scorer를 추가해 channel별 가중치를 학습한다. 이번 구현의 attention scorer는 `Linear(32, 8)`, `Tanh`, `Linear(8, 1)`만 사용하며 aggregation parameter는 273개이다. flatten head처럼 18개 embedding 전체를 큰 Linear에 넣지 않는다. parameter 증가가 작기 때문에 이번 v2에는 포함했다.

## Acc/Gyro Modality-shared 구조의 의도

`modality_shared_meanpool_v2`는 accelerometer 9개 channel에 acc encoder 하나를 공유하고, gyroscope 9개 channel에 gyro encoder 하나를 공유한다. acc와 gyro의 물리적 의미가 다르므로 두 encoder는 서로 다른 object로 둔다. 각 modality 내부에서는 channel mean pooling을 적용하고, 최종적으로 acc pooled embedding과 gyro pooled embedding만 concat한다.

이 구조는 channel-shared 모델보다 parameter는 크지만, acc/gyro modality 차이를 표현할 여지를 준다.

## Parameter Count 비교표

| model | total | encoder | aggregation | head | 해석 |
|---|---:|---:|---:|---:|---|
| `all_channel_conv1d_v1` | 16037 | 15712 | 0 | 325 | 기존 all-channel Conv1D baseline |
| `all_channel_conv1d_small` | 4181 | 4056 | 0 | 125 | v2 shared 모델과 비교하기 위한 작은 all-channel baseline |
| `channel_shared_1d_encoder` | 39317 | 2064 | 0 | 37253 | 기존 flatten-head shared 모델 |
| `channel_shared_meanpool_v2` | 2677 | 2064 | 0 | 613 | shared encoder 1개, mean pooling, small head |
| `channel_shared_attentionpool_v2` | 2950 | 2064 | 273 | 613 | shared encoder 1개, small attention pooling |
| `modality_shared_acc_gyro_encoder` | 41381 | 4128 | 0 | 37253 | 기존 modality shared flatten-head 모델 |
| `modality_shared_meanpool_v2` | 6373 | 4128 | 0 | 2245 | acc/gyro separate shared encoders, modality-wise pooling |
| `cnn2d_baseline_v1` | 8421 | 8256 | 0 | 165 | 기존 2D time-channel baseline |

`channel_shared_meanpool_v2`는 기존 `channel_shared_1d_encoder`보다 36,640 parameters 감소했다. 감소율은 약 93.19%이다. 또한 `all_channel_conv1d_v1`보다 13,360 parameters 작고, total parameter 기준 약 16.69% 수준이다.

`channel_shared_attentionpool_v2`도 기존 shared 모델보다 36,367 parameters 작고, attention scorer를 포함해도 `all_channel_conv1d_v1`보다 충분히 작다.

## 논문에서 사용할 수 있는 안전한 해석

안전한 표현은 다음과 같다.

- v1 shared 모델은 encoder weight sharing을 만족했지만, flatten aggregation head 때문에 total parameter count가 baseline보다 컸다.
- v2 shared 모델은 channel embedding pooling을 사용해 shared encoder의 parameter-reduction 의도를 total model capacity에도 반영했다.
- `channel_shared_meanpool_v2`는 all-channel Conv1D v1보다 훨씬 작은 parameter budget에서 성능을 비교할 수 있다.
- `all_channel_conv1d_small`은 작은 all-channel baseline으로, parameter-reduced shared 모델의 이점이 단순히 작은 모델 효과인지 구분하기 위한 비교군이다.
- `modality_shared_meanpool_v2`는 acc/gyro 물리량 차이를 보존하면서 modality 내부 channel 공유를 강제하는 중간 수준 capacity 모델이다.

주의할 점은 parameter count 감소가 곧 성능 향상을 보장하지 않는다는 것이다. 실제 주장은 LOSO real-data smoke training과 이후 full 실험에서 검증해야 한다.

## 다음 단계: CAU Real-data Smoke Training 계획

아직 실행하지 않는다. 사용자 승인 후 CAU 서버에서 다음 수준의 smoke training만 먼저 수행한다.

```bash
ssh CAU 'cd /home/user3/workspace/Jae/<project-path>/squat_imu_experiments && python -m unittest discover -s tests -v'
ssh CAU 'cd /home/user3/workspace/Jae/<project-path>/squat_imu_experiments && python scripts/audit_model_capacity.py --config configs/model_capacity_v2.yaml'
```

그 다음 승인된 별도 smoke-training script를 만들 경우에도 제한은 다음과 같다.

- full LOSO 금지
- hyperparameter tuning 금지
- SSL 금지
- external dataset 금지
- optimizer step 수와 epoch 수를 명시적으로 제한
- scaler fit은 train subject에만 수행
- held-out subject는 scaler, augmentation, validation tuning에 사용 금지

현재 단계에서는 forward pass와 parameter audit만 완료했다.
