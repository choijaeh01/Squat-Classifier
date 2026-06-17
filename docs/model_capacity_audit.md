# Model Capacity Audit

## 목적

현재 synthetic smoke test에서 `channel_shared_1d_encoder`와 `modality_shared_acc_gyro_encoder`의 전체 parameter count가 `all_channel_conv1d`보다 크다. 이는 "shared encoder가 작은 target dataset에서 파라미터 자유도와 과적합 위험을 줄일 수 있다"는 논문 가설과 겉으로 충돌한다. 이 문서는 현재 scaffold 기준으로 그 원인을 분해한다.

## Parameter Count

| model | total parameters | encoder parameters | head/classifier parameters | 비고 |
|---|---:|---:|---:|---|
| `classical_feature_baseline` | 4997 | 0 | 4997 | 통계 feature 추출은 parameter 없음 |
| `all_channel_conv1d` | 16037 | 15712 | 325 | 대부분이 all-channel Conv1D encoder |
| `channel_shared_1d_encoder` | 39317 | 2064 | 37253 | shared encoder는 작지만 flattened channel head가 큼 |
| `modality_shared_acc_gyro_encoder` | 41381 | 4128 | 37253 | acc encoder 2064 + gyro encoder 2064 |
| `cnn2d_baseline` | 8421 | 8256 | 165 | 2D conv encoder + 작은 classifier |

## Shared Encoder 확인

현재 구현에서 `channel_shared_1d_encoder`는 하나의 `shared_encoder` 객체를 모든 channel에 재사용한다. object identity 확인 결과 모든 `channel_encoder_refs`가 같은 `shared_encoder`를 가리킨다.

`modality_shared_acc_gyro_encoder`는 accelerometer용 `acc_encoder` 하나와 gyroscope용 `gyro_encoder` 하나를 사용한다. accelerometer reference는 모두 같은 `acc_encoder`를 가리키고, gyroscope reference는 모두 같은 `gyro_encoder`를 가리킨다. 두 encoder 객체는 서로 다르다.

따라서 parameter count 증가 원인은 encoder 공유 실패가 아니다.

## Parameter Count가 커진 원인

현재 shared encoder 계열은 각 channel을 같은 temporal encoder에 통과시킨 뒤 channel별 embedding을 모두 flatten한다.

- channel 수: 18
- channel별 embedding dimension: 32
- classifier 입력 dimension: 18 x 32 = 576
- 첫 head layer: `Linear(576, 64)`
- 이 layer만 `576 x 64 + 64 = 36928` parameters
- 최종 `Linear(64, 5)`는 325 parameters
- head 총합은 37253 parameters

즉, shared encoder 자체는 2064 parameters로 작다. 그러나 channel별 embedding을 모두 연결하는 aggregation head가 `all_channel_conv1d`보다 훨씬 크다.

현재 scaffold에는 GRU나 attention layer가 없다. parameter 증가 원인은 GRU, attention이 아니라 projection dimension과 flattened aggregation head/classifier이다.

## 가설과 현재 구현의 불일치

논문 가설은 shared encoder가 채널별로 별도 temporal filter를 학습하지 않게 하여 자유도를 줄일 수 있다는 것이다. 하지만 현재 구현은 encoder는 공유하면서도, channel embedding을 큰 dense head에 모두 넘기기 때문에 전체 model capacity는 오히려 커진다.

따라서 현재 smoke-test parameter count만으로는 shared encoder의 regularization 효과를 주장하기 어렵다. 논문 실험에서는 capacity-matched 비교가 필요하다.

## Capacity-matched 비교 제안

아직 모델 구조는 수정하지 않는다. 다음 단계에서 승인 후 다음 수정안을 검토한다.

1. shared encoder embedding dimension 축소
   - 예: 32에서 8 또는 16으로 축소
   - head 입력 차원이 18 x embedding_dim으로 줄어든다.

2. flatten 대신 channel pooling 사용
   - channel embeddings를 mean/max pooling하여 fixed embedding으로 줄인다.
   - 예: `(batch, 18, dim)`을 `(batch, dim)`으로 평균 내면 head가 크게 작아진다.

3. bottleneck aggregation 추가
   - channel embedding을 작은 shared projection으로 줄인 뒤 classifier에 넘긴다.
   - 단, projection 자체가 불필요하게 커지지 않도록 parameter budget을 계산해야 한다.

4. all-channel baseline과 matched budget 설정
   - `all_channel_conv1d` total parameter 16037 근처로 shared 모델의 total parameter를 맞춘 variant를 추가한다.
   - 예: `channel_shared_1d_encoder_small`, `modality_shared_acc_gyro_encoder_small`처럼 별도 registry key로 두고 기존 scaffold는 보존한다.

5. encoder-only capacity와 total capacity를 모두 보고
   - shared encoder의 장점은 encoder parameter sharing에 있으므로, 논문 표에는 total parameter와 encoder/head split을 함께 제시한다.

## 현재 결론

현재 shared encoder 모델은 encoder sharing 조건은 만족하지만 capacity-matched 비교 모델은 아니다. 가장 큰 문제는 18개 channel embedding을 flatten한 뒤 큰 dense classifier를 붙인 head 설계이다. 논문 가설을 검증하려면 shared encoder 모델의 aggregation head를 줄이거나, all-channel baseline과 비슷한 total parameter count가 되도록 별도 small variant를 추가해야 한다.
