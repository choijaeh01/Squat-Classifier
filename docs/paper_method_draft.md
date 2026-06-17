# Paper Method Draft

## Dataset Construction

본 연구는 3개의 MPU-6050 IMU를 사용한 squat posture dataset을 대상으로 한다. 센서 위치는 허리, 오른쪽 허벅지, 오른쪽 종아리이며, 각 IMU는 3축 accelerometer와 3축 gyroscope를 제공한다. 따라서 각 time step은 18개 channel로 구성된다.

Manually labeled boundary를 기준으로 raw session에서 squat window를 추출했고, 각 window는 phase-normalized linear interpolation으로 길이 512로 resampling했다. Conversion 단계에서는 z-score normalization, clipping, augmentation을 적용하지 않았다.

## Evaluation Protocol

평가는 subject-independent leave-one-subject-out cross validation으로 수행한다. 각 fold에서 한 명의 subject는 test set으로 완전히 제외한다. 나머지 5명 subject의 각 class 20 windows 중 16개는 train, 4개는 validation으로 사용한다. 각 fold의 크기는 train 400, validation 100, test 100이다.

StandardScaler는 fold별 train indices에만 fit하고 validation/test에는 transform만 적용한다. 이 정책은 scaler leakage audit으로 검증한다.

## Proposed Model

제안 모델은 `channel_shared_posres_attention_v3`이다. 이 모델은 18개 single-channel IMU signal에 동일한 1D temporal encoder object를 공유 적용한다.

각 channel token에는 다음 learnable identity embedding을 추가한다.

- channel ID embedding
- sensor ID embedding
- modality ID embedding
- axis ID embedding

이후 small attention pooling으로 channel tokens를 aggregate한다. 또한 raw channel summary statistics 기반 residual branch를 추가해 shared encoder가 놓칠 수 있는 channel-specific signal을 보존한다. 최종 representation은 attention branch와 residual branch를 concat한 뒤 classifier head로 전달한다.

## Baselines

비교 모델은 all-channel Conv1D, small all-channel Conv1D, 2D CNN baseline, naive channel-shared v2, channel-shared attention v2, modality-shared sensor-attention v3를 포함한다. 모든 모델은 동일한 final validation policy와 동일한 training configuration에서 평가한다.
