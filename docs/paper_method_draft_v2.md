# Paper Method Draft v2

## Dataset and Preprocessing

본 연구는 3개의 MPU-6050 IMU를 이용해 수집한 스쿼트 자세 데이터셋을 사용한다. 센서는 허리, 오른쪽 허벅지, 오른쪽 종아리에 부착되었으며, 각 센서는 3축 가속도와 3축 자이로스코프를 제공한다. 따라서 각 window는 총 18개 IMU channel로 구성된다.

원본 raw CSV와 수동 라벨링 metadata boundary를 이용해 각 squat repetition 구간을 추출하고, 각 variable-length window를 phase-normalized linear interpolation으로 512 time steps로 resampling하였다. Conversion 단계에서는 z-score normalization, clipping, augmentation을 적용하지 않았다. 최종 supervised input은 `X.npy` shape `(600, 512, 18)`, label은 `y.npy` shape `(600,)`, subject identifier는 `subject_id.npy` shape `(600,)`이다.

Class mapping은 다음과 같이 고정하였다.

| class id | label |
|---:|---|
| 0 | Correct |
| 1 | Knee Valgus |
| 2 | Butt Wink |
| 3 | Excessive Lean |
| 4 | Partial Squat |

## Evaluation Protocol

평가는 subject-independent LOSO protocol로 수행하였다. 각 fold에서 하나의 subject를 test subject로 완전히 제외하고, 나머지 5명의 subject에서 각 class별 20개 window 중 16개를 training, 4개를 validation으로 사용하였다. 따라서 각 fold의 sample 수는 train 400, validation 100, test 100이다.

StandardScaler는 각 fold의 training indices 400개에만 fit하였고, validation/test에는 transform만 적용하였다. Test subject는 scaler fitting, validation selection, early stopping, augmentation, SSL pretraining에 사용하지 않았다. 본 실험에서는 augmentation, focal loss, balanced sampling, mixup, Time-CutMix, SSL, external dataset transfer를 사용하지 않았다.

## Position-aware Channel-shared Encoder

제안 모델은 모든 단일 IMU channel에 동일한 1D encoder object를 재사용한다. 이 구조는 channel마다 별도의 encoder를 두는 all-channel model보다 일반적인 IMU temporal feature를 공유 학습하도록 유도한다. 단, naive shared encoder는 channel identity를 잃을 수 있으므로, channel/sensor/modality/axis identity embedding과 residual branch를 추가해 위치 정보를 보존한다.

모델의 핵심 구성은 다음과 같다.

1. 단일 channel time series를 shared 1D encoder에 입력한다.
2. 각 channel output을 token embedding으로 변환한다.
3. channel, sensor, modality, axis identity embedding을 추가한다.
4. attention pooling으로 channel token을 집계한다.
5. raw signal summary 기반 residual representation을 결합한다.
6. classifier head로 5-class logits를 출력한다.

## Controlled Feature Extractor Comparison

추가 분석으로, classifier head를 통제한 feature extractor comparison을 수행하였다. 모든 controlled neural model은 64차원 representation을 만들고, 같은 MLP head를 사용한다.

공통 head:

- Linear 64 to 64
- ReLU
- Dropout 0.1
- Linear 64 to 5
- parameter count: 4,485

이 실험에는 다음 feature extractor family가 포함되었다.

- raw flatten MLP
- per-channel summary statistics MLP
- all-channel 1D CNN
- small all-channel 1D CNN
- naive shared 1D encoder
- shared 1D encoder with identity embedding
- shared 1D encoder with residual branch
- shared 1D encoder with residual branch and identity embedding
- 2D CNN over time-channel matrix

Classical practical baselines로 RandomForest와 LinearSVM을 같은 split/scaler policy에서 평가하였다. Feature는 IMU signal에서만 계산했으며 metadata, subject ID, label, boundary, filename은 feature에 포함하지 않았다. XGBoost는 실행 환경에 설치되어 있지 않아 skipped 처리하였다.

## Metrics

주요 metric은 accuracy, Macro F1, Weighted F1이다. Class-wise precision, recall, F1과 subject-wise Macro F1, confusion matrix도 저장하였다. Bootstrap confidence interval과 paired model difference는 Macro F1을 primary metric으로 계산하였다.

Macro F1은 class imbalance가 크지 않은 현재 데이터에서도 class별 성능을 균등하게 반영하므로 주요 metric으로 사용하였다. 다만 paired difference의 CI가 0을 포함하는 경우 통계적 우월성은 주장하지 않는다.
