# KIEE Paper Outline Draft

## 1. 서론

- 스쿼트 자세 오류 분류의 필요성
- IMU-only 접근의 장점
- Small-subject dataset에서 overfitting과 channel identity 문제 제기
- 본 논문의 기여 요약

## 2. 관련 연구

- IMU 기반 운동 자세 분석
- Squat posture classification
- 1D CNN 및 time-series sensor classification
- Channel-shared encoder 및 parameter-efficient modeling

## 3. 데이터셋 및 전처리

- 3개 MPU-6050 IMU
- Sensor locations: lower back, right thigh, right calf
- 18 channels
- 5 classes
- Manually labeled boundaries
- 512-length resampling
- Conversion 단계에서 normalization 없음

## 4. 제안 방법

### Position-aware channel-shared encoder

18개 channel에 동일한 temporal encoder를 공유 적용한다.

### Channel identity embedding

각 channel token에 channel ID를 제공해 단순 공유 구조의 channel identity 손실을 줄인다.

### Sensor/modality/axis embedding

Sensor 위치, accelerometer/gyroscope modality, x/y/z axis 정보를 token에 추가한다.

### Residual branch

Raw channel summary statistics를 이용해 channel-specific 정보를 보존한다.

### Attention aggregation

Mean pooling 대신 small attention pooling으로 channel token importance를 학습한다.

## 5. 실험 설계

- Subject-independent LOSO
- Within-train-subject stratified validation
- 3 seeds
- 7 models
- 126 runs
- Train-only scaler fitting
- No augmentation, no focal loss, no SSL, no external dataset

## 6. 실험 결과

- Main model comparison table
- Parameter count comparison
- Class-wise recall heatmap
- Class 3 Excessive Lean analysis
- Paired difference analysis
- Confusion matrix comparison

## 7. 논의

- v3가 가장 높은 평균 Macro F1을 기록
- all-channel baselines 대비 통계적 우월성은 조심해야 함
- v2 shared 모델의 낮은 결과는 channel identity bottleneck 가능성을 시사
- Class 3에서 sensor-position-specific information 중요성 논의

## 8. 결론

- Clean-room supervised LOSO protocol 확립
- Position-aware shared encoder 제안
- Small-subject IMU squat classification에서 경쟁 가능한 결과 확인
- Transfer learning 및 SSL은 후속 연구로 남김

## 논문 기여점 초안

- 3개 IMU 기반 스쿼트 자세 오류 데이터셋을 clean-room protocol로 재구성하고 LOSO 기반 평가를 수행하였다.
- naive channel-shared encoder가 위치 정보를 잃어 underfit될 수 있음을 보였다.
- channel/sensor/modality/axis identity와 residual branch를 결합한 position-aware shared encoder를 제안하였다.
- 제안 모델은 full supervised matrix에서 가장 높은 평균 Macro F1을 기록했고, all-channel Conv1D와 경쟁 가능한 성능을 보였다.
- Class 3 Excessive Lean과 같은 어려운 클래스에서의 오분류 양상을 분석하였다.
