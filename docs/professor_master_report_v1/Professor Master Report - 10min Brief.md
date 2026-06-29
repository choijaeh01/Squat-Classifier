---
title: "Professor Master Report - 10min Brief"
project: "Squat IMU"
date: "2026-06-29"
status: "brief"
tags:
  - research
  - squat
  - imu
  - kiee
---

# Professor Master Report - 10min Brief

## 1. Why We Changed Scope

- 기존 졸업논문은 IMU+Vision+on-device system 중심이었다.
- KIEE 논문은 IMU-only supervised feature extractor comparison으로 좁히는 것이 안전하다.
- hardware/data collection은 dataset background로 두고, main contribution은 architecture/protocol로 둔다.

## 2. Dataset/Protocol

- 3 MPU-6050 IMU, 18 channels, 512-step windows.
- 6 subjects, 5 classes, 600 windows.
- LOSO: train 400, validation 100, test 100 per fold.
- StandardScaler는 train indices에만 fit한다.

## 3. Common Head Comparison

- 모든 controlled neural model은 64-dim representation과 같은 4,485-param MLP head를 사용했다.
- 따라서 비교 초점은 classifier가 아니라 feature extractor다.
- per-window z-score, augmentation, focal loss, SSL, external transfer는 사용하지 않았다.

## 4. Architecture Comparison

- All-Channel 1D CNN: 18채널을 처음부터 함께 본다.
- Shared 1D Encoder: 같은 encoder를 18개 channel에 재사용하지만 channel cue가 약해질 수 있다.
- Shared 1D + Identity: token origin을 알려준다.
- Residual Channel-Shared Feature Extractor: shared temporal branch와 mean/std/min/max residual statistics를 결합한다.

## 5. Main Results

- Statistical Summary MLP: Macro F1 0.8174.
- Residual Channel-Shared Feature Extractor: 0.8004.
- All-Channel 1D CNN: 0.7994.
- XGBoost: 0.7961, Random Forest: 0.7845.
- Shared 1D Encoder: 0.2806.

## 6. Residual vs Identity

- identity without residual: +0.2376.
- identity with residual: -0.0031.
- residual without identity: +0.5198.
- residual with identity: +0.2791.
- 해석: identity는 이름표, residual은 signal statistics 자체다.

## 7. XGBoost/RF/Stats Baselines

- summary statistics 기반 baseline이 강하다.
- RF/XGBoost feature importance에서는 s1_ax/s1_gx 관련 feature가 반복 관찰된다.
- 그러나 feature importance는 인과 증거가 아니다.

## 8. Safe Claim

- residual branch는 naive shared encoder 병목을 크게 완화했다.
- proposed extractor는 All-Channel CNN, XGBoost, RF와 경쟁 가능하다.
- 모든 baseline보다 통계적으로 우수하다는 주장은 피한다.
- identity와 attention은 core claim으로 두지 않는다.

## 9. Questions for Professor

- 제안 이름을 Residual Channel-Shared Feature Extractor로 확정할지?
- Statistical Summary MLP가 1위인 점을 main table에 둘지 practical baseline으로 분리할지?
- RF/XGBoost를 main comparison에 포함할지?
- identity/position encoding은 future work로 낮춰도 되는지?
- KIEE 범위를 supervised IMU-only classification으로 확정할지?
