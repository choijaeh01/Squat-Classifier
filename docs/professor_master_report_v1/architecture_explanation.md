# Architecture Explanation

이 문서는 교수님 질문에 대비해 master report의 architecture 설명을 조금 더 기술적으로 풀어쓴 노트다. 제안 이름은 `Residual Channel-Shared Feature Extractor`로 둔다. `Position-Aware`는 핵심 제목이 아니라 identity variant 또는 future work 맥락에서만 사용한다.

## 1. Notation

Let $X \in \mathbb{R}^{T \times C}$, where $T=512$ and $C=18$. A sample contains 3 IMUs and each IMU contributes accelerometer and gyroscope axes.

## 2. All-Channel 1D CNN

All-Channel 1D CNN은 입력 $X$의 18개 채널을 처음부터 함께 본다. 따라서 channel identity와 inter-channel relationship을 convolution filter가 직접 학습할 수 있다.

핵심 구현 위치:
- feature extractor 구현 파일: `src/models/` 아래 controlled neural extractor module
- common head 구현 파일: `src/models/common_head.py`

## 3. Shared 1D Encoder

Shared encoder는 각 channel을 single-channel stream으로 분리한 뒤 같은 temporal encoder $f_\theta$를 재사용한다.

```text
h_c = f_theta(x_c)
h_shared = Pool({h_c}_{c=1}^{18})
```

장점은 parameter sharing이고, 약점은 pooling 이후 channel origin과 channel-specific summary cue가 약해질 수 있다는 점이다.

## 4. Identity Embedding

Identity embedding은 token에 channel/sensor/modality/axis 정보를 더해 token origin을 제공한다.

```text
h'_c = h_c + e_channel(c) + e_sensor(c) + e_modality(c) + e_axis(c)
```

이번 결과에서 identity는 residual branch가 없을 때 Shared 1D Encoder를 0.2806에서 0.5182로 개선했다. 하지만 residual branch가 들어간 뒤에는 추가 이득이 거의 없었다.

## 5. Residual Branch

Residual branch는 train-only StandardScaler로 transform된 tensor에서 channel-wise summary statistics를 계산한다.

```text
r = phi([mean_t(x_c), std_t(x_c), min_t(x_c), max_t(x_c)]_{c=1}^{18})
```

즉 18개 채널마다 mean/std/min/max 4개를 계산하므로 72개 signal-derived feature가 된다. metadata, label, subject_id, filename, boundary, original length는 사용하지 않는다.

핵심 구현 위치:
- residual statistics 함수: feature extractor module의 `compute_channel_summary`
- `src/models/classical_features.py`: RF/XGBoost/SVM용 162개 signal-derived feature

## 6. Residual Channel-Shared Feature Extractor

제안 구조는 shared temporal representation과 residual statistics representation을 결합한 뒤 64차원 representation으로 projection한다.

```text
z = Project([h_shared, r])
y_hat = g_common(z)
```

여기서 $g_{common}$은 controlled neural model이 공유하는 MLP classifier head다.

```text
64-dim representation -> Linear 64->64 -> ReLU -> Dropout 0.1 -> Linear 64->5
```

## 7. 왜 Statistical Summary MLP가 강한가

Statistical Summary MLP는 temporal pattern을 크게 학습하지 않고도 mean/std/min/max 72개 feature만으로 높은 Macro F1을 보였다. 이는 이 dataset에서 자세 오류 class가 channel-wise amplitude, variability, range 같은 summary statistics로 상당 부분 구분될 수 있음을 시사한다. 다만 이것은 모델 성능 관찰이지 생체역학적 인과 증명이 아니다.

## 8. 왜 XGBoost/RF가 강한 practical baseline인가

RF/XGBoost는 162개 signal-derived summary feature를 사용한다. feature audit 기준 metadata, subject ID, label, boundary, original length는 포함되지 않았다. 이 모델들이 강하다는 점은 proposed neural extractor의 claim을 과장하지 않도록 만드는 중요한 기준선이다.

## 9. Claim Position

안전한 주장은 다음이다.

- naive shared encoder는 channel-specific cue를 잃어 성능이 낮을 수 있다.
- residual branch는 shared encoder bottleneck을 크게 완화했다.
- Residual Channel-Shared Feature Extractor는 All-Channel 1D CNN, RF, XGBoost와 경쟁 가능한 성능을 보였다.
- 그러나 모든 baseline보다 통계적으로 우수하다고 주장하면 안 된다.
