# Controlled Feature Extractor Interpretation Notes For User

이 문서는 최종 해석을 대신하지 않고, 사용자가 논문 서사를 판단할 때 필요한 수치 중심 메모만 정리한다.

## 질문별 메모

### 1. 공통 head 통제는 제대로 되었는가?

그렇다. controlled neural 9개 모델은 모두 64차원 representation을 만들고, 동일한 MLP head를 사용했다.

- head params: 4,485
- head signature: `linear:64->64`, `relu`, `dropout:0.1`, `linear:64->5`

따라서 controlled neural 간 비교에서는 classifier head 차이보다 extractor 차이에 초점을 둘 수 있다.

### 2. 단순 raw flatten MLP는 강한가?

`controlled_flatten_mlp`는 594,373 params로 가장 큰 모델이지만 Macro F1은 0.7046이었다. parameter 수가 매우 많아도 가장 좋은 성능을 보이지 않았다. 작은 subject 수에서 대형 flatten projection은 안정적인 inductive bias가 아닐 수 있다.

### 3. summary statistics만으로도 강한가?

`controlled_stats_mlp`가 Macro F1 0.8174로 이번 controlled comparison에서 가장 높았다. 이는 현재 데이터의 class 구분에 per-channel mean/std/min/max 같은 요약 통계가 매우 강한 정보를 담고 있음을 시사한다.

이 결과는 proposed neural model claim을 약화시키기보다, 논문에서 반드시 강한 feature baseline을 보고해야 한다는 근거가 된다.

### 4. naive shared 1D는 왜 낮은가?

`controlled_shared_1d`는 Macro F1 0.2806이었다. 같은 head 조건에서도 shared encoder 단독 mean pooling 구조는 위치별/채널별 정보를 충분히 보존하지 못한 것으로 보인다.

### 5. identity embedding은 충분한가?

`controlled_shared_1d_identity`는 Macro F1 0.5182로 naive shared보다 높지만, residual shared 계열에는 크게 못 미쳤다. identity embedding만으로는 raw signal의 위치별 차이를 충분히 회복하지 못한 것으로 관찰된다.

### 6. residual branch가 중요한가?

가장 큰 component 변화는 residual branch 추가였다.

- `controlled_shared_1d`: 0.2806
- `controlled_shared_1d_residual`: 0.8004
- delta: +0.5198

이번 controlled run에서는 residual branch가 shared encoder bottleneck을 완화하는 핵심 요인으로 관찰된다.

### 7. residual + identity가 residual-only보다 좋은가?

`controlled_shared_1d_residual_identity`는 0.7973, `controlled_shared_1d_residual`은 0.8004였다. 차이는 매우 작고 방향도 residual-only가 약간 높다. 따라서 controlled setting만 보면 identity embedding의 추가 이득은 명확하지 않다.

다만 locked proposed v3는 attention, identity, residual이 결합된 별도 구조이므로, 이 결과만으로 identity embedding이 불필요하다고 단정하지 않는다.

### 8. all-channel controlled CNN과 shared residual은 비슷한가?

비슷하게 관찰된다.

- `controlled_all_channel_1d_cnn`: Macro F1 0.7994
- `controlled_shared_1d_residual`: Macro F1 0.8004
- paired delta, all-channel minus shared residual: -0.0010, CI [-0.0752, 0.0688]

이 결과는 residual이 있는 channel-shared extractor가 all-channel 1D CNN과 경쟁 가능한 수준에 도달했다는 서사를 뒷받침할 수 있다. 단, 통계적 우월성 주장은 하지 않는다.

### 9. RandomForest는 계속 강한가?

RandomForest는 Macro F1 0.7845로 강했다. controlled shared residual 계열과 큰 차이는 아니지만, feature summary baseline이 강하다는 점은 논문에서 투명하게 보고해야 한다.

상위 feature는 `s1_ax` 관련 통계가 많았다. 이는 오른쪽 허벅지 전후/방향 가속도 특성이 squat posture class 구분에 유용했을 가능성을 보여준다.

### 9-1. XGBoost completion 결과는 RF와 비교해 어떤가?

별도 completion으로 실행한 `feature_xgboost_v1`은 Macro F1 0.7961이었다. RandomForest는 0.7845였고, paired delta는 XGBoost minus RandomForest 기준 +0.0116, CI [-0.0145, 0.0398]이었다.

따라서 수치상 XGBoost가 약간 높지만 CI가 0을 포함하므로 RF보다 명확히 우월하다고 해석하지 않는다. 두 모델 모두 동일한 signal-derived summary feature set을 사용했다.

XGBoost 상위 feature는 `s1_ax_std`, `s1_gx_mean`, `s1_ax_energy`였고, RF와 마찬가지로 오른쪽 허벅지 `s1` 관련 feature가 중요하게 나타났다. Feature importance는 모델 내부 기준이므로 생체역학적 인과 설명으로 직접 사용하지 않는다.

### 10. Lee-style CNN-LSTM은 강한가?

`lee_style_cnn_lstm_2d_v1`는 Macro F1 0.6269였다. `controlled_shared_1d_residual_identity` 대비 paired delta는 -0.1704, CI [-0.2936, -0.0592]로 낮은 방향이 비교적 명확했다.

이 결과는 Lee-style adapted baseline이 현재 dataset/protocol에서는 strong baseline이 아니었다는 관찰로 정리할 수 있다. 단, Lee et al. exact reproduction이 아니라 adapted clean-room baseline임을 반드시 명시해야 한다.

### 11. Class 3 Excessive Lean은 누가 강한가?

Class 3 F1은 residual shared 계열이 가장 높았다.

- `controlled_shared_1d_residual`: F1 0.7542
- `controlled_shared_1d_residual_identity`: F1 0.7509
- `controlled_stats_mlp`: F1 0.7150
- `feature_random_forest_v1`: F1 0.6848
- `controlled_all_channel_1d_cnn`: F1 0.6692

이는 어려운 class에서 residual shared 구조가 의미 있는 관찰을 제공할 수 있음을 보여준다.

## 사용자가 판단해야 할 지점

- 논문 main story를 proposed v3 중심으로 유지할지, 강한 summary/statistics baseline을 더 전면에 둘지.
- `controlled_stats_mlp`를 main result table에 포함할지, ablation/control analysis table로 분리할지.
- residual branch의 역할을 주요 claim으로 강화할지.
- identity embedding의 역할을 보수적으로 서술할지.
- RandomForest와 stats MLP가 강한 이유를 dataset 특성 분석으로 확장할지.

## Codex가 하지 않은 것

- 성능을 보고 hyperparameter를 바꾸지 않았다.
- model width, learning rate, split, preprocessing을 바꾸지 않았다.
- 시스템 Python, `sudo`, `pip install --user` 방식으로 XGBoost를 설치하지 않았다.
- `xgboost_only_completion_v1`에서는 CAU project-local `.deps/xgboost`에만 XGBoost를 설치해 별도 실행했다.
- 기존 locked result 디렉터리를 수정하지 않았다.
- 결과를 보고 추가 재실행이나 tuning run을 수행하지 않았다.
