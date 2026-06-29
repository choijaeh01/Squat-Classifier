# Paper Claim Audit v2

## 범위

이 문서는 기존 `paper_claim_audit_v1`에 controlled feature extractor comparison 결과를 반영한 v2 claim audit이다. 기존 locked full supervised matrix 결과는 유지하며, 이번 controlled comparison은 extractor 구조와 feature baseline 해석을 보강하기 위한 추가 실험이다.

## 안전하게 주장 가능한 내용

1. 동일한 final LOSO validation policy와 train-only StandardScaler 조건에서 clean-room supervised 실험을 수행했다.
2. 기존 locked full matrix에서 `channel_shared_posres_attention_v3`는 가장 높은 평균 Macro F1을 기록했다.
3. naive channel-shared 구조는 낮은 성능을 보였고, residual 또는 position-aware 정보를 추가한 shared 구조는 큰 폭으로 개선되었다.
4. controlled comparison에서 common 64-dim MLP head를 고정했을 때도 residual shared 1D extractor는 all-channel controlled CNN과 유사한 Macro F1을 보였다.
5. controlled comparison에서 residual branch 추가는 naive shared 1D 대비 가장 큰 성능 회복과 관련되어 관찰되었다.
6. handcrafted feature 기반 RandomForest, XGBoost, stats MLP는 현재 dataset에서 강한 baseline으로 나타났다.
7. Class 3 Excessive Lean에서는 residual shared 계열이 높은 recall/F1을 보였다.

## 통계적으로 조심해야 하는 내용

1. `channel_shared_posres_attention_v3`가 all-channel Conv1D보다 통계적으로 우월하다고 단정하면 안 된다. paired CI가 0을 포함했다.
2. controlled comparison에서도 residual shared 계열과 all-channel controlled CNN의 paired difference CI가 0을 포함했다.
3. `controlled_stats_mlp`가 가장 높은 Macro F1을 보였지만, 이것을 모든 IMU squat dataset에서 summary statistics가 deep model보다 우월하다는 일반 결론으로 확장하면 안 된다.
4. RandomForest 상위 feature가 `s1_ax`에 집중된 것은 현재 dataset/protocol에서의 관찰이며, 생체역학적 원인 단정은 추가 검증이 필요하다.
5. Lee-style CNN-LSTM 결과는 adapted clean-room baseline 결과이지 Lee et al. exact reproduction이 아니다.

## 주장하면 안 되는 내용

1. proposed v3가 all baseline보다 통계적으로 유의하게 우월하다는 주장.
2. transfer learning 효과에 대한 주장. 이번 단계에서도 external dataset adapter와 SSL은 수행하지 않았다.
3. XGBoost와의 명확한 성능 우열. 별도 completion에서 XGBoost는 실행되었지만, RF, stats MLP, residual shared, all-channel CNN과의 paired CI가 모두 0을 포함했다.
4. preprocessing, split, scaler policy를 바꾸면 성능이 더 좋아진다는 주장.
5. RandomForest feature importance를 인과적 설명으로 해석하는 주장.

## 리뷰어가 물어볼 가능성이 높은 질문과 답변 초안

### Q1. 왜 feature baseline이 이렇게 강한가?

현재 데이터는 5-class squat posture이고 각 window가 512-step으로 정렬되어 있다. per-channel summary statistics, 특히 오른쪽 허벅지 `s1_ax` 관련 분산/에너지/RMS/최대값이 class 구분 정보를 많이 담는 것으로 관찰되었다. 따라서 강한 classical feature baseline을 포함해 neural model 성능을 과장하지 않도록 보고했다.

XGBoost completion에서도 같은 feature set으로 Macro F1 0.7961을 기록했고, 상위 feature는 `s1_ax_std`, `s1_gx_mean`, `s1_ax_energy`였다. 이는 feature baseline이 특정 구현 하나에만 의존하지 않는다는 보조 관찰이지만, feature importance를 인과 설명으로 단정하지 않는다.

### Q2. proposed shared encoder가 정말 필요한가?

locked full matrix에서는 proposed v3가 가장 높은 평균 Macro F1을 기록했다. controlled comparison에서는 residual shared 1D가 all-channel controlled CNN과 유사한 성능을 보였고, naive shared 1D 대비 크게 개선되었다. 따라서 shared 구조 자체보다 위치/잔차 정보 보존이 핵심이라는 보수적 해석이 적절하다.

### Q3. identity embedding이 꼭 필요한가?

controlled comparison에서는 residual-only와 residual+identity의 차이가 작았다. 하지만 locked v3는 attention, identity, residual이 결합된 구조이며, component ablation에서도 attention/meanpool/residual-only 변형과 함께 비교되어야 한다. 따라서 identity embedding의 독립 기여는 제한적으로 서술한다.

### Q4. Lee-style CNN-LSTM이 낮은 이유는 무엇인가?

이번 모델은 Lee et al. exact reproduction이 아니라 현재 18채널, 512-step, 5-class LOSO protocol에 맞춘 adapted clean-room baseline이다. 내부 40-step downsampling과 2D CNN-LSTM 구조가 현재 데이터의 class-discriminative summary signal을 충분히 활용하지 못했을 수 있다. 결과를 보고 구조나 hyperparameter는 바꾸지 않았다.

### Q5. Class 3 Excessive Lean에서 제안 모델은 어떤 의미가 있는가?

Class 3은 초기 pilot에서 어려운 class로 확인되었다. controlled comparison에서는 residual shared 계열이 Class 3 recall/F1에서 높은 값을 보였다. 이는 position-aware/residual 정보 보존이 어려운 자세 오류 구분에 도움이 될 가능성을 보여주는 관찰이다.

## v3를 proposed model로 둘 때의 장점

- locked full matrix에서 평균 Macro F1 0.8108로 가장 높았다.
- parameter count가 작고 all-channel baseline과 경쟁 가능한 수준이다.
- naive shared underfit 문제를 position-aware embedding, residual branch, attention aggregation으로 해결하려는 명확한 설계 논리가 있다.
- controlled comparison에서 residual shared 계열이 all-channel controlled CNN과 유사하게 작동했다.

## v3를 proposed model로 둘 때의 위험

- controlled comparison에서는 `controlled_stats_mlp`가 더 높은 Macro F1을 보였다.
- XGBoost completion까지 포함하면 feature 기반 baseline이 neural shared residual 계열과 매우 가까운 수준으로 관찰된다.
- feature baseline이 강하므로 neural proposed model만으로 논문 novelty를 밀기 어렵다.
- identity embedding의 독립 기여는 controlled comparison에서 명확하지 않았다.
- v3가 all-channel Conv1D보다 통계적으로 유의하게 우월하다고 말하기 어렵다.

## 추천 서술 방향

`proposed v3 is the best-performing neural architecture in the locked supervised matrix and remains competitive with strong all-channel and feature baselines` 정도가 안전하다. `v3 significantly outperforms all baselines`는 피한다.

국문으로는 다음처럼 쓸 수 있다.

> 제안한 position-aware channel-shared encoder는 naive shared encoder의 underfitting 문제를 크게 완화했으며, 최종 supervised matrix에서 all-channel Conv1D 및 강한 feature baseline과 경쟁 가능한 성능을 보였다.

## 추가 ablation 필요성

이미 v3 component ablation과 controlled feature extractor comparison이 있다. 현재 단계에서 즉시 추가 training을 늘리기보다, 논문 초안을 작성한 뒤 교수님 피드백에 따라 최소 추가 실험을 정하는 편이 낫다.

필요할 수 있는 최소 후속 실험 후보:

- stats MLP, RandomForest, XGBoost를 논문 main table에 포함할지 검토
- residual branch without temporal encoder 해석 보강
- feature importance의 sensor/channel별 집계 figure 정리

이 문서는 최종 학술 판단을 대신하지 않는다.
