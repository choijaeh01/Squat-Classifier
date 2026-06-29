# Paper Discussion Draft v2

## 핵심 논의 방향

본 연구의 주요 결과는 position-aware channel-shared encoder가 naive shared encoder의 underfitting 문제를 완화하고, all-channel Conv1D 및 강한 feature baseline과 경쟁 가능한 성능을 보였다는 점이다. 다만 paired confidence interval이 0을 포함하는 비교가 있으므로 통계적 우월성보다는 경쟁 가능성과 구조적 해석을 중심으로 서술한다.

## Shared Encoder 해석

초기 shared v2 모델은 parameter 수만 줄인 구조로는 충분하지 않았다. Mean pooling 또는 단순 attention pooling은 channel identity와 sensor 위치 정보를 약화시켜 Class 3 Excessive Lean 같은 어려운 class에서 낮은 성능을 보였다.

v3 구조는 shared encoder를 유지하면서 channel ID, sensor ID, modality ID, axis ID와 residual summary branch를 추가했다. Overfit diagnostic과 full supervised matrix에서 v3는 학습 가능성과 일반화 가능성을 모두 보였다. Component ablation과 controlled comparison을 함께 보면 residual branch가 가장 큰 성능 회복 요인으로 관찰된다.

## Feature Baseline 해석

Controlled feature extractor comparison과 XGBoost completion은 summary feature 기반 baseline이 매우 강하다는 점을 보여준다. `controlled_stats_mlp`, `feature_xgboost_v1`, `feature_random_forest_v1`은 모두 높은 Macro F1을 기록했다.

이 결과는 neural proposed model의 의미를 부정하기보다, 현재 IMU squat dataset에서 per-channel summary statistics가 큰 discriminative information을 가진다는 것을 보여준다. 따라서 논문에서는 feature baseline을 숨기지 않고 투명하게 보고해야 한다.

## XGBoost Completion 관찰

별도 completion run에서 `feature_xgboost_v1`은 Macro F1 0.7961, accuracy 0.8156, weighted F1 0.7961을 기록했다. RF 대비 paired delta는 +0.0116이지만 CI [-0.0145, 0.0398]가 0을 포함했다. `controlled_stats_mlp`, residual shared, all-channel controlled CNN과의 비교도 모두 CI가 0을 포함했다.

따라서 XGBoost가 특정 baseline보다 확실히 우월하다고 주장하지 않는다. 대신 같은 signal-derived feature set을 사용하는 tree-based baseline도 강하다는 관찰로 정리한다.

## Class 3 Excessive Lean

Class 3은 pilot 단계부터 어려운 class였다. Full supervised matrix에서는 proposed v3가 Class 3 F1에서 강한 편이었고, controlled comparison에서는 residual shared 계열이 높은 Class 3 recall/F1을 보였다. XGBoost completion의 Class 3 F1은 0.6614로, precision은 높지만 recall이 제한적이었다.

이 관찰은 residual/position-aware neural representation이 일부 어려운 posture error에서 feature baseline과 다른 장점을 가질 수 있음을 시사한다. 단, class-wise 차이는 subject 수가 작기 때문에 과도하게 일반화하지 않는다.

## 주장 수위

안전한 서술:

> The proposed position-aware channel-shared encoder achieved competitive performance under subject-independent LOSO evaluation and mitigated the underfitting observed in naive shared encoders.

피해야 할 서술:

> The proposed model significantly outperforms all classical and neural baselines.

현재 결과에서는 stats MLP, XGBoost, RF가 강하고 여러 paired CI가 0을 포함하므로, 논문 주장은 “우월성”보다 “작은 IMU dataset에서 shared encoder를 안정화하기 위한 구조적 설계와 경쟁 가능한 성능”에 맞추는 편이 안전하다.
