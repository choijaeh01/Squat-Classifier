# Figure Captions

## fig_01_feature_extractor_macro_f1.png

- Korean caption: 공통 classifier head 조건에서 feature extractor별 Macro F1을 비교한 그림.
- English caption: Macro F1 comparison across feature extractors under a shared classifier head.
- 핵심 메시지: Statistical Summary MLP가 가장 높고, Residual Channel-Shared Encoder는 All-Channel 1D CNN 및 XGBoost와 유사한 수준이다.
- 논문 사용 추천: main 또는 ablation/control figure 후보.

## fig_02_residual_effect_flow.png

- Korean caption: Shared 1D Encoder에서 identity와 residual branch를 추가했을 때 Macro F1 변화.
- English caption: Macro F1 progression from shared 1D encoder to residual channel-shared variants.
- 핵심 메시지: identity만으로는 부족했고 residual branch가 가장 큰 성능 회복을 만들었다.
- 논문 사용 추천: proposed structure motivation figure 후보.

## fig_03_xgboost_practical_baselines.png

- Korean caption: XGBoost completion을 포함한 practical/neural baseline Macro F1 비교.
- English caption: Macro F1 comparison including XGBoost completion and practical baselines.
- 핵심 메시지: XGBoost와 RF도 강하므로 practical baseline을 투명하게 보고해야 한다.
- 논문 사용 추천: baseline comparison 또는 appendix 후보.

## fig_04_class3_excessive_lean.png

- Korean caption: Class 3 Excessive Lean에서 주요 모델의 recall 및 F1 비교.
- English caption: Class 3 Excessive Lean recall and F1 across key models.
- 핵심 메시지: Residual Channel-Shared 계열이 어려운 class에서 비교적 높은 F1을 보였다.
- 논문 사용 추천: class-wise analysis 후보.

## fig_05_rf_xgboost_feature_importance.png

- Korean caption: RF와 XGBoost의 상위 signal-derived feature importance 비교.
- English caption: Top signal-derived feature importances from Random Forest and XGBoost.
- 핵심 메시지: s1_ax, s1_gx 관련 feature가 반복적으로 관찰되지만 인과 단정은 금지한다.
- 논문 사용 추천: discussion 또는 appendix 후보.

## fig_06_subjectwise_macro_f1.png

- Korean caption: 주요 모델의 subject-wise Macro F1 비교.
- English caption: Subject-wise Macro F1 comparison for key models.
- 핵심 메시지: 평균 성능만이 아니라 subject별 변동을 함께 봐야 한다.
- 논문 사용 추천: appendix 후보.

## fig_07_parameter_vs_macro_f1.png

- Korean caption: controlled neural model의 parameter count와 Macro F1 관계.
- English caption: Parameter count versus Macro F1 for controlled neural models.
- 핵심 메시지: parameter 수가 많다고 자동으로 좋은 성능을 보장하지 않는다.
- 논문 사용 추천: supplementary 또는 discussion 후보.
