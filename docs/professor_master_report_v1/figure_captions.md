# Figure Captions

## fig_01_full_story_map.png
- Korean caption: 연구 범위가 학부 시스템 구현에서 clean-room IMU-only feature extractor 비교 논문으로 전환되는 전체 흐름.
- English caption: Full story map from the undergraduate system scope to the clean-room IMU-only feature extractor comparison.
- 핵심 메시지: 교수님께 보고 시작 시 전체 방향 전환을 설명한다.
- 논문 사용 추천: 본문 도입 그림 후보

## fig_02_dataset_and_loso_protocol.png
- Korean caption: 3개 IMU, 18채널, 600개 window와 LOSO train/validation/test 구조.
- English caption: Dataset and LOSO protocol with three IMUs, 18 channels, 600 windows, and train/validation/test sizes.
- 핵심 메시지: 데이터와 누수 통제의 기본 구조를 한 번에 보여준다.
- 논문 사용 추천: 본문 방법 그림 후보

## fig_03_normalization_pipeline.png
- Korean caption: raw boundary 기반 conversion 이후 train-only StandardScaler와 feature extraction까지의 순서.
- English caption: Processing order from raw-boundary conversion to train-only StandardScaler and feature extraction.
- 핵심 메시지: normalization이 conversion 단계가 아니라 fold 내부에서 수행됨을 명확히 한다.
- 논문 사용 추천: 본문 방법 그림 후보

## fig_04_common_head_and_feature_extractor_design.png
- Korean caption: controlled neural model에서 classifier head를 고정하고 feature extractor만 비교한 설계.
- English caption: Common-head design that fixes the classifier head and compares feature extractors.
- 핵심 메시지: 교수님 피드백인 classifier 앞 feature extractor 비교를 설명한다.
- 논문 사용 추천: 본문 방법 그림 후보

## fig_05_architecture_comparison_overview.png
- Korean caption: All-Channel 1D CNN, Shared 1D, Shared 1D + Identity, Residual Channel-Shared 구조 비교.
- English caption: Architecture comparison among All-Channel 1D CNN, Shared 1D, Shared 1D + Identity, and Residual Channel-Shared.
- 핵심 메시지: 어떤 정보가 보존되거나 약화되는지 설명한다.
- 논문 사용 추천: 본문 방법 그림 후보

## fig_06_residual_branch_detailed.png
- Korean caption: Residual branch가 train-scaled tensor에서 mean/std/min/max 72개 signal-derived feature를 계산하는 과정.
- English caption: Residual branch computing 72 mean/std/min/max signal-derived features from the train-scaled tensor.
- 핵심 메시지: metadata leakage 없이 residual branch가 무엇을 제공하는지 보여준다.
- 논문 사용 추천: 본문 방법 그림 후보

## fig_07_main_results_macro_f1.png
- Korean caption: 주요 모델의 Macro F1과 bootstrap CI 비교.
- English caption: Macro F1 with bootstrap confidence intervals for key models.
- 핵심 메시지: proposed core와 strong practical baselines가 근접함을 보여준다.
- 논문 사용 추천: 본문 결과 그림 후보

## fig_08_residual_identity_2x2.png
- Korean caption: Residual branch와 identity embedding의 2x2 interaction 결과.
- English caption: 2x2 interaction result between the residual branch and identity embeddings.
- 핵심 메시지: identity보다 residual branch가 핵심이라는 해석을 뒷받침한다.
- 논문 사용 추천: 본문 결과 그림 후보

## fig_09_effect_decomposition.png
- Korean caption: identity/residual effect와 bootstrap CI를 paired delta로 분해한 그림.
- English caption: Effect decomposition of identity and residual components using paired deltas and bootstrap CIs.
- 핵심 메시지: residual effect가 가장 크다는 정량 근거다.
- 논문 사용 추천: 본문 또는 appendix 후보

## fig_10_practical_baselines.png
- Korean caption: Statistical Summary MLP, Residual Channel-Shared, All-Channel 1D CNN, XGBoost, Random Forest 비교.
- English caption: Comparison among Statistical Summary MLP, Residual Channel-Shared, All-Channel 1D CNN, XGBoost, and Random Forest.
- 핵심 메시지: simple summary/tree baselines가 강함을 투명하게 보고한다.
- 논문 사용 추천: 본문 결과 그림 후보

## fig_11_feature_importance_rf_xgboost.png
- Korean caption: RF/XGBoost 상위 feature importance 비교. importance는 인과 증거가 아니다.
- English caption: Top RF/XGBoost feature importance. Importance is not causal evidence.
- 핵심 메시지: s1_ax/s1_gx 반복 관찰을 제한적으로 설명한다.
- 논문 사용 추천: appendix 후보

## fig_12_class3_excessive_lean.png
- Korean caption: Class 3 Excessive Lean의 recall/F1 비교.
- English caption: Recall and F1 comparison for Class 3 Excessive Lean.
- 핵심 메시지: 어려운 클래스에서 모델별 차이를 확인한다.
- 논문 사용 추천: 본문 또는 appendix 후보

## fig_13_confusion_matrix_grid.png
- Korean caption: 주요 모델의 row-normalized confusion matrix. 행은 true class, 열은 predicted class다.
- English caption: Row-normalized confusion matrices for key models. Rows are true classes and columns are predicted classes.
- 핵심 메시지: Shared-only failure pattern과 class-wise confusion을 진단한다.
- 논문 사용 추천: appendix 후보

## fig_14_parameter_vs_macro_f1.png
- Korean caption: controlled neural model의 parameter count와 Macro F1 관계.
- English caption: Parameter count versus Macro F1 for controlled neural models.
- 핵심 메시지: parameter count와 성능이 단순 비례하지 않음을 보여준다.
- 논문 사용 추천: appendix 후보
