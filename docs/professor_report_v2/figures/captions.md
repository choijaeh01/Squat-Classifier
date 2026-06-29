# Figure Captions v2

## fig_01_feature_extractor_macro_f1.png
- Korean caption: 공통 classifier head 조건에서 feature extractor별 Macro F1과 CI를 비교한 그림.
- English caption: Macro F1 with confidence intervals across feature extractors under a shared classifier head.
- 핵심 메시지: Residual Channel-Shared Encoder는 All-Channel 1D CNN, XGBoost, RF와 경쟁 가능한 수준이며 Statistical Summary MLP가 가장 높다.

## fig_normalization_pipeline.png
- Korean caption: raw window conversion 이후 LOSO split, train-only scaler fit, transform, model/feature extraction까지의 처리 순서.
- English caption: Processing order from raw-window conversion to LOSO split, train-only scaler fitting, transformation, and model or feature extraction.
- 핵심 메시지: validation/test subject는 scaler fitting에 사용되지 않았고 per-window z-score 및 augmentation은 사용하지 않았다.

## fig_parameter_count_vs_macro_f1.png
- Korean caption: controlled neural model의 parameter count와 Macro F1 관계.
- English caption: Parameter count versus Macro F1 for controlled neural models.
- 핵심 메시지: common head parameter는 동일하며, parameter 수와 성능은 단순 비례하지 않는다.

## fig_confusion_matrix_row_normalized_grid.png
- Korean caption: 주요 모델의 row-normalized confusion matrix. 행은 true class, 열은 predicted class다.
- English caption: Row-normalized confusion matrices for key models. Rows are true classes and columns are predicted classes.
- 핵심 메시지: Shared 1D Encoder의 failure pattern과 Class 3 Excessive Lean의 혼동 패턴을 확인하기 위한 진단용 그림이다.
