# Controlled Feature Extractor Comparison v1 Artifacts

원본 경로: `results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1/`

공통 64-dim representation과 동일 MLP classifier head를 고정해 feature extractor 차이를 비교한 결과다.

## 핵심 파일

- `aggregate_metrics_by_model.csv`: controlled model별 평균 성능
- `common_head_verification.csv`: 공통 head 검증
- `analysis/component_contribution_summary.csv`: naive shared, identity, residual component 비교
- `analysis/controlled_architecture_comparison.csv`: controlled architecture comparison
- `analysis/practical_baseline_comparison.csv`: RF, SVM 등 practical baseline 비교
- `feature_definitions.csv`, `feature_audit.csv`: feature baseline audit
- `random_forest_feature_importance_aggregate.csv`: RF top features
- `figures/`: controlled comparison figure

## 해석 주의

원래 run에서는 `feature_xgboost_v1`이 dependency 부재로 skipped되었다. XGBoost는 `xgboost_only_completion_v1/`에서 동일 정책으로 별도 완료했다.
