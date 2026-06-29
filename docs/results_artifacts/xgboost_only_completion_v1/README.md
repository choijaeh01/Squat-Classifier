# XGBoost Only Completion v1 Artifacts

원본 경로: `results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1/`

`controlled_feature_extractor_comparison_v1`에서 skipped된 `feature_xgboost_v1`만 CAU project-local XGBoost dependency로 별도 실행한 결과다.

## 핵심 파일

- `aggregate_metrics_by_model.csv`: XGBoost 3-seed 6-fold 평균 성능
- `bootstrap_confidence_intervals.csv`: CI
- `paired_model_differences.csv`: RF, stats MLP, shared residual, all-channel CNN 대비 paired difference
- `feature_definitions.csv`, `feature_audit.csv`: XGBoost feature audit
- `xgboost_feature_importance_aggregate.csv`: XGBoost top feature
- `merged_with_controlled_comparison/merged_controlled_plus_xgboost.csv`: controlled comparison에 XGBoost completion을 병합한 표
- `figures/`: XGBoost comparison 및 feature importance figure

## 해석 주의

XGBoost는 Macro F1 0.7961을 기록했지만, 주요 reference model과의 paired CI가 모두 0을 포함한다. 특정 baseline 대비 명확한 우월성은 주장하지 않는다.
