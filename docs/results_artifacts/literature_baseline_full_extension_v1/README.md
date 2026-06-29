# Literature Baseline Full Extension v1 Artifacts

원본 경로: `results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1/`

문헌 기반 temporal baseline과 classical feature baseline을 final validation policy에서 3 seeds x 6 folds로 평가한 결과다.

## 핵심 파일

- `aggregate_metrics_by_model.csv`: extension 모델별 평균 성능
- `merged_with_locked_matrix/merged_full_comparison.csv`: locked full matrix와 extension 결과 병합표
- `merged_with_locked_matrix/merged_paired_differences.csv`: 주요 paired comparison
- `feature_definitions.csv`, `feature_audit.csv`: classical feature가 signal-derived인지 검증
- `random_forest_feature_importance_aggregate.csv`: RandomForest 상위 feature
- `linear_svm_coefficients_aggregate.csv`: Linear SVM coefficient summary
- `figures/`: literature extension 성능과 feature importance figure

## 해석 주의

이 extension은 기존 locked full matrix를 덮어쓰지 않는다. 모델 family가 다르므로 최종 논문 main table 포함 여부는 별도 판단이 필요하다.
