# Curated Results Artifacts

이 폴더는 GitHub에서 바로 검토하기 좋은 경량 결과 산출물 mirror다. 원본 generated result tree인 `results/` 전체는 크기와 재생성 가능성 때문에 git에 올리지 않는다. 대신 논문 검토와 결과 해석에 필요한 CSV, Markdown, PNG figure만 선별해 복사했다.

## 포함 기준

- 포함: aggregate metric, confidence interval, paired difference, class-wise metric, subject-wise metric, confusion matrix, feature audit, feature importance, 핵심 figure, 해석 memo
- 제외: checkpoint, optimizer state, raw processed `.npy`, 대규모 full predictions/history 원본, `.pt`, `.pth`, `.ckpt`, dependency cache
- 예외: 사용자가 검토용으로 요청했던 작은 diagnostic `training_history.csv` 일부는 `review_bundles/` 안에 포함한다.
- 원칙: 원본 `results/`는 수정하지 않고 read-only source로만 사용한다.

## 폴더 구조

| folder | source result path | purpose |
|---|---|---|
| `full_supervised_matrix_v1/` | `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/` | 논문 후보 locked supervised matrix 결과 |
| `literature_baseline_full_extension_v1/` | `results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1/` | 문헌/전통 baseline 3-seed extension |
| `controlled_feature_extractor_comparison_v1/` | `results/controlled_feature_extractor_comparison/20260629_041712_controlled_feature_extractor_comparison_v1/` | common-head feature extractor 통제 비교 |
| `v3_component_ablation_v1/` | `results/v3_component_ablation/20260617_202714_v3_component_ablation_v1/` | proposed v3 component ablation |
| `xgboost_only_completion_v1/` | `results/xgboost_only_completion/20260629_053235_xgboost_only_completion_v1/` | skipped XGBoost baseline completion |
| `review_bundles/` | `results/review_bundles/` | 사용자가 요청했던 검토용 파일 묶음 |

## 해석 순서

1. `full_supervised_matrix_v1/aggregate_metrics_by_model.csv`로 main supervised result를 본다.
2. `literature_baseline_full_extension_v1/merged_with_locked_matrix/merged_full_comparison.csv`로 literature/classical baseline을 함께 본다.
3. `controlled_feature_extractor_comparison_v1/analysis/component_contribution_summary.csv`로 common-head 조건에서 feature extractor 차이를 본다.
4. `v3_component_ablation_v1/merged_with_existing_results/component_contribution_summary.csv`로 residual, identity, attention component를 본다.
5. `xgboost_only_completion_v1/merged_with_controlled_comparison/merged_controlled_plus_xgboost.csv`로 XGBoost completion을 controlled comparison에 끼워 본다.

## 주의

- 이 폴더는 원본 result directory의 전체 복사본이 아니다.
- 재현성 검증에는 원본 `results/`와 `configs/`, `src/`, `scripts/`를 함께 확인해야 한다.
- 결과 수치를 보고 hyperparameter, split, preprocessing, feature set을 변경하지 않았다.
- 최종 논문 표 포함 여부와 학술적 해석은 사용자가 판단한다.
