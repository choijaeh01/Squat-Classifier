# Full Supervised Matrix v1 Artifacts

원본 경로: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`

이 폴더는 논문 후보 locked supervised matrix 결과의 핵심 요약본이다.

## 핵심 파일

- `aggregate_metrics_by_model.csv`: 모델별 3-seed 6-fold 평균 성능
- `bootstrap_confidence_intervals.csv`: accuracy, Macro F1, weighted F1 CI
- `paired_model_differences.csv`: 주요 모델 간 paired difference
- `classwise_metrics_by_model.csv`: class-wise precision, recall, F1
- `subjectwise_metrics_by_model.csv`: subject-wise Macro F1
- `confusion_matrices.csv`: 모델별 aggregate confusion matrix
- `paper_lock/`: paper result lock 산출물
- `figures/`: main model comparison, class-wise, subject-wise, confusion figure

## 해석 주의

`channel_shared_posres_attention_v3`가 평균 Macro F1 기준 가장 높았지만, all-channel baseline 대비 paired CI가 0을 포함한다. 통계적 우월성보다 경쟁 가능한 성능으로 해석한다.
