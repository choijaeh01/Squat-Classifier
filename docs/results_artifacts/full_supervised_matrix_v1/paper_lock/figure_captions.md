# Figure Captions for Paper Result Lock v1

## fig1_model_macro_f1_ci.png
- Korean: 모델별 평균 Macro F1과 bootstrap confidence interval을 비교한 그림.
- English: Mean Macro F1 with bootstrap confidence intervals across supervised LOSO runs.
- Key message: position-aware shared v3가 가장 높은 평균 Macro F1을 보인다.
- Suggested placement: Results, main model comparison.

## fig2_parameter_count_vs_macro_f1.png
- Korean: 모델 parameter count와 평균 Macro F1의 관계.
- English: Parameter count versus mean Macro F1 for each model.
- Key message: v3는 all-channel v1과 유사한 parameter budget에서 더 높은 평균 Macro F1을 보인다.
- Suggested placement: Results or Discussion.

## fig3_classwise_recall_heatmap.png
- Korean: 모델별 class-wise recall heatmap.
- English: Class-wise recall heatmap across models.
- Key message: naive shared v2는 특히 class 3에서 낮고, v3는 class-wise 균형이 개선된다.
- Suggested placement: Class-wise analysis.

## fig4_class3_recall_f1_by_model.png
- Korean: Class 3 Excessive Lean의 recall과 F1 비교.
- English: Recall and F1 for Class 3 Excessive Lean.
- Key message: v3가 naive shared v2 대비 class 3 성능을 크게 개선한다.
- Suggested placement: Class-wise analysis.

## fig5_subjectwise_macro_f1_heatmap.png
- Korean: subject-wise Macro F1 heatmap.
- English: Subject-wise Macro F1 heatmap.
- Key message: subject 5가 여러 모델에서 어려운 held-out subject로 나타난다.
- Suggested placement: Robustness analysis.

## fig6_v3_vs_baseline_paired_delta.png
- Korean: v3와 all-channel baselines의 paired Macro F1 차이.
- English: Paired Macro F1 differences between the proposed v3 model and all-channel baselines.
- Key message: v3 평균 차이는 양수지만 CI가 0을 포함하므로 우월성 주장은 조심해야 한다.
- Suggested placement: Main results or Discussion.

## fig7_aggregate_confusion_matrix_v3.png
- Korean: v3 모델의 aggregate confusion matrix.
- English: Aggregate confusion matrix for the proposed v3 model.
- Key message: v3의 주요 class confusion 구조를 보여준다.
- Suggested placement: Error analysis.

## fig8_aggregate_confusion_matrix_all_channel_v1.png
- Korean: all-channel Conv1D v1의 aggregate confusion matrix.
- English: Aggregate confusion matrix for the all-channel Conv1D v1 baseline.
- Key message: v3와 baseline의 error pattern 비교 기준을 제공한다.
- Suggested placement: Error analysis.
