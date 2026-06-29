# v3 Component Ablation v1 Artifacts

원본 경로: `results/v3_component_ablation/20260617_202714_v3_component_ablation_v1/`

`channel_shared_posres_attention_v3`의 residual branch, identity embedding, attention pooling component를 비교한 ablation 결과다.

## 핵심 파일

- `aggregate_metrics_by_model.csv`: ablation 모델별 평균 성능
- `paired_model_differences.csv`: ablation paired differences
- `merged_with_existing_results/component_contribution_summary.csv`: original v3 대비 component contribution
- `merged_with_existing_results/attention_rf_alignment_summary.md`: v3 attention과 RF feature importance 정성 alignment
- `merged_with_existing_results/attention_rf_alignment_channel.csv`: channel-level alignment
- `figures/`: ablation 성능, component delta, attention/RF alignment figure

## 해석 주의

Residual branch 제거는 큰 성능 하락을 보였고, no-attention meanpool ablation은 original v3와 가까웠다. 따라서 attention 우월성보다 residual 및 position-aware signal 보존을 보수적으로 해석한다.
