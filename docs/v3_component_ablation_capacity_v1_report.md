# v3_component_ablation_capacity_v1 Report

## 범위

이 보고서는 `channel_shared_posres_attention_v3`와 component ablation 모델의 parameter count를 비교한다. 학습은 수행하지 않았다.

## Parameter Count

| model | total | encoder | identity | attention | residual | head | removed component |
|---|---:|---:|---:|---:|---:|---:|---|
| channel_shared_posres_attention_v3 | 14742 | 2064 | 832 | 545 | 6752 | 4485 | none |
| channel_shared_posres_attention_v3_no_residual | 5942 | 2064 | 832 | 545 | 0 | 2437 | residual_branch |
| channel_shared_posres_attention_v3_residual_only_mlp | 9189 | 0 | 0 | 0 | 6752 | 2437 | shared_encoder_and_attention |
| channel_shared_posres_attention_v3_no_identity | 13910 | 2064 | 0 | 545 | 6752 | 4485 | channel_sensor_modality_axis_embeddings |
| channel_shared_posres_meanpool_v3_no_attention | 14197 | 2064 | 832 | 0 | 6752 | 4485 | attention_pooling |

## 해석 주의점

- 이 표는 capacity 확인용이며 성능 해석을 포함하지 않는다.
- `no_identity`는 token identity embedding만 제거하며, residual branch는 여전히 channel order 기반 summary를 포함한다.
- `residual_only_mlp`는 v3 residual branch와 동일한 mean/std/min/max channel summary만 사용한다.
- 결과가 낮거나 높아도 model width, learning rate, split, preprocessing을 변경하지 않는다.
