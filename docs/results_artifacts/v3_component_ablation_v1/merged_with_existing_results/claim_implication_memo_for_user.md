# Claim Implication Memo For User

이 메모는 가능한 해석 시나리오만 정리한다. 최종 학술적 판단은 사용자가 한다.

| comparison | delta comparison-reference | CI |
|---|---:|---|
| v3 original vs no_residual | -0.2171 | [-0.2944900616475894, -0.14488209545126599] |
| v3 original vs residual_only_mlp | -0.0072 | [-0.0519981588868294, 0.04208977267229026] |
| v3 original vs no_identity | -0.0434 | [-0.08517572210191719, -0.007160423698970655] |
| v3 original vs no_attention | -0.0026 | [-0.040217695349763306, 0.03426833903503383] |
| residual_only_mlp vs feature_random_forest_v1 | 0.0191 | [-0.026543276472870473, 0.06735540582654134] |
| no_residual vs all_channel_conv1d_small | -0.1803 | [-0.2548175715687325, -0.11483080256506831] |
| no_identity vs channel_shared_attentionpool_v2 | 0.4539 | [0.39877103586009743, 0.5045881496619908] |
| no_identity vs channel_shared_meanpool_v2 | 0.5318 | [0.48598918396379265, 0.5812875857263413] |

## 주의

- paired CI가 0을 포함하면 우월성/열등성 표현을 피한다.
- `no_identity` 결과는 residual branch가 남아 있어 token identity embedding 효과만 분리한 것이다.
- `residual_only_mlp`가 높게 나오면 v3의 일부 성능이 summary statistics로 설명될 수 있음을 뜻하지만 RandomForest와 같은 모델은 아니다.
