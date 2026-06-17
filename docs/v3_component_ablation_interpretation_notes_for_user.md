# v3 Component Ablation Interpretation Notes For User

이 문서는 수치 중심 메모다. 최종 학술적 해석과 논문 반영 여부는 사용자가 판단한다.

## 질문별 관찰

### residual branch가 중요한가?

`no_residual`은 mean Macro F1 0.5937로 original v3 0.8108보다 낮았다. Paired delta는 -0.2171이고 CI는 [-0.2945, -0.1449]였다. 이 결과만 보면 residual branch 제거가 큰 성능 하락과 함께 관찰됐다.

### residual branch만으로도 강한가?

`residual_only_mlp`는 mean Macro F1 0.8036이었다. Original v3 대비 paired delta는 -0.0072이고 CI는 [-0.0520, 0.0421]로 0을 포함했다. 이 결과는 residual summary branch만으로도 강한 baseline이 될 수 있음을 보여준다.

### channel/sensor/modality/axis identity embedding이 중요한가?

`no_identity`는 mean Macro F1 0.7675였다. Original v3 대비 paired delta는 -0.0434이고 CI는 [-0.0852, -0.0072]였다. 다만 이 모델은 residual branch가 남아 있어 channel-order 기반 정보가 완전히 제거된 것은 아니다.

### attention pooling이 꼭 필요한가?

`no_attention`은 mean Macro F1 0.8082였다. Original v3 대비 paired delta는 -0.0026이고 CI는 [-0.0402, 0.0343]으로 0을 포함했다. 이 결과만 보면 이번 설정에서는 attention pooling의 추가 이득이 명확하지 않다.

### residual_only_mlp는 RandomForest와 비교해 어떤가?

`residual_only_mlp`는 mean Macro F1 0.8036이고 `feature_random_forest_v1`은 0.7845였다. Paired delta는 residual_only_mlp - RF 기준 +0.0191이고 CI는 [-0.0265, 0.0674]로 0을 포함했다. 둘 사이의 우열은 단정하지 않는다.

### Class 3 Excessive Lean에서는 어떤가?

Class 3 F1은 `no_identity` 0.7266, `residual_only_mlp` 0.7241, `no_attention` 0.7196, original v3 0.7184, RF 0.6848, ResCNN-BiGRU-Attention-lite 0.6552, `no_residual` 0.5321 순으로 관찰됐다.

### Attention과 RandomForest feature importance는 정렬되는가?

Channel-level Pearson correlation은 0.6361, Spearman correlation은 0.3870이었다. v3 attention은 `s1_ax`, `s0_ay`, `s0_ax`, `s1_az`, `s2_ax`에 상대적으로 컸고, RF top feature는 `s1_ax` 관련 summary feature에 집중됐다. 단, attention weight와 feature importance는 같은 의미가 아니므로 정성적 참고로만 사용한다.

## 논문 claim에 주는 영향

- residual branch는 proposed v3 설명에서 중요한 역할로 제시할 수 있다.
- identity embedding은 token branch에서 관찰상 도움이 됐지만, residual branch가 남아 있는 ablation이므로 "모든 position cue가 필수"라고 표현하면 과하다.
- attention pooling은 이번 ablation에서 mean pooling 대비 명확한 이득이 없었다. 따라서 attention을 핵심 성능 원인으로 강하게 주장하기보다, token aggregation 후보 중 하나로 설명하는 편이 안전하다.
- residual-only MLP가 강하므로, v3 성능의 상당 부분이 channel-wise summary statistics로 설명될 수 있다는 점을 discussion에 포함할 필요가 있다.

## 사용자 판단이 필요한 지점

1. Proposed model 설명에서 attention pooling을 핵심 기여로 둘지, 보조 aggregation 방식으로 낮출지.
2. Residual branch를 "lightweight channel-summary branch"로 더 명확히 강조할지.
3. RandomForest와 residual-only MLP 결과를 본문 main discussion에 넣을지, appendix ablation으로 둘지.
4. 추가로 "v3 without residual and without identity" 같은 더 강한 제거 실험이 필요한지.

## 현재 단계에서 하지 않은 것

- 새 모델 구조 변경 없음
- hyperparameter tuning 없음
- split/preprocessing/scaler policy 변경 없음
- focal loss, balanced sampling, augmentation, SSL, external dataset 사용 없음
