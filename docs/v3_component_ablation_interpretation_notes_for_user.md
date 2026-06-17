# v3 Component Ablation Interpretation Notes For User

이 문서는 수치 중심 메모다. 최종 학술적 해석과 논문 main narrative 반영 여부는 사용자가 판단한다.

## 질문별 수치 메모

### residual branch가 필요한가?

- original v3 Macro F1: 0.8108
- no residual Macro F1: 0.5937
- paired delta no_residual minus original v3: -0.2171
- CI: [-0.2945, -0.1449]

관찰: residual branch 제거 시 성능 저하가 크고 CI가 0을 포함하지 않았다.

### residual-only MLP가 얼마나 강한가?

- residual_only_mlp Macro F1: 0.8036
- original v3 대비 paired delta: -0.0072
- CI: [-0.0520, 0.0421]
- RandomForest 대비 paired delta: +0.0191
- CI: [-0.0265, 0.0674]

관찰: raw summary statistics MLP만으로도 높은 성능이 관찰됐다. RandomForest와의 paired CI는 0을 포함한다.

### identity embedding이 필요한가?

- no_identity Macro F1: 0.7675
- original v3 대비 paired delta: -0.0434
- CI: [-0.0852, -0.0072]

관찰: token identity embedding 제거는 original v3 대비 낮았다. 단, `no_identity`에는 residual branch가 남아 있으므로 channel-position cue를 완전히 제거한 실험은 아니다.

### attention pooling이 필요한가?

- no_attention meanpool Macro F1: 0.8082
- original v3 대비 paired delta: -0.0026
- CI: [-0.0402, 0.0343]

관찰: mean pooling ablation은 original attention v3와 거의 같은 수준이었다. 이 결과만 보면 attention pooling의 필수성은 강하게 주장하기 어렵다.

### Class 3 Excessive Lean에서 component별 차이는 어떤가?

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| original v3 | 0.6889 | 0.7184 |
| no residual | 0.5639 | 0.5321 |
| residual only MLP | 0.6972 | 0.7241 |
| no identity | 0.7278 | 0.7266 |
| no attention | 0.6806 | 0.7196 |

관찰: Class 3만 보면 no residual을 제외한 ablation들이 original v3와 유사한 수준이다. 따라서 Class 3 개선을 attention pooling 단독 효과로 설명하면 안 된다.

### attention과 RandomForest feature importance는 같은 채널을 보는가?

- channel-level Pearson correlation: 0.6361
- channel-level Spearman correlation: 0.3870
- v3 attention 상위 채널: `s1_ax`, `s0_ay`, `s0_ax`, `s1_az`, `s2_ax`
- RF top features: `s1_ax_std`, `s1_ax_energy`, `s1_ax_rms`, `s1_ax_max`, `s1_gx_mean`

관찰: 둘 다 `s1_ax`를 중요하게 보는 경향은 있다. 다만 attention weight와 feature importance는 산출 의미가 다르므로 동일한 설명 변수로 취급하면 안 된다.

## 논문 narrative에 줄 수 있는 영향

### 유지 가능한 방향

- naive shared v2의 실패와 v3의 개선은 여전히 유효하다.
- residual branch와 channel identity 정보가 중요한 역할을 했다는 설명은 수치적으로 방어 가능하다.
- proposed v3는 locked full matrix에서 여전히 가장 높은 평균 Macro F1이다.

### 조심해야 할 방향

- attention pooling이 핵심 component라고 강하게 주장하기 어렵다.
- residual-only MLP가 매우 강하므로, v3 성능이 shared temporal encoder만의 효과라고 주장하면 안 된다.
- `no_attention`과 original v3의 차이가 작기 때문에, 논문에서는 attention을 “성능을 보장하는 핵심”보다 “learnable aggregation option” 정도로 표현하는 편이 안전하다.

## 사용자 판단이 필요한 선택지

1. 논문 proposed model은 original v3로 유지하되, ablation에서 residual branch와 identity embedding의 중요성을 강조한다.
2. Attention pooling claim을 낮추고, position-aware identity + residual summary가 핵심이라는 narrative로 조정한다.
3. `residual_only_mlp` 결과를 limitations 또는 ablation discussion에 넣어 small-subject IMU에서는 handcrafted/statistical summaries도 강하다고 명시한다.
4. 추가 실험 없이 논문 작성으로 진행하거나, 교수님 피드백 후 `v3 without residual but with stronger encoder` 같은 후속 ablation을 별도 검토한다.

## 권장 문장 초안

안전한 문장:

> The component ablation suggests that the residual summary branch and positional identity information contributed materially to the proposed v3 model, whereas attention pooling alone showed limited separation from mean pooling under the current protocol.

피해야 할 문장:

> Attention pooling is the primary reason for the proposed model's performance.

피해야 할 문장:

> The shared encoder alone explains the performance gain over classical feature baselines.
