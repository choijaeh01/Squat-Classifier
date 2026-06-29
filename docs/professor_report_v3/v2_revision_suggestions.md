# v2 Revision Suggestions

## Position identity 표현 조정

- v2에서 position identity가 핵심 contribution처럼 읽히는 문장은 `identity는 shared-only 구조에서 일부 보완 효과가 있었지만, 최종 claim은 residual branch 중심`으로 낮추는 것이 좋다.
- `identity가 성능 개선의 핵심`이라는 표현은 피하고, `residual branch가 들어간 뒤 identity 추가 이득은 제한적`이라는 2x2 결과를 함께 제시한다.

## Residual branch 용어

- `raw summary residual branch`라는 표현이 있으면 `train-scaled channel-wise statistical branch`로 고치는 것이 더 정확하다.
- residual/statistical MLP branch는 train-only StandardScaler transform 이후 model 내부에서 mean/std/min/max를 계산한다.

## Feature set 구분

- Statistical Summary MLP와 residual branch는 mean/std/min/max, 18 channels x 4 = 72 features다.
- RF/XGBoost/SVM feature set은 162 signal-derived features이며, energy/RMS/median/peak-to-peak/dominant frequency 계열이 포함된다.
- Statistical Summary MLP 설명에 energy/RMS가 들어간 것처럼 보이는 문구는 제거한다.

## Attention 표현 조정

- 이번 교수님 보고와 국내 논문 claim에서는 attention을 핵심으로 두지 않는다.
- attention이나 position identity는 appendix 또는 future work로 낮추고, 본문은 residual branch의 shared encoder bottleneck 완화에 집중한다.
