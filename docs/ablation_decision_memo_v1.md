# Ablation Decision Memo v1

## 결론

지금은 추가 ablation을 실행하기보다 논문 초안을 먼저 작성하고 교수님 피드백을 받는 것을 추천한다. 다만 v3 claim 방어를 위해 최소 ablation을 1개만 추가한다면 `v3 without residual branch` 또는 `v3 without identity embeddings`가 가장 직접적이다.

## 후보별 판단

| 후보 | 필요한 run 수 | claim 강화 정도 | 시간 비용 | 국내 저널 필수성 | 후속 국제 논문 가능성 | 추천 |
|---|---:|---|---|---|---|---|
| v3 without residual branch | 18 | 높음 | 중간 | 선택 | 가능 | 교수님 피드백 후 최소 후보 |
| v3 without channel/sensor/modality/axis embeddings | 18 | 매우 높음 | 중간 | 선택 | 가능 | 교수님 피드백 후 최소 후보 |
| v3 without attention pooling | 18 | 중간 | 중간 | 선택 | 가능 | 후순위 |
| per-window z-score on/off | 126 이상 | 낮음 | 큼 | 필수 아님 | 가능 | 지금 비추천 |
| focal loss | 126 이상 | 낮음 | 큼 | 필수 아님 | 가능 | 지금 금지 유지 |
| SSL diagnostics | 별도 protocol 필요 | 높을 수 있음 | 큼 | 필수 아님 | 매우 적합 | 후속 연구 |
| external dataset benchmark | adapter 필요 | 높을 수 있음 | 큼 | 필수 아님 | 매우 적합 | 후속 연구 |

## 판단 근거

현재 full supervised matrix는 이미 126 runs로 충분히 큰 실험 단위다. v3는 all-channel baselines와 경쟁 가능하고 naive shared v2보다 명확히 높다. 다만 all-channel 대비 paired CI가 0을 포함하므로, 논문 초안에서는 통계적 우월성 대신 구조적 타당성과 경쟁 가능성을 강조해야 한다.

## 추천 실행 순서

1. 현재 결과로 논문 초안 작성
2. 교수님에게 result table, class-wise analysis, claim audit 보고
3. 피드백에서 ablation 요구가 있으면 최소 1-2개만 실행
4. SSL/external transfer는 후속 국제 논문 주제로 분리
