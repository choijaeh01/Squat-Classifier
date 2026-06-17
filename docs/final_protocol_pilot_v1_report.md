# Final Protocol Pilot v1 Report

## 실행 개요

- 목적: 논문용 supervised LOSO에서 사용할 primary validation policy 확정 전 제한 pilot
- 실행 위치: CAU 서버 `/home/user3/workspace/Jae/squat_imu_experiments`
- CAU hostname: `4d244d15d634`
- Python: 3.10.12
- Device: `cuda`, NVIDIA RTX 6000 Ada Generation
- Commit: `a92b830f5cd921292676058ca250f54fcabb38a0`
- Config: `configs/final_protocol_pilot_v1.yaml`
- 결과 경로: `results/final_protocol_pilot/20260617_140012_final_protocol_pilot_v1/`

이 실행은 1 seed final-protocol pilot이다. 최종 논문 성능이나 모델 우열의 최종 결론으로 해석하지 않는다.

## Final Validation Policy

Split type은 `loso_with_within_train_subject_stratified_validation`이다.

각 fold에서 test subject 1명은 완전히 held-out으로 둔다. 나머지 5명 subject의 각 class 20 windows 중 16개는 train, 4개는 validation으로 분리한다. 따라서 fold별 기대 크기는 train 400, validation 100, test 100이다.

Scaler는 각 fold의 train indices 400개에만 fit한다. Validation/test windows는 scaler fit, early stopping 외 tuning, normalization fitting, augmentation, SSL pretraining에 사용하지 않는다.

## Cyclic Validation과의 차이

기존 cyclic validation pilot은 test subject 1명, validation subject 1명, train subject 4명으로 구성했다. 이 경우 train set이 400개이고 validation/test는 각각 subject 전체 100개였다.

Final protocol pilot은 test subject는 동일하게 100개로 유지하되, validation을 candidate train subjects 5명 내부에서 class-stratified로 뽑는다. 이 정책은 train 후보 subject 전체를 학습/검증 후보로 쓰면서도 test subject isolation을 유지한다.

## Leakage Check

- Fold 수: 6
- 모든 fold 크기: train 400, validation 100, test 100
- Test subject isolation: 모든 fold `True`
- Train/validation index disjoint: 모든 fold `True`
- Scaler fit windows: 모든 fold 및 모델에서 400
- Val indices used in scaler: 모든 row `False`
- Test indices used in scaler: 모든 row `False`
- Scaler leakage check: 모든 row `True`

세부 기록은 `validation_policy_summary.csv`와 `scaler_fit_audit.csv`에 저장했다.

## Model Parameter Count

| model | total params | encoder | aggregation | head |
|---|---:|---:|---:|---:|
| all_channel_conv1d_v1 | 16037 | 15712 | 0 | 325 |
| all_channel_conv1d_small | 4181 | 4056 | 0 | 125 |
| channel_shared_posres_attention_v3 | 14742 | 2064 | 8193 | 4485 |
| modality_shared_sensorattn_v3 | 16103 | 4128 | 9730 | 2245 |

## Pilot Metrics

| model | mean accuracy | mean macro F1 | success folds | failed folds |
|---|---:|---:|---:|---:|
| all_channel_conv1d_small | 0.8517 | 0.8298 | 6 | 0 |
| all_channel_conv1d_v1 | 0.8150 | 0.7893 | 6 | 0 |
| channel_shared_posres_attention_v3 | 0.8067 | 0.7950 | 6 | 0 |
| modality_shared_sensorattn_v3 | 0.5800 | 0.5357 | 6 | 0 |

`channel_shared_posres_attention_v3`는 `all_channel_conv1d_v1`보다 accuracy는 0.0083 낮고 macro F1은 0.0057 높았다. 이 차이는 1 seed pilot 수준에서는 결론으로 해석하지 않는다. 다만 full supervised matrix에 포함할 근거는 충분하다.

## Subject-wise 결과

| model | lowest subject | lowest macro F1 |
|---|---:|---:|
| all_channel_conv1d_small | 1 | 0.6180 |
| all_channel_conv1d_v1 | 1 | 0.5572 |
| channel_shared_posres_attention_v3 | 5 | 0.5941 |
| modality_shared_sensorattn_v3 | 5 | 0.1745 |

## Class-wise 결과

| model | lowest recall class | lowest recall | class 3 recall | class 3 F1 |
|---|---:|---:|---:|---:|
| all_channel_conv1d_small | 3 | 0.7333 | 0.7333 | 0.7288 |
| all_channel_conv1d_v1 | 3 | 0.6583 | 0.6583 | 0.6884 |
| channel_shared_posres_attention_v3 | 2 | 0.7083 | 0.7667 | 0.7562 |
| modality_shared_sensorattn_v3 | 0 | 0.4417 | 0.4833 | 0.4197 |

Class 3 Excessive Lean은 cyclic validation pilot에서 어려운 class였지만, final protocol pilot에서는 `channel_shared_posres_attention_v3`가 class 3 recall 0.7667을 보였다. `modality_shared_sensorattn_v3`는 class 3 개선이 제한적이다.

## Cyclic Pilot 대비 변화

| model | cyclic mean macro F1 | final-protocol pilot macro F1 | change |
|---|---:|---:|---:|
| all_channel_conv1d_small | 0.5902 | 0.8298 | +0.2396 |
| all_channel_conv1d_v1 | 0.6457 | 0.7893 | +0.1436 |
| channel_shared_posres_attention_v3 | 0.6439 | 0.7950 | +0.1511 |
| modality_shared_sensorattn_v3 | 0.3852 | 0.5357 | +0.1505 |

Final protocol은 validation subject를 별도 1명으로 고정하지 않고 training candidate subjects 내부에서 class-stratified validation을 구성한다. 이 변화는 validation set의 대표성을 높이고 early stopping 안정성을 개선했을 가능성이 있다. 단, 이 해석은 1 seed pilot에 한정된다.

## 판단

Final validation policy는 split leakage, scaler leakage, class balance, test subject isolation 조건을 만족했다. 3 seed full supervised matrix로 넘어가도 된다.

Full matrix 후보는 다음으로 제한하는 것을 권장한다.

- `all_channel_conv1d_v1`
- `all_channel_conv1d_small`
- `channel_shared_posres_attention_v3`
- `modality_shared_sensorattn_v3`는 보조/ablation 후보

Full supervised matrix에서도 augmentation, focal loss, SSL, external dataset은 별도 승인 전까지 사용하지 않는다.
