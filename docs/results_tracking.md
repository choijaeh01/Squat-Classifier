# Results Tracking

이 문서는 clean-room squat IMU 실험의 실행 이력을 추적한다. `results/` 아래 산출물은 git에 commit하지 않으며, 문서에는 실행 조건과 해석 범위만 기록한다.

## real_smoke_training_v1

- 목적: 실제 processed target dataset으로 제한된 학습 루프가 CAU에서 동작하는지 확인
- 실행 위치: CAU
- 결과 경로: `results/smoke_training/20260617_094334_real_smoke_training_v1/`
- 범위: test subject 1, validation subject 2, train subjects 3/4/5/6
- 제한: max epoch 1, max train batches 3
- 상태: 6개 모델 모두 성공
- 해석: smoke metric은 성능 해석에 사용하지 않음

## pilot_loso_v1

- 목적: 1 seed 전체 6-fold LOSO runner 검증
- 실행 위치: CAU
- 원격 경로: `/home/user3/workspace/Jae/squat_imu_experiments`
- 결과 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/`
- seed: 42
- folds: 6
- models: 6
- 성공/실패: 36 성공, 0 실패
- leakage: 모든 fold에서 `leakage_check_passed=True`
- scaler: fold별 train subjects only, fold별 400 windows fit

Pilot aggregate metrics:

| model | mean accuracy | mean macro F1 |
|---|---:|---:|
| all_channel_conv1d_v1 | 0.6817 | 0.6457 |
| all_channel_conv1d_small | 0.6417 | 0.5902 |
| modality_shared_meanpool_v2 | 0.3467 | 0.2485 |
| channel_shared_attentionpool_v2 | 0.3000 | 0.2133 |
| cnn2d_baseline_v1 | 0.3133 | 0.2078 |
| channel_shared_meanpool_v2 | 0.2550 | 0.1400 |

주의: 이 표는 pipeline 검증용 1 seed pilot 결과이다. 최종 논문 성능이나 모델 우열로 해석하지 않는다.

## pilot_loso_diagnostics_v1

- 목적: `pilot_loso_v1`의 shared 모델 부진 원인 진단
- 입력 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/`
- 출력 경로: `results/pilot_loso/20260617_120216_pilot_loso_v1/diagnostics/`
- 실행 위치: 로컬 분석 및 CAU figure 생성
- collapse 기준: dominant predicted class 70% 이상 또는 unique predicted class 2개 이하
- 결론: 모든 모델이 5개 class를 예측했으며 완전 collapse는 아님
- 주요 문제: channel-shared 모델이 class 3 Excessive Lean을 거의 맞추지 못하고 class 2 Butt Wink 및 class 4 Partial Squat으로 밀림

Class 3 recall:

| model | class 3 recall |
|---|---:|
| all_channel_conv1d_v1 | 0.5083 |
| all_channel_conv1d_small | 0.4750 |
| modality_shared_meanpool_v2 | 0.1417 |
| cnn2d_baseline_v1 | 0.1333 |
| channel_shared_meanpool_v2 | 0.0417 |
| channel_shared_attentionpool_v2 | 0.0417 |

## overfit_diagnostic_v1

- 목적: 40개 balanced subset memorization sanity check
- 실행 위치: CAU
- 결과 경로: `results/overfit_diagnostics/20260617_122833_overfit_diagnostic_v1/`
- subset: subjects 3, 4, 5, 6에서 class별 8개, 총 40개
- scaler: selected subset only
- 일반화 성능 해석 금지

| model | reached 95% train acc | epoch | final train acc |
|---|---|---:|---:|
| all_channel_conv1d_v1 | true | 10 | 0.9500 |
| all_channel_conv1d_small | true | 18 | 0.9500 |
| channel_shared_meanpool_v2 | false | - | 0.6250 |
| channel_shared_attentionpool_v2 | false | - | 0.7000 |
| modality_shared_meanpool_v2 | false | - | 0.7500 |
| cnn2d_baseline_v1 | false | - | 0.6250 |

해석: shared 계열은 full generalization 이전 단계인 small subset memorization에서도 제한이 보인다. 다음 architecture correction에서는 position-aware shared encoder를 검토한다.

## model_capacity_v3

- 목적: position-aware shared encoder v3 모델의 parameter budget 확인
- 결과 경로: `results/model_capacity_v3/model_capacity_table.csv`
- target range: all-channel Conv1D v1의 0.5배-2배, 약 8k-32k

| model | total params | encoder | aggregation | head |
|---|---:|---:|---:|---:|
| channel_shared_posres_attention_v3 | 14742 | 2064 | 8193 | 4485 |
| modality_shared_sensorattn_v3 | 16103 | 4128 | 9730 | 2245 |

두 모델 모두 목표 parameter range 안에 있다.

## overfit_diagnostic_v2_with_v3

- 목적: v3 모델이 40개 balanced subset을 외울 수 있는지 확인
- 실행 위치: CAU
- 결과 경로: `results/overfit_diagnostics/20260617_130057_overfit_diagnostic_v2_with_v3/`
- 일반화 성능 해석 금지

| model | reached 95% train acc | epoch | final train acc |
|---|---|---:|---:|
| all_channel_conv1d_v1 | true | 10 | 0.9500 |
| all_channel_conv1d_small | true | 18 | 0.9500 |
| channel_shared_meanpool_v2 | false | - | 0.5250 |
| channel_shared_attentionpool_v2 | false | - | 0.6250 |
| modality_shared_meanpool_v2 | false | - | 0.7500 |
| channel_shared_posres_attention_v3 | true | 30 | 0.9500 |
| modality_shared_sensorattn_v3 | true | 87 | 0.9500 |

## pilot_loso_v2_with_v3

- 목적: v3 shared 모델의 1 seed 6-fold pilot 검증
- 실행 위치: CAU
- 결과 경로: `results/pilot_loso/20260617_130150_pilot_loso_v2_with_v3/`
- 성공/실패: 42 성공, 0 실패
- leakage: 모든 fold에서 subject isolation 통과
- scaler: fold별 train subjects only

| model | mean accuracy | mean macro F1 |
|---|---:|---:|
| all_channel_conv1d_v1 | 0.6817 | 0.6457 |
| channel_shared_posres_attention_v3 | 0.6850 | 0.6439 |
| all_channel_conv1d_small | 0.6417 | 0.5902 |
| modality_shared_sensorattn_v3 | 0.4550 | 0.3852 |
| modality_shared_meanpool_v2 | 0.3467 | 0.2485 |
| channel_shared_attentionpool_v2 | 0.3000 | 0.2133 |
| channel_shared_meanpool_v2 | 0.2550 | 0.1400 |

`channel_shared_posres_attention_v3`는 v2 shared 모델 대비 큰 개선을 보였고, full supervised matrix에 포함할 가치가 있다. 단, 이 결과는 1 seed pilot이다.

## final_protocol_pilot_v1

- 목적: 논문용 primary LOSO validation policy 확정 전 제한 pilot
- 실행 위치: CAU
- commit: `a92b830f5cd921292676058ca250f54fcabb38a0`
- 결과 경로: `results/final_protocol_pilot/20260617_140012_final_protocol_pilot_v1/`
- split policy: `loso_with_within_train_subject_stratified_validation`
- fold 크기: train 400, validation 100, test 100
- leakage: test subject isolation 및 train/validation index disjoint 모두 통과
- scaler: train indices only, val/test index 사용 없음
- 성공/실패: 24 성공, 0 실패

| model | mean accuracy | mean macro F1 |
|---|---:|---:|
| all_channel_conv1d_small | 0.8517 | 0.8298 |
| all_channel_conv1d_v1 | 0.8150 | 0.7893 |
| channel_shared_posres_attention_v3 | 0.8067 | 0.7950 |
| modality_shared_sensorattn_v3 | 0.5800 | 0.5357 |

Class 3 Excessive Lean:

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| all_channel_conv1d_small | 0.7333 | 0.7288 |
| all_channel_conv1d_v1 | 0.6583 | 0.6884 |
| channel_shared_posres_attention_v3 | 0.7667 | 0.7562 |
| modality_shared_sensorattn_v3 | 0.4833 | 0.4197 |

이 결과는 1 seed final-protocol pilot이다. 최종 논문 성능으로 해석하지 않는다. Split/scaler policy가 통과했으므로 3 seed full supervised matrix로 넘어갈 수 있다.

## full_supervised_matrix_v1

- 목적: 논문 결과 후보 supervised LOSO full matrix
- 실행 위치: CAU
- commit: `9e49ddcb64b0cfef1d77a05c83fa9d68de961ca7`
- 결과 경로: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`
- run 수: 126
- 성공/실패: 126 성공, 0 실패
- split leakage: 통과
- scaler leakage: 통과

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| channel_shared_posres_attention_v3 | 0.8217 | 0.8108 | 0.8108 | [0.7644, 0.8537] |
| all_channel_conv1d_small | 0.8044 | 0.7740 | 0.7740 | [0.7026, 0.8424] |
| all_channel_conv1d_v1 | 0.7839 | 0.7531 | 0.7531 | [0.6502, 0.8429] |
| modality_shared_sensorattn_v3 | 0.6444 | 0.6023 | 0.6023 | [0.5003, 0.6944] |
| cnn2d_baseline_v1 | 0.4472 | 0.3688 | 0.3688 | [0.2904, 0.4489] |
| channel_shared_attentionpool_v2 | 0.3794 | 0.3136 | 0.3136 | [0.2553, 0.3731] |
| channel_shared_meanpool_v2 | 0.3172 | 0.2356 | 0.2356 | [0.1780, 0.2955] |

Paired macro F1 difference:

- `channel_shared_posres_attention_v3` minus `all_channel_conv1d_v1`: +0.0577, CI [-0.0251, 0.1511]
- `channel_shared_posres_attention_v3` minus `all_channel_conv1d_small`: +0.0368, CI [-0.0319, 0.1097]

Class 3 Excessive Lean:

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| channel_shared_posres_attention_v3 | 0.6889 | 0.7184 |
| all_channel_conv1d_v1 | 0.7000 | 0.6693 |
| all_channel_conv1d_small | 0.6444 | 0.6373 |
| modality_shared_sensorattn_v3 | 0.5250 | 0.4548 |
| cnn2d_baseline_v1 | 0.5028 | 0.3937 |
| channel_shared_attentionpool_v2 | 0.1917 | 0.1664 |
| channel_shared_meanpool_v2 | 0.0917 | 0.0840 |

Main shared model 후보는 `channel_shared_posres_attention_v3`이다. 단, all-channel baseline 대비 통계적 우월성은 아직 주장하지 않는다.

## literature_baseline_screening_v1

- 목적: clean-room 문헌 기반 temporal baseline 1 seed screening
- locked matrix 상태: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/` 수정 금지
- config: `configs/literature_baseline_screening_v1.yaml`
- capacity config: `configs/literature_baseline_capacity_v1.yaml`
- full extension template: `configs/literature_baseline_full_extension_template_v1.yaml`
- local dry-run: 완료, 학습 없음
- CAU screening: 완료
- 결과 경로: `results/literature_baseline_screening/20260617_174804_literature_baseline_screening_v1/`
- 실행 commit: `0842ffc5a8926e3156e844495e45cfd39dca5f39`
- 성공/실패/skipped: 54 성공, 0 실패, 0 skipped
- leakage/scaler check: 모두 통과

Baseline 목록:

| model | 의도 |
|---|---|
| cnn_lstm_literature_v1 | IMU 문헌에서 흔한 CNN-LSTM family 대표 baseline |
| cnn_gru_literature_v1 | CNN-GRU recurrent unit 비교 |
| rescnn_bigru_attention_lite_v1 | clean-room lite Residual CNN-BiGRU-Attention reference |
| tcn_literature_v1 | recurrent 없이 dilated temporal convolution 비교 |
| lstm_only_literature_v1 | CNN 없이 recurrent modeling만 비교 |
| gru_only_literature_v1 | LSTM-only 대비 GRU-only 비교 |
| transformer_encoder_lite_v1 | small Transformer encoder temporal baseline |
| feature_random_forest_v1 | handcrafted feature classical baseline, sklearn 필요 |
| feature_linear_svm_v1 | handcrafted feature linear SVM baseline, sklearn 필요 |

주의: 이 결과는 1 seed screening이므로 paper main table에 바로 포함하지 않는다. 3 seed full extension 후보 제안용으로만 사용한다.

1 seed screening 결과:

| model | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
| feature_random_forest_v1 | 0.8150 | 0.7900 | 0.7900 |
| rescnn_bigru_attention_lite_v1 | 0.7817 | 0.7485 | 0.7485 |
| feature_linear_svm_v1 | 0.7350 | 0.7164 | 0.7164 |
| tcn_literature_v1 | 0.7467 | 0.7049 | 0.7049 |
| transformer_encoder_lite_v1 | 0.7217 | 0.6847 | 0.6847 |
| cnn_gru_literature_v1 | 0.5600 | 0.5096 | 0.5096 |
| cnn_lstm_literature_v1 | 0.5167 | 0.4632 | 0.4632 |
| gru_only_literature_v1 | 0.4517 | 0.4065 | 0.4065 |
| lstm_only_literature_v1 | 0.3617 | 0.3021 | 0.3021 |

Class 3 Excessive Lean:

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| feature_random_forest_v1 | 0.6833 | 0.6917 |
| feature_linear_svm_v1 | 0.6667 | 0.6512 |
| rescnn_bigru_attention_lite_v1 | 0.7000 | 0.6332 |
| tcn_literature_v1 | 0.6000 | 0.5860 |
| transformer_encoder_lite_v1 | 0.3167 | 0.3646 |
