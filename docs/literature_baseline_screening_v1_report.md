# Literature Baseline Screening v1 Report

## 실행 범위

- 목적: 문헌 기반 temporal baseline 및 classical feature baseline의 1 seed screening
- 실행 위치: CAU 서버
- CAU hostname: `4d244d15d634`
- Python: `3.10.12`
- Torch: `2.8.0+cu129`
- Device: `cuda`, `NVIDIA RTX 6000 Ada Generation`
- scikit-learn availability: `true`
- 결과 경로: `results/literature_baseline_screening/20260617_174804_literature_baseline_screening_v1/`
- 실행 commit: `0842ffc5a8926e3156e844495e45cfd39dca5f39`

이번 실행은 1 seed screening이다. 3 seed full extension이 아니며, 기존 locked full supervised matrix 결과를 수정하지 않았다.

## 추가 Baseline 목록

| model | 의도 |
|---|---|
| `cnn_lstm_literature_v1` | IMU 문헌에서 널리 쓰이는 CNN-LSTM family 대표 baseline |
| `cnn_gru_literature_v1` | CNN-GRU recurrent unit 비교 |
| `rescnn_bigru_attention_lite_v1` | clean-room lite Residual CNN-BiGRU-Attention reference baseline |
| `tcn_literature_v1` | recurrent 없이 dilated temporal convolution으로 sequence modeling |
| `lstm_only_literature_v1` | CNN front-end 없는 LSTM-only baseline |
| `gru_only_literature_v1` | CNN front-end 없는 GRU-only baseline |
| `transformer_encoder_lite_v1` | small Transformer encoder temporal baseline |
| `feature_random_forest_v1` | handcrafted feature + RandomForest classical baseline |
| `feature_linear_svm_v1` | handcrafted feature + LinearSVM classical baseline |

## Parameter Count

| model | total params | notes |
|---|---:|---|
| `cnn_lstm_literature_v1` | 15645 | neural temporal |
| `cnn_gru_literature_v1` | 13533 | neural temporal |
| `rescnn_bigru_attention_lite_v1` | 38934 | neural temporal |
| `tcn_literature_v1` | 32069 | neural temporal |
| `lstm_only_literature_v1` | 6821 | neural temporal |
| `gru_only_literature_v1` | 5157 | neural temporal |
| `transformer_encoder_lite_v1` | 39173 | neural temporal |
| `feature_random_forest_v1` | not applicable | sklearn classical |
| `feature_linear_svm_v1` | not applicable | sklearn classical |

## 실행 결과 요약

- 총 run 수: 54
- 성공: 54
- 실패: 0
- skipped: 0
- split leakage check: 모두 통과
- scaler leakage check: 모두 통과
- scaler fit scope: train indices only
- augmentation, focal loss, balanced sampling, SSL, external dataset: 모두 비활성화

## 1 Seed Screening Metrics

| model | accuracy | macro F1 | weighted F1 |
|---|---:|---:|---:|
| `feature_random_forest_v1` | 0.8150 | 0.7900 | 0.7900 |
| `rescnn_bigru_attention_lite_v1` | 0.7817 | 0.7485 | 0.7485 |
| `feature_linear_svm_v1` | 0.7350 | 0.7164 | 0.7164 |
| `tcn_literature_v1` | 0.7467 | 0.7049 | 0.7049 |
| `transformer_encoder_lite_v1` | 0.7217 | 0.6847 | 0.6847 |
| `cnn_gru_literature_v1` | 0.5600 | 0.5096 | 0.5096 |
| `cnn_lstm_literature_v1` | 0.5167 | 0.4632 | 0.4632 |
| `gru_only_literature_v1` | 0.4517 | 0.4065 | 0.4065 |
| `lstm_only_literature_v1` | 0.3617 | 0.3021 | 0.3021 |

## Class 3 Excessive Lean

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| `feature_random_forest_v1` | 0.6833 | 0.6917 |
| `feature_linear_svm_v1` | 0.6667 | 0.6512 |
| `rescnn_bigru_attention_lite_v1` | 0.7000 | 0.6332 |
| `tcn_literature_v1` | 0.6000 | 0.5860 |
| `cnn_gru_literature_v1` | 0.5583 | 0.4400 |
| `cnn_lstm_literature_v1` | 0.4750 | 0.3799 |
| `transformer_encoder_lite_v1` | 0.3167 | 0.3646 |
| `gru_only_literature_v1` | 0.3583 | 0.3181 |
| `lstm_only_literature_v1` | 0.1917 | 0.1931 |

## Locked Matrix와 병합한 참고 Ranking

주의: 아래 ranking은 3 seed locked full result와 1 seed screening result를 함께 놓은 참고표다. `result_type`이 다르므로 직접 동등 비교하면 안 된다.

| rank | model | result type | mean macro F1 |
|---:|---|---|---:|
| 1 | `channel_shared_posres_attention_v3` | locked_3seed_full | 0.8108 |
| 2 | `feature_random_forest_v1` | literature_1seed_screening | 0.7900 |
| 3 | `all_channel_conv1d_small` | locked_3seed_full | 0.7740 |
| 4 | `all_channel_conv1d_v1` | locked_3seed_full | 0.7531 |
| 5 | `rescnn_bigru_attention_lite_v1` | literature_1seed_screening | 0.7485 |
| 6 | `feature_linear_svm_v1` | literature_1seed_screening | 0.7164 |
| 7 | `tcn_literature_v1` | literature_1seed_screening | 0.7049 |
| 8 | `transformer_encoder_lite_v1` | literature_1seed_screening | 0.6847 |

## 3 Seed Full Extension 후보

기계적 기준으로는 1 seed macro F1 0.70 이상 또는 reviewer-facing baseline 가치가 있는 모델을 후보로 둘 수 있다.

- 1순위 후보: `feature_random_forest_v1`, `rescnn_bigru_attention_lite_v1`
- 2순위 후보: `feature_linear_svm_v1`, `tcn_literature_v1`
- 조건부 후보: `transformer_encoder_lite_v1`

`cnn_lstm_literature_v1`, `cnn_gru_literature_v1`, `lstm_only_literature_v1`, `gru_only_literature_v1`은 이번 1 seed screening에서 상대적으로 낮았다. 다만 최종 포함 여부는 논문에서 reviewer가 기대할 baseline coverage와 실행 비용을 고려해 사용자가 판단해야 한다.

## 해석 제한

- 이 결과는 1 seed screening이다.
- 기존 locked full matrix main result를 대체하지 않는다.
- literature baseline 대비 우월성 claim은 3 seed full extension 전에는 금지한다.
- RandomForest/SVM은 neural model parameter count와 직접 비교하지 않는다.
