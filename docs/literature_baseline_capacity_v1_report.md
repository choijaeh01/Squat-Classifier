# Literature Baseline Capacity v1 Report

## 범위

이 보고서는 literature temporal baseline extension v1의 parameter count audit이다. 학습은 수행하지 않았다.

## Parameter Count

| model | total params | encoder | aggregation | head | notes |
|---|---:|---:|---:|---:|---|
| cnn_lstm_literature_v1 | 15645 | 7032 | 0 | 165 | representative clean-room CNN-LSTM temporal baseline |
| cnn_gru_literature_v1 | 13533 | 7032 | 0 | 165 | CNN-GRU variant for recurrent-unit comparison |
| rescnn_bigru_attention_lite_v1 | 38934 | 22432 | 1585 | 2245 | lite clean-room Residual CNN-BiGRU-Attention reference baseline |
| tcn_literature_v1 | 32069 | 31904 | 0 | 165 | dilated temporal convolutional network baseline |
| lstm_only_literature_v1 | 6821 | 6656 | 0 | 165 | recurrent-only LSTM baseline without CNN front-end |
| gru_only_literature_v1 | 5157 | 4992 | 0 | 165 | recurrent-only GRU baseline without CNN front-end |
| transformer_encoder_lite_v1 | 39173 | 38832 | 0 | 341 | small Transformer encoder temporal baseline |
| feature_random_forest_v1 |  |  |  |  | classical baseline; parameter count not applicable; training requires scikit-learn |
| feature_linear_svm_v1 |  |  |  |  | classical baseline; parameter count not applicable; training requires scikit-learn |
| channel_shared_posres_attention_v3 | 14742 | 2064 | 8193 | 4485 | locked full matrix reference model |
| all_channel_conv1d_v1 | 16037 | 15712 | 0 | 325 | locked full matrix reference model |
| all_channel_conv1d_small | 4181 | 4056 | 0 | 125 | locked full matrix reference model |

## 주의점

- CNN-LSTM/GRU 계열은 IMU 문헌에서 널리 쓰이는 family representative baseline이며 특정 외부 논문의 exact reproduction이 아니다.
- rescnn_bigru_attention_lite_v1은 이전 졸업논문 결과 재사용이 아니라 동일 protocol에서 새로 평가할 clean-room reference baseline이다.
- classical baseline은 scikit-learn 설치 여부에 따라 screening에서 실행 또는 skipped 처리된다.
- full extension 포함 여부는 1 seed screening 이후 사용자가 승인해야 한다.
