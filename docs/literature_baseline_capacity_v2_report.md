# literature_baseline_capacity_v2 Report

## 범위

이 보고서는 literature temporal baseline extension의 parameter count audit이다. 학습은 수행하지 않았다.

## Parameter Count

| model | total params | encoder | aggregation | head | notes |
|---|---:|---:|---:|---:|---|
| feature_random_forest_v1 |  |  |  |  | classical baseline; parameter count not applicable; training requires scikit-learn |
| feature_linear_svm_v1 |  |  |  |  | classical baseline; parameter count not applicable; training requires scikit-learn |
| rescnn_bigru_attention_lite_v1 | 38934 | 22432 | 1585 | 2245 | lite clean-room Residual CNN-BiGRU-Attention reference baseline |
| tcn_literature_v1 | 32069 | 31904 | 0 | 165 | dilated temporal convolutional network baseline |
| transformer_encoder_lite_v1 | 39173 | 38832 | 0 | 341 | small Transformer encoder temporal baseline |
| cnn_lstm_literature_v1 | 15645 | 7032 | 0 | 165 | simple 1D CNN-LSTM family representative |
| cnn_gru_literature_v1 | 13533 | 7032 | 0 | 165 | simple 1D CNN-GRU family representative |
| lee_style_cnn_lstm_2d_v1 | 62637 | 10728 | 0 | 2245 | adapted Lee-style internal 40-step downsampling plus 2D CNN plus LSTM |
| channel_shared_posres_attention_v3 | 14742 | 2064 | 8193 | 4485 | locked full matrix reference model |
| all_channel_conv1d_v1 | 16037 | 15712 | 0 | 325 | locked full matrix reference model |
| all_channel_conv1d_small | 4181 | 4056 | 0 | 125 | locked full matrix reference model |

## 주의점

- CNN-LSTM/GRU 계열은 IMU 문헌에서 널리 쓰이는 family representative baseline이며 특정 외부 논문의 exact reproduction이 아니다.
- lee_style_cnn_lstm_2d_v1은 512-step 입력을 모델 내부에서 deterministic 40-step representation으로 downsample한 뒤 2D CNN과 LSTM을 적용하는 adapted clean-room baseline이다.
- Lee-style 모델은 Lee et al.의 exact reproduction이 아니며, 현재 센서 수, channel 수, label, LOSO protocol에 맞춘 reviewer-facing baseline이다.
- rescnn_bigru_attention_lite_v1은 이전 졸업논문 결과 재사용이 아니라 동일 protocol에서 새로 평가할 clean-room reference baseline이다.
- classical baseline은 scikit-learn 설치 여부에 따라 실행 또는 skipped 처리된다.
- 이 audit은 capacity 확인용이며 성능 해석을 포함하지 않는다.
