# Literature Baseline Interpretation Notes For User

이 문서는 결론을 대신 내리지 않고, 사용자가 논문 main result 포함 여부를 판단할 때 볼 수 있는 수치 중심 메모다.

## RandomForest가 3 seed에서도 강한가?

네. `feature_random_forest_v1`은 literature full extension 내에서 가장 높은 평균 Macro F1을 기록했다.

- Accuracy: 0.8056
- Macro F1: 0.7845
- Weighted F1: 0.7845
- Macro F1 CI: [0.7318, 0.8376]

기존 locked matrix의 `channel_shared_posres_attention_v3` Macro F1 0.8108보다는 낮고, `all_channel_conv1d_small` 0.7740보다는 높다. 단, 모델 family와 목적이 다르므로 최종 해석은 별도로 판단해야 한다.

## CNN-LSTM simple baseline은 계속 낮은가?

네. 3 seed full extension에서도 simple CNN-LSTM/GRU 계열은 낮은 편이다.

| model | macro F1 |
|---|---:|
| `cnn_lstm_literature_v1` | 0.5325 |
| `cnn_gru_literature_v1` | 0.5045 |

## Lee-style adapted CNN-LSTM은 simple CNN-LSTM보다 개선되는가?

네. `lee_style_cnn_lstm_2d_v1`은 simple CNN-LSTM보다 높다.

| comparison | macro F1 |
|---|---:|
| `lee_style_cnn_lstm_2d_v1` | 0.6269 |
| `cnn_lstm_literature_v1` | 0.5325 |
| delta | +0.0944 |

다만 Lee-style adapted baseline은 `rescnn_bigru_attention_lite_v1`, `transformer_encoder_lite_v1`, `tcn_literature_v1`보다 낮았다.

## ResCNN-BiGRU-Attention-lite는 proposed v3와 얼마나 가까운가?

| model | result type | macro F1 |
|---|---|---:|
| `channel_shared_posres_attention_v3` | locked_3seed_full | 0.8108 |
| `rescnn_bigru_attention_lite_v1` | literature_extension_3seed_full | 0.7691 |
| difference |  | -0.0417 |

수치상 proposed v3가 높지만, 최종 claim은 paired difference 및 논문 구성 방향과 함께 판단해야 한다.

## TCN은 recurrent baseline보다 강한가?

네. 이번 extension에서는 TCN이 simple CNN-LSTM/GRU보다 높다.

| model | macro F1 |
|---|---:|
| `tcn_literature_v1` | 0.7151 |
| `lee_style_cnn_lstm_2d_v1` | 0.6269 |
| `cnn_lstm_literature_v1` | 0.5325 |
| `cnn_gru_literature_v1` | 0.5045 |

## Class 3에서는 어떤 baseline이 강한가?

Class 3 F1 기준으로는 `feature_random_forest_v1`이 가장 높았다.

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| `feature_random_forest_v1` | 0.6500 | 0.6848 |
| `rescnn_bigru_attention_lite_v1` | 0.6833 | 0.6552 |
| `feature_linear_svm_v1` | 0.6056 | 0.5800 |
| `tcn_literature_v1` | 0.6167 | 0.5650 |
| `transformer_encoder_lite_v1` | 0.5056 | 0.5390 |
| `lee_style_cnn_lstm_2d_v1` | 0.4861 | 0.4607 |

기존 locked proposed v3의 Class 3 F1은 0.7184였다.

## RandomForest top features는 특정 sensor/channel에 집중되는가?

상위 importance는 오른쪽 허벅지 센서 `s1`의 `ax` feature에 강하게 집중됐다.

| rank | feature | mean importance |
|---:|---|---:|
| 1 | `s1_ax_std` | 0.0612 |
| 2 | `s1_ax_energy` | 0.0505 |
| 3 | `s1_ax_rms` | 0.0479 |
| 4 | `s1_ax_max` | 0.0432 |
| 5 | `s1_gx_mean` | 0.0409 |

이 관찰은 신호 기반 classical feature가 label, subject, filename, boundary metadata를 사용하지 않았다는 feature audit 통과 결과와 함께 봐야 한다.

## 기존 main claim을 바꿔야 할 가능성이 있는가?

수치만 보면 `feature_random_forest_v1`과 `rescnn_bigru_attention_lite_v1`이 강한 reference baseline으로 추가됐다. 그러나 기존 locked main model인 `channel_shared_posres_attention_v3`는 merged ranking에서 여전히 가장 높은 Macro F1을 유지했다.

결론적으로 main model 변경 여부는 자동으로 결정할 문제가 아니다. 논문 narrative를 neural shared encoder 제안으로 유지할지, classical feature baseline을 strong baseline으로 강조할지는 사용자가 판단해야 한다.
