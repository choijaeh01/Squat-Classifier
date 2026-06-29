# Paper Results Draft v2

## Main Supervised Matrix

Locked full supervised matrix v1은 3 seeds, 6 LOSO folds, 7 models로 수행되었다. 모든 run에서 leakage check와 scaler leakage check가 통과했다. 평균 Macro F1 기준으로 `channel_shared_posres_attention_v3`가 0.8108로 가장 높았고, `all_channel_conv1d_small`은 0.7740, `all_channel_conv1d_v1`은 0.7531이었다.

그러나 paired difference에서 `channel_shared_posres_attention_v3`와 all-channel baselines의 confidence interval은 0을 포함했다. 따라서 제안 모델이 all-channel baseline보다 통계적으로 유의하게 우월하다고 주장하지 않고, 경쟁 가능한 성능을 보였다고 서술한다.

## Literature and Classical Baselines

Literature full extension에서는 RandomForest, LinearSVM, ResCNN-BiGRU-Attention-lite, TCN, Transformer, CNN-LSTM 계열 baseline을 같은 final validation policy로 평가하였다. RandomForest와 ResCNN-BiGRU-Attention-lite는 강한 reference baseline으로 관찰되었다. Lee-style adapted CNN-LSTM은 simple CNN-LSTM보다 개선되었지만, top neural/shared 또는 RandomForest 수준에는 미치지 못했다.

이 baseline들은 old repository 결과 재사용이 아니라 clean-room codebase에서 동일 split/scaler protocol로 새로 평가한 결과이다.

## Controlled Feature Extractor Comparison

Controlled feature extractor comparison은 classifier head를 고정하여 extractor 차이를 확인하기 위한 추가 분석이다. 모든 controlled neural model은 64차원 representation과 동일한 MLP head를 사용했다.

| model | accuracy | macro F1 | macro F1 CI |
|---|---:|---:|---|
| controlled_stats_mlp | 0.8250 | 0.8174 | [0.7744, 0.8584] |
| controlled_shared_1d_residual | 0.8094 | 0.8004 | [0.7423, 0.8540] |
| controlled_all_channel_1d_cnn | 0.8250 | 0.7994 | [0.7404, 0.8564] |
| controlled_shared_1d_residual_identity | 0.8072 | 0.7973 | [0.7397, 0.8493] |
| feature_random_forest_v1 | 0.8056 | 0.7845 | [0.7318, 0.8376] |
| rescnn_bigru_attention_lite_v1 | 0.7944 | 0.7691 | [0.7076, 0.8270] |
| controlled_all_channel_1d_cnn_small | 0.7806 | 0.7562 | [0.6786, 0.8242] |
| feature_linear_svm_v1 | 0.7461 | 0.7213 | [0.6612, 0.7828] |
| controlled_flatten_mlp | 0.7344 | 0.7046 | [0.5953, 0.8074] |
| lee_style_cnn_lstm_2d_v1 | 0.6622 | 0.6269 | [0.5237, 0.7165] |
| controlled_shared_1d_identity | 0.5617 | 0.5182 | [0.4335, 0.5968] |
| controlled_2d_cnn | 0.5056 | 0.4452 | [0.3472, 0.5413] |
| controlled_shared_1d | 0.3500 | 0.2806 | [0.2233, 0.3385] |

이 결과에서 summary statistics MLP가 가장 높았고, residual shared 1D가 all-channel controlled CNN과 유사한 수준을 보였다. Naive shared 1D는 낮았으며, residual branch 추가 후 큰 폭으로 회복되었다.

## XGBoost Completion

`feature_xgboost_v1`은 원래 controlled comparison에서 dependency 부재로 skipped되었기 때문에, 동일 split/scaler/feature audit 조건에서 별도 completion run으로 실행했다. 기존 controlled result directory는 수정하지 않았다.

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| feature_xgboost_v1 | 0.8156 | 0.7961 | 0.7961 | [0.7567, 0.8328] |

Paired Macro F1 difference는 XGBoost minus reference 기준으로 RF 대비 +0.0116, controlled stats MLP 대비 -0.0212, controlled shared residual 대비 -0.0043, controlled all-channel CNN 대비 -0.0033이었다. 모든 CI는 0을 포함했다. 따라서 XGBoost를 포함하더라도 특정 baseline 대비 명확한 우월성 주장은 하지 않는다.

XGBoost 상위 feature는 `s1_ax_std`, `s1_gx_mean`, `s1_ax_energy`였다. 이는 RandomForest의 `s1_ax` 중심 feature importance와 방향이 유사하지만, feature importance는 모델 내부 기준이므로 생체역학적 인과 설명으로 직접 사용하지 않는다.

## Component Analysis

Naive shared 1D encoder는 Macro F1 0.2806으로 낮았다. Identity embedding만 추가하면 0.5182로 상승했지만, residual branch를 추가하면 0.8004로 크게 상승했다. Residual+identity는 0.7973으로 residual-only와 거의 같은 수준이었다.

이 결과는 단순 weight sharing 자체보다, shared encoder가 놓칠 수 있는 channel-specific signal을 residual branch로 보존하는 것이 중요하다는 해석을 가능하게 한다.

## Class-wise Analysis

Class 3 Excessive Lean은 초기 pilot에서 어려운 class로 관찰되었다. Controlled comparison에서는 residual shared 계열이 Class 3에서 높은 F1을 보였다.

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| controlled_shared_1d_residual | 0.7222 | 0.7542 |
| controlled_shared_1d_residual_identity | 0.7417 | 0.7509 |
| controlled_stats_mlp | 0.6917 | 0.7150 |
| feature_random_forest_v1 | 0.6500 | 0.6848 |
| controlled_all_channel_1d_cnn | 0.6806 | 0.6692 |
| rescnn_bigru_attention_lite_v1 | 0.6833 | 0.6552 |
| lee_style_cnn_lstm_2d_v1 | 0.4861 | 0.4607 |
| controlled_shared_1d | 0.1556 | 0.1573 |

## Feature Importance Observation

RandomForest의 상위 feature는 오른쪽 허벅지 `s1_ax`의 std, energy, RMS, max, mean, peak-to-peak에 집중되었다. 이는 현재 dataset에서 thigh acceleration summary가 class 구분에 강한 정보를 담고 있음을 보여준다. 단, feature importance는 인과적 설명이 아니며 생체역학적 결론으로 바로 확장하지 않는다.

XGBoost completion에서도 오른쪽 허벅지 `s1` 관련 feature가 상위에 나타났다. 이는 hand-crafted signal summary baseline이 현재 데이터에서 강하다는 관찰을 보강한다.

## Results Statement Draft

영문 초안:

> In the locked supervised matrix, the proposed position-aware channel-shared encoder achieved the highest mean Macro F1 among the neural models, while paired confidence intervals against all-channel Conv1D baselines included zero. A controlled feature-extractor comparison further showed that the residual branch substantially improved the naive shared encoder, bringing shared 1D extractors close to all-channel Conv1D under a common classifier head.

국문 해석:

> 최종 supervised matrix에서 제안한 position-aware channel-shared encoder는 neural model 중 가장 높은 평균 Macro F1을 기록했다. 다만 all-channel Conv1D와의 paired confidence interval이 0을 포함하므로 통계적 우월성은 주장하지 않는다. 추가 controlled comparison에서는 residual branch가 naive shared encoder의 성능 저하를 크게 완화하여, 공통 classifier head 조건에서 shared 1D extractor가 all-channel 1D CNN과 유사한 수준에 도달했다.
