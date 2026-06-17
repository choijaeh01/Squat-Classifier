# Paper Discussion Draft

## Main Finding

Full supervised matrix v1에서 `channel_shared_posres_attention_v3`는 평균 Macro F1 0.8108로 가장 높은 값을 기록했다. `all_channel_conv1d_v1`은 0.7531, `all_channel_conv1d_small`은 0.7740이었다. Paired difference는 v3가 all-channel baselines보다 양수였지만 confidence interval이 0을 포함했다.

따라서 본 결과는 v3가 all-channel Conv1D보다 통계적으로 우수하다는 결론이 아니라, channel identity를 보존한 shared encoder가 strong all-channel baselines와 경쟁 가능한 성능을 보였다는 결론으로 해석해야 한다.

## Why v2 Failed

Naive shared v2 모델은 encoder object를 공유했지만 channel token들을 단순 mean pooling 또는 작은 attention pooling으로 압축했다. 이 과정에서 sensor 위치, axis, modality identity가 충분히 보존되지 않았을 가능성이 크다. 특히 Class 3 Excessive Lean에서 recall이 매우 낮았다.

## Why v3 Helped

v3는 shared temporal encoder를 유지하면서도 channel ID, sensor ID, modality ID, axis ID를 token에 명시적으로 제공한다. Residual branch는 raw channel-level summary를 보존한다. 이 두 장치가 channel identity bottleneck을 완화해 v2 대비 큰 개선을 만든 것으로 해석할 수 있다.

## Class 3 Excessive Lean

Class 3은 여전히 어려운 class다. 그러나 v3는 class 3 recall 0.6889, F1 0.7184를 기록해 v2 meanpool 및 attentionpool보다 크게 개선됐다. 이는 trunk lean과 관련된 sensor-position-specific features가 중요할 수 있음을 시사한다.

## Limitations

Subject 수는 6명으로 작다. External dataset transfer, SSL, augmentation, focal loss는 본 실험에서 사용하지 않았다. Hyperparameter tuning도 수행하지 않았다. 따라서 본 논문은 supervised target dataset에서의 architecture comparison으로 범위를 제한한다.

## Future Work

후속 연구에서는 v3 ablation, external IMU dataset transfer, self-supervised pretraining을 검토할 수 있다. 단, 현재 논문 단계에서는 결과 lock 이후 추가 실험보다 manuscript draft를 먼저 정리하는 것이 적절하다.

## Literature Baseline Full Extension

Literature baseline full extension v1에서는 RandomForest, LinearSVM, ResCNN-BiGRU-Attention-lite, TCN, Transformer, simple CNN-LSTM/GRU, Lee-style adapted CNN-LSTM을 같은 final LOSO protocol에서 추가 평가했다. 모든 run은 3 seeds x 6 folds로 실행됐고 leakage/scaler check를 통과했다.

수치상 `feature_random_forest_v1`은 Macro F1 0.7845로 extension 내 가장 높았고, `rescnn_bigru_attention_lite_v1`은 Macro F1 0.7691로 neural temporal baseline 중 가장 높았다. 기존 locked `channel_shared_posres_attention_v3`는 Macro F1 0.8108로 merged ranking에서 가장 높았다.

이 결과는 두 가지 논의점을 만든다. 첫째, handcrafted feature classical baseline이 small-subject IMU dataset에서 강할 수 있으므로 neural model claim은 strong classical baseline을 함께 제시해야 한다. 둘째, Lee-style adapted CNN-LSTM은 simple CNN-LSTM보다 높았지만 여전히 TCN 및 ResCNN-BiGRU-Attention-lite보다 낮았으므로, CNN-LSTM family 자체가 본 데이터에서 항상 강한 것은 아니다.

최종 논문에서 이 extension을 main table에 포함할지, appendix 또는 additional comparison으로 둘지는 사용자 판단으로 남긴다.

## v3 Component Ablation

v3 component ablation v1은 original `channel_shared_posres_attention_v3`의 구조적 요소를 제거한 4개 variant를 동일 final LOSO protocol에서 평가했다. 기존 locked full matrix와 literature full extension 결과 디렉터리는 수정하지 않았다.

가장 큰 차이는 residual branch 제거에서 나타났다. `channel_shared_posres_attention_v3_no_residual`은 Macro F1 0.5937로 original v3의 0.8108보다 크게 낮았다. Paired delta는 -0.2171이고 confidence interval은 [-0.2945, -0.1449]였다. 이는 residual summary branch가 현재 dataset에서 중요한 signal을 제공했을 가능성을 보여준다.

Identity embedding 제거 variant는 Macro F1 0.7675로 original v3보다 낮았다. Paired delta는 -0.0434이고 confidence interval은 [-0.0852, -0.0072]였다. 단, 이 variant에는 residual branch가 남아 있으므로 channel-position cue를 완전히 제거한 실험은 아니다.

Attention pooling을 mean pooling으로 대체한 variant는 Macro F1 0.8082로 original v3와 매우 가까웠다. Paired delta는 -0.0026이고 confidence interval은 [-0.0402, 0.0343]였다. 따라서 attention pooling을 v3 성능의 핵심 원인으로 강하게 주장하기는 어렵다. 더 안전한 논의는 attention pooling을 learnable aggregation option으로 설명하고, position-aware identity 및 residual summary branch를 더 중요한 component로 다루는 것이다.

Residual-only MLP는 Macro F1 0.8036으로 original v3와 가까웠다. 이는 small-subject IMU squat dataset에서 channel-wise summary statistics가 강한 baseline signal을 가질 수 있음을 보여준다. 이 결과는 proposed model의 shared encoder claim을 과장하지 않도록 만드는 중요한 limitation이다.

Attention-RF alignment 분석에서는 v3 attention weight와 RandomForest feature importance가 모두 `s1_ax` 관련 정보를 상위로 보았고, channel-level Pearson correlation은 0.6361이었다. 그러나 attention weight와 feature importance는 계산 의미가 다르므로 동일한 해석 지표로 간주하지 않는다.

## v3 Component Ablation

v3 component ablation v1에서는 residual branch 제거, residual-only MLP, identity embedding 제거, attention 제거 mean-pooling variant를 3 seeds x 6 folds로 평가했다. 모든 run은 동일 final validation policy와 train-only scaler policy를 사용했고, augmentation, focal loss, balanced sampling, SSL, external dataset은 사용하지 않았다.

가장 큰 변화는 residual branch 제거에서 나타났다. `channel_shared_posres_attention_v3_no_residual`의 Macro F1은 0.5937로 original v3의 0.8108보다 낮았고, paired delta는 -0.2171이었다. 이는 residual branch가 본 데이터셋에서 중요한 역할을 했을 가능성을 시사한다.

반면 `channel_shared_posres_attention_v3_residual_only_mlp`는 Macro F1 0.8036, `channel_shared_posres_meanpool_v3_no_attention`은 0.8082로 original v3와 매우 가까웠다. 따라서 논문에서는 attention pooling 자체가 성능 향상의 핵심 원인이라고 단정하기보다, channel-wise residual summary와 position-aware token representation의 결합을 보수적으로 설명하는 편이 안전하다.

`no_identity`는 Macro F1 0.7675로 original v3보다 낮았다. 그러나 residual branch가 channel-order 기반 summary를 계속 사용하므로, 이 실험은 token identity embedding의 제거를 의미할 뿐 모든 channel position cue 제거를 의미하지 않는다.

Attention-RF alignment에서는 v3 attention과 RandomForest channel importance 사이의 Pearson correlation이 0.6361, Spearman correlation이 0.3870으로 관찰됐다. `s1_ax`는 두 분석 모두에서 중요하게 나타났지만, attention weight와 feature importance는 동일한 의미가 아니므로 정성적 참고로만 사용한다.
