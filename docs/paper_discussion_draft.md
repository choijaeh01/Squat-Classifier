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
