# Overfit Diagnostic v2 With v3 Report

## 목적

이 실험은 일반화 성능 평가가 아니다. v3 모델이 작은 balanced training subset을 외울 수 있는지 확인하는 capacity sanity check이다.

- 실행 위치: CAU
- 결과 경로: `results/overfit_diagnostics/20260617_130057_overfit_diagnostic_v2_with_v3/`
- config: `configs/overfit_diagnostic_v2_with_v3.yaml`
- device: `cuda`
- dataset: `data/target/processed/v1_manual_windows_resample512`
- subset: subjects 3, 4, 5, 6에서 class별 8개, 총 40개
- scaler: selected subset only
- augmentation, focal loss, SSL, external dataset: disabled

## 결과

| model | reached 95% train acc | epoch | final train acc | final train macro F1 |
|---|---|---:|---:|---:|
| all_channel_conv1d_v1 | true | 10 | 0.9500 | 0.9479 |
| all_channel_conv1d_small | true | 18 | 0.9500 | 0.9479 |
| channel_shared_meanpool_v2 | false | - | 0.5250 | 0.4511 |
| channel_shared_attentionpool_v2 | false | - | 0.6250 | 0.6127 |
| modality_shared_meanpool_v2 | false | - | 0.7500 | 0.7248 |
| channel_shared_posres_attention_v3 | true | 30 | 0.9500 | 0.9511 |
| modality_shared_sensorattn_v3 | true | 87 | 0.9500 | 0.9499 |

## 해석

v2 shared 모델들은 150 epoch 안에 95% train accuracy에 도달하지 못했다. 반면 두 v3 모델은 모두 도달했다.

`channel_shared_posres_attention_v3`는 30 epoch에 target에 도달했다. 이는 channel identity embedding과 residual branch가 v2의 underfit 문제를 크게 완화했음을 시사한다.

`modality_shared_sensorattn_v3`도 87 epoch에 target에 도달했다. acc/gyro 분리, sensor-aware attention, gated fusion이 기존 modality mean pooling보다 capacity를 개선했다.

주의: 이 결과는 같은 40개 subset에서의 memorization sanity check다. held-out subject 일반화 성능으로 해석하지 않는다.
