# Literature Baseline Full Extension v1 Report

## 실행 개요

- 목적: Lee-style adapted CNN-LSTM과 선택된 literature/classical baseline을 final validation policy에서 3 seed full extension으로 평가
- 실행 위치: CAU 서버
- CAU hostname: `4d244d15d634`
- CAU 경로: `/home/user3/workspace/Jae/squat_imu_experiments`
- 실행 commit: `ce0245a746dea9174e24ec578a352123baf49f0b`
- Python: `3.10.12`
- Torch: `2.8.0+cu129`
- scikit-learn: `1.7.2`
- device: `cuda`, `NVIDIA RTX 6000 Ada Generation`
- 결과 경로: `results/literature_baseline_full_extension/20260617_183952_literature_baseline_full_extension_v1/`
- locked matrix reference: `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`

실행 명령:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_literature_baseline_full_extension.py \
  --config configs/literature_baseline_full_extension_v1.yaml \
  --confirm-full-extension
```

## 프로토콜

- split: `loso_with_within_train_subject_stratified_validation`
- seeds: `42`, `123`, `2025`
- folds: 6 LOSO folds
- 각 fold 크기: train 400, validation 100, test 100
- scaler: train indices 400개에만 fit, validation/test에는 transform만 적용
- per-window z-score: false
- augmentation, focal loss, balanced sampling, SSL, external dataset: 모두 disabled
- loss/optimizer: neural model은 cross entropy와 Adam
- classical baseline: 동일 split/scaler policy 아래 IMU signal-derived handcrafted feature만 사용

## 추가 모델

| model | 설명 |
|---|---|
| `feature_random_forest_v1` | IMU signal handcrafted feature 기반 RandomForest |
| `feature_linear_svm_v1` | 동일 feature 기반 LinearSVM |
| `rescnn_bigru_attention_lite_v1` | clean-room lite Residual CNN-BiGRU-Attention reference baseline |
| `tcn_literature_v1` | dilated residual TCN baseline |
| `transformer_encoder_lite_v1` | small Transformer encoder temporal baseline |
| `cnn_lstm_literature_v1` | simple 1D CNN-LSTM family representative |
| `cnn_gru_literature_v1` | simple 1D CNN-GRU family representative |
| `lee_style_cnn_lstm_2d_v1` | 40-step internal downsampling + 2D CNN + LSTM adapted baseline |

## Lee-style adapted CNN-LSTM 구현

`lee_style_cnn_lstm_2d_v1`은 Lee et al.의 exact reproduction이 아니라 현재 데이터셋과 final protocol에 맞춘 clean-room adapted baseline이다.

- 입력은 `(batch, 512, 18)` 그대로 사용한다.
- 모델 내부에서 `AdaptiveAvgPool1d`로 512 time steps를 deterministic하게 40 time steps로 downsample한다.
- downsampled matrix는 `(batch, 40, 18)`이며, 이를 `(batch, 1, 40, 18)`로 바꿔 time x channel 2D CNN에 전달한다.
- 3개의 `3x3 Conv2D -> BatchNorm2D -> ReLU -> Dropout2D` block을 사용한다.
- channel axis는 `AdaptiveAvgPool2d((40, 4))`로 축약한다.
- sequence feature를 1-layer LSTM hidden size 64에 전달하고, dense classifier로 5-class logits를 출력한다.
- Softmax는 적용하지 않는다.
- parameter count: `62,637`

Lee et al.의 3-model probability averaging이나 input-order ensemble은 이번 v1에 포함하지 않았다.

## Run 결과

- 총 run 수: 144
- 성공: 144
- 실패: 0
- skipped: 0
- `fold_metrics.csv` leakage check: 144/144 True
- `fold_metrics.csv` scaler leakage check: 144/144 True
- `scaler_fit_audit.csv` scaler leakage check: 144/144 True

## 모델별 3 seed 결과

| model | accuracy | macro F1 | weighted F1 | macro F1 CI |
|---|---:|---:|---:|---|
| `feature_random_forest_v1` | 0.8056 | 0.7845 | 0.7845 | [0.7318, 0.8376] |
| `rescnn_bigru_attention_lite_v1` | 0.7944 | 0.7691 | 0.7691 | [0.7076, 0.8270] |
| `transformer_encoder_lite_v1` | 0.7883 | 0.7580 | 0.7580 | [0.6860, 0.8188] |
| `feature_linear_svm_v1` | 0.7461 | 0.7213 | 0.7213 | [0.6612, 0.7828] |
| `tcn_literature_v1` | 0.7561 | 0.7151 | 0.7151 | [0.6090, 0.8155] |
| `lee_style_cnn_lstm_2d_v1` | 0.6622 | 0.6269 | 0.6269 | [0.5237, 0.7165] |
| `cnn_lstm_literature_v1` | 0.5756 | 0.5325 | 0.5325 | [0.4298, 0.6334] |
| `cnn_gru_literature_v1` | 0.5478 | 0.5045 | 0.5045 | [0.3901, 0.6202] |

## Class 3 Excessive Lean

| model | class 3 recall | class 3 F1 |
|---|---:|---:|
| `feature_random_forest_v1` | 0.6500 | 0.6848 |
| `rescnn_bigru_attention_lite_v1` | 0.6833 | 0.6552 |
| `feature_linear_svm_v1` | 0.6056 | 0.5800 |
| `tcn_literature_v1` | 0.6167 | 0.5650 |
| `cnn_gru_literature_v1` | 0.5944 | 0.5462 |
| `transformer_encoder_lite_v1` | 0.5056 | 0.5390 |
| `cnn_lstm_literature_v1` | 0.4889 | 0.4656 |
| `lee_style_cnn_lstm_2d_v1` | 0.4861 | 0.4607 |

## RandomForest feature importance

상위 feature는 오른쪽 허벅지 센서 `s1`의 `ax` 채널 통계에 많이 집중됐다.

| rank | feature | mean importance |
|---:|---|---:|
| 1 | `s1_ax_std` | 0.0612 |
| 2 | `s1_ax_energy` | 0.0505 |
| 3 | `s1_ax_rms` | 0.0479 |
| 4 | `s1_ax_max` | 0.0432 |
| 5 | `s1_gx_mean` | 0.0409 |
| 6 | `s1_ax_mean` | 0.0374 |
| 7 | `s1_ax_ptp` | 0.0347 |
| 8 | `s1_gy_mean` | 0.0248 |
| 9 | `s2_gy_mean` | 0.0221 |
| 10 | `s0_ax_mean` | 0.0220 |

## LinearSVM coefficient summary

상위 absolute coefficient도 `s1_ax` 관련 feature가 많이 포함됐다.

| rank | feature | mean abs coefficient |
|---:|---|---:|
| 1 | `s1_ax_energy` | 0.8335 |
| 2 | `s1_ax_max` | 0.6363 |
| 3 | `s1_az_energy` | 0.4067 |
| 4 | `s1_ay_median` | 0.3944 |
| 5 | `s0_az_energy` | 0.3943 |

## 기존 locked matrix와 병합 참고 ranking

아래 병합표는 기존 locked full matrix와 이번 literature extension을 함께 정리한 참고 ranking이다. 두 실험 모두 3 seed full protocol이지만, 모델 family와 실험 목적이 다르므로 최종 학술적 해석은 사용자가 판단해야 한다.

| rank | model | result type | macro F1 |
|---:|---|---|---:|
| 1 | `channel_shared_posres_attention_v3` | locked_3seed_full | 0.8108 |
| 2 | `feature_random_forest_v1` | literature_extension_3seed_full | 0.7845 |
| 3 | `all_channel_conv1d_small` | locked_3seed_full | 0.7740 |
| 4 | `rescnn_bigru_attention_lite_v1` | literature_extension_3seed_full | 0.7691 |
| 5 | `transformer_encoder_lite_v1` | literature_extension_3seed_full | 0.7580 |
| 6 | `all_channel_conv1d_v1` | locked_3seed_full | 0.7531 |
| 7 | `feature_linear_svm_v1` | literature_extension_3seed_full | 0.7213 |
| 8 | `tcn_literature_v1` | literature_extension_3seed_full | 0.7151 |
| 9 | `lee_style_cnn_lstm_2d_v1` | literature_extension_3seed_full | 0.6269 |

## 관찰

- `feature_random_forest_v1`은 3 seed full extension에서도 literature/classical extension 내 가장 높은 Macro F1을 기록했다.
- `rescnn_bigru_attention_lite_v1`은 neural literature baseline 중 가장 높은 Macro F1을 기록했다.
- `transformer_encoder_lite_v1`은 `rescnn_bigru_attention_lite_v1`에 가까운 수치를 보였다.
- `tcn_literature_v1`은 simple CNN-LSTM/GRU보다 높았다.
- `lee_style_cnn_lstm_2d_v1`은 simple CNN-LSTM보다 Macro F1이 높았지만, ResCNN-BiGRU-Attention-lite 및 TCN보다 낮았다.
- 최종 main result 포함 여부 및 claim 조정은 사용자가 판단해야 한다.
