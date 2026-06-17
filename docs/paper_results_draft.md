# Paper Results Draft

## 1. Experimental Setup

본 연구는 3개의 MPU-6050 IMU만을 사용해 5-class squat posture classification을 수행했다. IMU 위치는 허리, 오른쪽 허벅지, 오른쪽 종아리이며, 각 센서는 3축 accelerometer와 3축 gyroscope를 제공한다. 입력 window는 raw manually labeled boundaries를 phase-normalized linear interpolation으로 길이 512로 resampling해 구성했다. 최종 입력 형태는 `(512, 18)`이다.

평가한 모델은 all-channel Conv1D, parameter-reduced all-channel Conv1D, 2D CNN baseline, naive channel-shared encoders, 그리고 position-aware shared encoder v3이다. 모든 supervised 모델은 cross entropy loss와 Adam optimizer를 사용했고, augmentation, focal loss, balanced sampling, SSL, external dataset pretraining은 사용하지 않았다.

## 2. Evaluation Protocol

평가는 subject-independent LOSO protocol로 수행했다. 각 fold에서 1명의 subject는 test set으로 완전히 제외했다. 나머지 5명 subject의 각 class 20 windows 중 16개는 train, 4개는 validation으로 분리했다. 따라서 각 fold는 train 400 windows, validation 100 windows, test 100 windows로 구성된다.

StandardScaler는 각 fold의 train indices 400개에만 fit했다. Validation 및 test windows는 train scaler로 transform만 수행했다. Per-window z-score normalization은 사용하지 않았다. 실험은 seeds 42, 123, 2025에서 반복했으며, 총 3 seeds x 6 folds = 18 evaluations per model을 수행했다.

## 3. Model Comparison

| model | accuracy | macro F1 | 95% bootstrap CI |
|---|---:|---:|---|
| channel_shared_posres_attention_v3 | 0.8217 | 0.8108 | [0.7644, 0.8537] |
| all_channel_conv1d_small | 0.8044 | 0.7740 | [0.7026, 0.8424] |
| all_channel_conv1d_v1 | 0.7839 | 0.7531 | [0.6502, 0.8429] |
| modality_shared_sensorattn_v3 | 0.6444 | 0.6023 | [0.5003, 0.6944] |
| cnn2d_baseline_v1 | 0.4472 | 0.3688 | [0.2904, 0.4489] |
| channel_shared_attentionpool_v2 | 0.3794 | 0.3136 | [0.2553, 0.3731] |
| channel_shared_meanpool_v2 | 0.3172 | 0.2356 | [0.1780, 0.2955] |

The position-aware shared encoder achieved the highest mean macro F1 among the evaluated models. Its mean paired macro F1 difference was +0.0577 against all_channel_conv1d_v1 and +0.0368 against all_channel_conv1d_small. The bootstrap confidence intervals of these paired differences included zero, so the result should be interpreted as competitive rather than statistically superior.

## 4. Class-wise Analysis

Class 3 Excessive Lean remained difficult for several models. The naive shared encoders performed poorly on class 3, with recall 0.0917 for channel_shared_meanpool_v2 and 0.1917 for channel_shared_attentionpool_v2. In contrast, channel_shared_posres_attention_v3 reached class 3 recall 0.6889 and class 3 F1 0.7184.

This suggests that weight sharing alone is insufficient for IMU squat classification when channel identity is removed by simple pooling. Adding channel, sensor, modality, and axis identity information, together with a lightweight residual branch, substantially improved class-wise behavior.

## 5. Discussion

The results support the use of position-aware shared encoders for small-subject IMU squat classification. The v3 architecture keeps the scientific motivation of shared temporal feature extraction but avoids the severe channel identity bottleneck observed in v2 mean-pooling and attention-pooling variants.

The strongest interpretation is that channel-shared temporal encoding can be competitive with all-channel Conv1D when channel identity and sensor position are explicitly represented. The current evidence does not prove that the shared encoder is statistically superior to all-channel Conv1D, but it does show that a carefully designed shared encoder is a viable main model candidate.

## 6. Limitations

The dataset contains six subjects and 600 windows, so uncertainty remains high. The current experiment does not evaluate SSL, external IMU transfer, augmentation, focal loss, or balanced sampling. Hyperparameters were fixed rather than tuned. The paired confidence intervals against all-channel baselines include zero, so superiority claims should be avoided.

Future ablations should isolate the contribution of channel identity embeddings, sensor/modality/axis embeddings, attention pooling, and the residual branch.

## 7. Preliminary Literature Temporal Baseline Screening

`literature_baseline_screening_v1`은 기존 locked full matrix 이후 별도 extension으로 수행한다. 이 screening은 CNN-LSTM, CNN-GRU, lite Residual CNN-BiGRU-Attention, TCN, LSTM-only, GRU-only, lite Transformer encoder, 그리고 scikit-learn 사용 가능 시 handcrafted feature RandomForest/LinearSVM을 포함한다.

This section is preliminary. The main paper result table should continue to use the locked 3-seed full supervised matrix until a separate 3-seed full literature extension is explicitly approved and completed. The one-seed screening results must not be used to claim final superiority over literature temporal baselines.

## Locked Result Note

본 문서는 `paper_result_lock_v1` 이후의 locked full supervised matrix 결과를 기준으로 한다. Locked run은 `results/full_supervised_matrix/20260617_144309_full_supervised_matrix_v1/`이며, integrity check는 126 runs 성공, 0 실패, leakage/scaler check 전체 통과로 기록됐다.
