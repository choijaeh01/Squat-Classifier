# Netron Model Visualization v1

## 목적

이 문서는 논문용 full supervised matrix v1에 포함된 7개 모델의 구조를 Netron으로 확인하기 위한 산출물을 정리한다. 모든 export는 구조 도식화 목적의 initialized model이다. 학습된 checkpoint를 불러오지 않았고, 새 학습도 실행하지 않았다.

Netron은 neural network와 deep learning model viewer이며 TorchScript 파일을 지원한다. 따라서 현재 PyTorch registry 모델을 `(1, 512, 18)` synthetic input으로 trace하여 TorchScript 파일로 저장하고, Netron 화면을 PNG로 캡처하였다.

## 산출물 위치

- Netron export root: `results/model_diagrams/netron_exports_v1/`
- TorchScript files: `results/model_diagrams/netron_exports_v1/torchscript/`
- Netron screenshots: `results/model_diagrams/netron_exports_v1/screenshots/`
- Model index: `results/model_diagrams/netron_exports_v1/netron_model_index.csv`
- Export manifest: `results/model_diagrams/netron_exports_v1/netron_export_manifest.json`

## 재생성 명령

```bash
python scripts/export_netron_models.py --config configs/netron_model_export_v1.yaml
```

Netron에서 개별 모델을 열 때:

```bash
.venv/bin/netron results/model_diagrams/netron_exports_v1/torchscript/channel_shared_posres_attention_v3.torchscript.pt
```

## Export 대상 모델

| model | params | TorchScript file | Netron screenshot |
|---|---:|---|---|
| all_channel_conv1d_v1 | 16037 | `all_channel_conv1d_v1.torchscript.pt` | `all_channel_conv1d_v1_netron.png` |
| all_channel_conv1d_small | 4181 | `all_channel_conv1d_small.torchscript.pt` | `all_channel_conv1d_small_netron.png` |
| cnn2d_baseline_v1 | 8421 | `cnn2d_baseline_v1.torchscript.pt` | `cnn2d_baseline_v1_netron.png` |
| channel_shared_meanpool_v2 | 2677 | `channel_shared_meanpool_v2.torchscript.pt` | `channel_shared_meanpool_v2_netron.png` |
| channel_shared_attentionpool_v2 | 2950 | `channel_shared_attentionpool_v2.torchscript.pt` | `channel_shared_attentionpool_v2_netron.png` |
| channel_shared_posres_attention_v3 | 14742 | `channel_shared_posres_attention_v3.torchscript.pt` | `channel_shared_posres_attention_v3_netron.png` |
| modality_shared_sensorattn_v3 | 16103 | `modality_shared_sensorattn_v3.torchscript.pt` | `modality_shared_sensorattn_v3_netron.png` |

## Figure Candidates and Captions

### Figure N1. all_channel_conv1d_v1

![all_channel_conv1d_v1 Netron](../results/model_diagrams/netron_exports_v1/screenshots/all_channel_conv1d_v1_netron.png)

- Korean caption: `all_channel_conv1d_v1`의 Netron 구조도. 18개 IMU channel을 하나의 Conv1D encoder에 함께 입력하여 channel 간 결합을 직접 학습한다.
- English caption: Netron visualization of `all_channel_conv1d_v1`. The model jointly processes all 18 IMU channels with a single all-channel Conv1D encoder.
- 핵심 메시지: 가장 기본적인 all-channel baseline이며 shared encoder가 아니다.

### Figure N2. all_channel_conv1d_small

![all_channel_conv1d_small Netron](../results/model_diagrams/netron_exports_v1/screenshots/all_channel_conv1d_small_netron.png)

- Korean caption: `all_channel_conv1d_small`의 Netron 구조도. all-channel Conv1D 구조를 유지하되 channel_shared 모델과 비교하기 위해 parameter budget을 줄인 baseline이다.
- English caption: Netron visualization of `all_channel_conv1d_small`, a reduced-capacity all-channel Conv1D baseline for parameter-aware comparison.
- 핵심 메시지: v3와 비교할 강한 small baseline이다.

### Figure N3. cnn2d_baseline_v1

![cnn2d_baseline_v1 Netron](../results/model_diagrams/netron_exports_v1/screenshots/cnn2d_baseline_v1_netron.png)

- Korean caption: `cnn2d_baseline_v1`의 Netron 구조도. time by channel 입력을 2D grid로 취급하여 Conv2D baseline을 구성한다.
- English caption: Netron visualization of `cnn2d_baseline_v1`, which treats the time-channel IMU window as a two-dimensional input grid.
- 핵심 메시지: 2D time-channel baseline으로, proposed shared encoder와는 다른 inductive bias를 가진다.

### Figure N4. channel_shared_meanpool_v2

![channel_shared_meanpool_v2 Netron](../results/model_diagrams/netron_exports_v1/screenshots/channel_shared_meanpool_v2_netron.png)

- Korean caption: `channel_shared_meanpool_v2`의 Netron 구조도. 하나의 shared 1D encoder를 18개 channel에 반복 적용한 뒤 channel mean pooling으로 집계한다.
- English caption: Netron visualization of `channel_shared_meanpool_v2`, which reuses one shared 1D encoder across 18 channels and aggregates token embeddings by mean pooling.
- 핵심 메시지: parameter는 작지만 channel identity를 거의 제거하는 naive shared 구조이다.

### Figure N5. channel_shared_attentionpool_v2

![channel_shared_attentionpool_v2 Netron](../results/model_diagrams/netron_exports_v1/screenshots/channel_shared_attentionpool_v2_netron.png)

- Korean caption: `channel_shared_attentionpool_v2`의 Netron 구조도. shared encoder는 유지하고 mean pooling 대신 작은 attention scorer로 channel token을 집계한다.
- English caption: Netron visualization of `channel_shared_attentionpool_v2`, which keeps the shared channel encoder but replaces mean pooling with a lightweight attention scorer.
- 핵심 메시지: attention pooling만으로는 위치 정보 손실을 충분히 보완하지 못했던 v2 비교 모델이다.

### Figure N6. channel_shared_posres_attention_v3

![channel_shared_posres_attention_v3 Netron](../results/model_diagrams/netron_exports_v1/screenshots/channel_shared_posres_attention_v3_netron.png)

- Korean caption: `channel_shared_posres_attention_v3`의 Netron 구조도. 하나의 shared 1D encoder를 유지하면서 channel, sensor, modality, axis embedding과 residual summary branch를 결합한다.
- English caption: Netron visualization of `channel_shared_posres_attention_v3`, the proposed position-aware channel-shared encoder with identity embeddings, attention aggregation, and a residual summary branch.
- 핵심 메시지: proposed model 후보이다. encoder sharing을 유지하면서 channel identity bottleneck을 완화한다.

### Figure N7. modality_shared_sensorattn_v3

![modality_shared_sensorattn_v3 Netron](../results/model_diagrams/netron_exports_v1/screenshots/modality_shared_sensorattn_v3_netron.png)

- Korean caption: `modality_shared_sensorattn_v3`의 Netron 구조도. accelerometer와 gyroscope channel에 서로 다른 shared encoder를 적용하고 modality별 sensor-aware attention으로 집계한다.
- English caption: Netron visualization of `modality_shared_sensorattn_v3`, which uses separate accelerometer and gyroscope shared encoders with sensor-aware attention and gated fusion.
- 핵심 메시지: acc와 gyro를 분리한 modality-aware shared encoder 비교 모델이다.

## 주의 사항

- 이 산출물은 구조 확인용이다. 학습된 checkpoint 성능이나 final metric을 포함하지 않는다.
- TorchScript tracing은 `(1, 512, 18)` 입력에 맞춘 정적 graph를 만든다. Netron 구조 확인에는 적합하지만, 임의 shape 일반화를 보장하는 export가 아니다.
- `results/` 하위 산출물은 gitignore 대상이므로, 논문 작성 시 필요한 PNG만 별도 manuscript asset 폴더로 선별 복사해야 한다.
