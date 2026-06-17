# Model Architecture Diagrams

Netron 기반 TorchScript 구조도와 PNG 캡처는 [Netron Model Visualization v1](netron_model_visualization.md)에 정리하였다. 해당 산출물은 `results/model_diagrams/netron_exports_v1/` 아래에 있으며, 학습된 checkpoint가 아닌 initialized architecture export이다.

## `all_channel_conv1d_v1`

```mermaid
flowchart LR
  X["Input X<br/>(batch, 512, 18)"] --> T["Transpose<br/>(batch, 18, 512)"]
  T --> C1["Conv1D 18 -> 32<br/>BN + ReLU"]
  C1 --> C2["Conv1D 32 -> 64<br/>BN + ReLU"]
  C2 --> P["AdaptiveAvgPool1D"]
  P --> H["Linear 64 -> 5"]
  H --> Y["Logits<br/>(batch, 5)"]
```

이 모델은 18채널을 한 번에 입력받는다. 채널 간 결합을 Conv1D encoder 내부에서 자유롭게 학습한다.

## `channel_shared_meanpool_v2`

```mermaid
flowchart LR
  X["Input X<br/>(batch, 512, 18)"] --> S["Split into 18 single-channel sequences"]
  S --> E["One shared TemporalEncoder1D object<br/>reused for every channel"]
  E --> M["Embeddings<br/>(batch, 18, 32)"]
  M --> P["Mean pool over channels<br/>(batch, 32)"]
  P --> H["Small head<br/>Linear 32 -> 16 -> 5"]
  H --> Y["Logits<br/>(batch, 5)"]
```

weight sharing은 `shared_encoder` 하나를 18개 channel에 반복 적용하는 지점에서 일어난다. channel별 encoder를 따로 만들지 않는다.

## `modality_shared_meanpool_v2`

```mermaid
flowchart LR
  X["Input X<br/>(batch, 512, 18)"] --> A["Accelerometer channels<br/>9 one-channel sequences"]
  X --> G["Gyroscope channels<br/>9 one-channel sequences"]
  A --> AE["One acc shared TemporalEncoder1D object"]
  G --> GE["One gyro shared TemporalEncoder1D object"]
  AE --> AP["Mean pool acc embeddings<br/>(batch, 32)"]
  GE --> GP["Mean pool gyro embeddings<br/>(batch, 32)"]
  AP --> C["Concat<br/>(batch, 64)"]
  GP --> C
  C --> H["Small head<br/>Linear 64 -> 32 -> 5"]
  H --> Y["Logits<br/>(batch, 5)"]
```

acc encoder와 gyro encoder는 서로 다른 object이다. 각 modality 내부 channel은 해당 modality encoder 하나를 공유한다.

## `cnn2d_baseline_v1`

```mermaid
flowchart LR
  X["Input X<br/>(batch, 512, 18)"] --> U["Unsqueeze<br/>(batch, 1, 512, 18)"]
  U --> C1["Conv2D 1 -> 16<br/>kernel time x channel"]
  C1 --> C2["Conv2D 16 -> 32<br/>BN + ReLU"]
  C2 --> P["AdaptiveAvgPool2D"]
  P --> H["Linear 32 -> 5"]
  H --> Y["Logits<br/>(batch, 5)"]
```

이 모델은 time x channel grid를 2D 입력처럼 다루는 baseline이다.

## `channel_shared_posres_attention_v3`

```mermaid
flowchart LR
  X["Input X<br/>(batch, 512, 18)"] --> S["Split into 18 single-channel sequences"]
  S --> E["One shared TemporalEncoder1D object<br/>reused for all channels"]
  E --> T["Channel token embeddings<br/>(batch, 18, 32)"]
  T --> M["Add channel, sensor, modality, axis embeddings"]
  M --> A["Small attention pooling<br/>over 18 tokens"]
  X --> R["Residual branch<br/>per-channel mean, std, min, max"]
  R --> RP["Small residual projection"]
  A --> C["Concat shared attention rep + residual rep"]
  RP --> C
  C --> H["Classifier<br/>Linear 64 -> 64 -> 5"]
  H --> Y["Logits<br/>(batch, 5)"]
```

이 모델은 encoder weight sharing을 유지하면서 channel identity를 보존한다. residual branch는 shared encoder가 공통 feature를 학습하는 동안 channel별 raw summary 정보를 잃지 않도록 한다.

## `modality_shared_sensorattn_v3`

```mermaid
flowchart LR
  X["Input X<br/>(batch, 512, 18)"] --> A["Accelerometer channels<br/>9 sequences"]
  X --> G["Gyroscope channels<br/>9 sequences"]
  A --> AE["One acc shared TemporalEncoder1D object"]
  G --> GE["One gyro shared TemporalEncoder1D object"]
  AE --> AT["Acc tokens + sensor, modality, axis embeddings"]
  GE --> GT["Gyro tokens + sensor, modality, axis embeddings"]
  AT --> AP["Acc attention pooling"]
  GT --> GP["Gyro attention pooling"]
  AP --> C["Concat acc + gyro representations"]
  GP --> C
  C --> F["Small gated fusion"]
  F --> H["Classifier<br/>Linear 64 -> 32 -> 5"]
  H --> Y["Logits<br/>(batch, 5)"]
```

이 모델은 acc/gyro modality별 encoder를 분리하고, 각 modality 내부 channel을 sensor-aware attention으로 집계한다. 기존 mean pooling v2와 달리 modality별 token 중요도를 학습할 수 있다.
