# Model Architecture Diagrams

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
