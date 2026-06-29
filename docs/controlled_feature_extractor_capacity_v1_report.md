# controlled_feature_extractor_capacity_v1 Report

## 목적

이 보고서는 controlled feature extractor comparison v1의 capacity audit이다. 학습, 역전파, optimizer step은 수행하지 않았다.

## 공통 Head 검증

- controlled neural 모델 수: 9
- 공통 head parameter count 후보: 4485
- 공통 head 구조 후보: linear:64->64 &#124; relu &#124; dropout:0.1 &#124; linear:64->5
- 모든 controlled neural 모델은 64차원 representation을 같은 MLP head에 입력하도록 설계했다.

## Parameter Count

| model | group | total params | extractor params | common head params | representation dim | notes |
|---|---|---:|---:|---:|---:|---|
| controlled_flatten_mlp | controlled_neural | 594373 | 589888 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_stats_mlp | controlled_neural | 9157 | 4672 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_all_channel_1d_cnn | controlled_neural | 20197 | 15712 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_all_channel_1d_cnn_small | controlled_neural | 11317 | 6832 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_shared_1d | controlled_neural | 7093 | 2608 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_shared_1d_identity | controlled_neural | 8885 | 4400 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_shared_1d_residual | controlled_neural | 20021 | 15536 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_shared_1d_residual_identity | controlled_neural | 21813 | 17328 | 4485 | 64 | controlled neural model with common classifier head |
| controlled_2d_cnn | controlled_neural | 14853 | 10368 | 4485 | 64 | controlled neural model with common classifier head |
| feature_random_forest_v1 | classical_practical |  |  |  | not_applicable | classical baseline; available=False |
| feature_xgboost_v1 | classical_practical |  |  |  | not_applicable | classical baseline; available=False |
| feature_linear_svm_v1 | classical_practical |  |  |  | not_applicable | classical baseline; available=False |
| rescnn_bigru_attention_lite_v1 | literature_reference | 38934 |  |  |  | literature reference model |
| lee_style_cnn_lstm_2d_v1 | literature_reference | 62637 |  |  |  | literature reference model |
| channel_shared_posres_attention_v3 | locked_reference | 14742 |  | not_applicable |  | read-only locked matrix reference architecture |
| all_channel_conv1d_v1 | locked_reference | 16037 |  | not_applicable |  | read-only locked matrix reference architecture |
| all_channel_conv1d_small | locked_reference | 4181 |  | not_applicable |  | read-only locked matrix reference architecture |

## 해석 범위

- 이 단계는 extractor 구조만 통제 비교하기 위한 사전 점검이다.
- common head가 같기 때문에 controlled neural 내부 비교에서는 classifier head 차이를 주요 confounder에서 제외할 수 있다.
- classical baseline은 train fold 내부 IMU signal-derived feature만 사용하며 parameter count는 신경망 parameter count와 직접 비교하지 않는다.
- XGBoost는 설치되어 있지 않으면 training 단계에서 skipped 처리한다.
- locked reference 모델은 기존 결과 디렉터리를 수정하지 않고 구조 비교 참고용으로만 포함했다.
