# Real-data Smoke Training v1 Report

## 실행 위치와 환경

이번 실행은 full training이 아니라 real-data smoke training이다. 목적은 processed target dataset, LOSO smoke split, train-only scaler, 모델 forward/backward/optimizer 제한 루프, 결과 저장 형식이 실제 CAU 환경에서 작동하는지 확인하는 것이다.

| 항목 | 값 |
|---|---|
| 로컬 프로젝트 | `/home/jae/Projects/aiot/New_Squat/squat_imu_experiments` |
| CAU 프로젝트 | `/home/user3/workspace/Jae/squat_imu_experiments` |
| CAU hostname | `4d244d15d634` |
| CAU Python | `Python 3.10.12` |
| CAU torch | `2.8.0+cu129` |
| device | `cuda` |
| CUDA device 0 | `NVIDIA RTX 6000 Ada Generation` |
| 결과 폴더 | `results/smoke_training/20260617_094334_real_smoke_training_v1` |

CAU 실행 명령:

```bash
cd /home/user3/workspace/Jae/squat_imu_experiments
CUDA_VISIBLE_DEVICES=0 python scripts/smoke_train_real.py --config configs/real_smoke_training_v1.yaml --confirm-smoke
```

## Config 요약

- experiment: `real_smoke_training_v1`
- dataset: `data/target/processed/v1_manual_windows_resample512`
- split: `loso_smoke`
- test subject: `1`
- validation subject: `2`
- train subjects: `3`, `4`, `5`, `6`
- max epochs: `1`
- max train batches: `3`
- max validation batches: `3`
- max test batches: `3`
- batch size: `16`
- loss: `cross_entropy`
- optimizer: `adam`
- learning rate: `0.001`
- augmentation: disabled
- per-window z-score: disabled
- global scaler: enabled, fit on train subjects only

## Split 요약

| split | subjects | n | class counts |
|---|---|---:|---|
| train | `3|4|5|6` | 400 | class별 80 |
| validation | `2` | 100 | class별 20 |
| test | `1` | 100 | class별 20 |

`split_summary.csv`의 모든 모델 행에서 `leakage_check_passed=True`였다. test subject 1은 train/validation에 포함되지 않았고, validation subject 2도 train에 포함되지 않았다.

## Scaler Leakage Check

`scaler_stats.json` 기준:

- `fit_scope`: `train_subjects_only`
- `fit_sample_count`: `400`
- `fit_window_count`: `204800`

즉 scaler는 train subjects 3, 4, 5, 6의 400 windows에 대해서만 fit되었다. validation subject 2와 test subject 1은 scaler fit에 사용하지 않았다.

## 모델별 Parameter Count

| model | total | encoder | aggregation | head |
|---|---:|---:|---:|---:|
| `all_channel_conv1d_v1` | 16037 | 15712 | 0 | 325 |
| `all_channel_conv1d_small` | 4181 | 4056 | 0 | 125 |
| `channel_shared_meanpool_v2` | 2677 | 2064 | 0 | 613 |
| `channel_shared_attentionpool_v2` | 2950 | 2064 | 273 | 613 |
| `modality_shared_meanpool_v2` | 6373 | 4128 | 0 | 2245 |
| `cnn2d_baseline_v1` | 8421 | 8256 | 0 | 165 |

## Smoke Metrics

아래 수치는 smoke loop가 끝까지 실행되고 저장되는지 확인하기 위한 참고값이다. max 3 train batches, max 3 val/test batches만 사용했으므로 성능 해석에 사용하면 안 된다.

| model | status | train batches | val batches | test batches | val acc | test acc | val macro F1 | test macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `all_channel_conv1d_v1` | ok | 3 | 3 | 3 | 0.4167 | 0.4167 | 0.1176 | 0.1176 |
| `all_channel_conv1d_small` | ok | 3 | 3 | 3 | 0.1667 | 0.1667 | 0.0571 | 0.0571 |
| `channel_shared_meanpool_v2` | ok | 3 | 3 | 3 | 0.1667 | 0.1667 | 0.0571 | 0.0571 |
| `channel_shared_attentionpool_v2` | ok | 3 | 3 | 3 | 0.1667 | 0.1667 | 0.0571 | 0.0571 |
| `modality_shared_meanpool_v2` | ok | 3 | 3 | 3 | 0.4167 | 0.4167 | 0.1176 | 0.1176 |
| `cnn2d_baseline_v1` | ok | 3 | 3 | 3 | 0.1458 | 0.1667 | 0.0571 | 0.0593 |

## 학습 루프 성공 여부

모든 모델에서 제한된 smoke loop가 성공했다.

- failed model: 없음
- optimizer step: 모델별 최대 3 train batches
- epoch: 1
- validation/test: 모델별 최대 3 batches
- augmentation: 사용 안 함
- SSL: 사용 안 함
- external dataset: 사용 안 함
- full LOSO: 사용 안 함

## 저장된 결과 파일

결과 폴더에는 다음 파일이 저장되었다.

- `config_snapshot.yaml`
- `git_status.txt`
- `git_commit.txt`
- `manifest_checksums.json`
- `split_summary.csv`
- `scaler_stats.json`
- `model_parameter_counts.csv`
- `per_model_smoke_metrics.csv`
- `per_model_training_history.csv`
- `per_model_confusion_matrix.csv`
- `predictions_preview.csv`
- `run_log.txt`

## Full LOSO 전 수정 또는 확인 필요 사항

1. smoke metric은 성능 판단에 사용하지 않는다. full LOSO 설계 전에는 metric 해석을 보류한다.
2. 현재 trainer는 제한 루프용으로 충분하지만, full experiment 전에는 checkpoint 저장, fold별 config snapshot, fold별 scaler stats, fold별 confusion matrix 저장 규칙을 더 엄격히 확장해야 한다.
3. full LOSO에서는 6개 held-out subject를 모두 반복하되, 각 fold에서 scaler fit은 train subjects만 사용해야 한다.
4. validation subject 선택 정책을 fold마다 명확히 고정해야 한다. 현재 smoke에서는 subject 2를 validation으로 고정했다.
5. 성능 개선 목적으로 focal loss, mixup, Time-CutMix, augmentation, balanced sampling을 도입하지 않는다. 그런 설정은 별도 ablation으로만 검토한다.
