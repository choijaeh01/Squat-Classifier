# 데이터 인벤토리

## 전체 요약

`datasets/` 폴더는 총 195M이며, 주요 구성은 `raw/` 115M, `manually_labeled/` 80M이다. 전체 파일 수는 957개이고 확장자별로는 CSV 816개, JSON 66개, PNG 66개, `.DS_Store` 9개이다. `npy`, `npz`, `pkl`, `mat` 파일은 발견되지 않았다.

생성한 파일 단위 인벤토리는 `squat_imu_experiments/data/manifests/data_inventory.csv`에 저장했다. 이 CSV는 각 파일의 상대 경로, 확장자, 크기, 추정 역할, shape, column/key, subject/class/channel 포함 여부, 주석을 담는다.

## 폴더별 구조

| 폴더 | 파일 수 | 크기 bytes | 해석 |
|---|---:|---:|---|
| `datasets/raw/labeled` | 126 | 102274671 | class별 raw continuous session CSV 60개와 PNG sidecar 60개, 기타 메타 파일 |
| `datasets/raw/unlabeled` | 13 | 17567594 | subject별 unlabeled SSL raw continuous session CSV 6개와 PNG sidecar 6개, 기타 메타 파일 |
| `datasets/manually_labeled/class0` | 132 | 12717823 | class 0 supervised window CSV 120개, metadata JSON 12개, PNG 12개 |
| `datasets/manually_labeled/class1` | 132 | 14016455 | class 1 supervised window CSV 120개, metadata JSON 12개, PNG 12개 |
| `datasets/manually_labeled/class2` | 132 | 15618308 | class 2 supervised window CSV 120개, metadata JSON 12개, PNG 12개 |
| `datasets/manually_labeled/class3` | 132 | 14620641 | class 3 supervised window CSV 120개, metadata JSON 12개, PNG 12개 |
| `datasets/manually_labeled/class4` | 132 | 11807421 | class 4 supervised window CSV 120개, metadata JSON 12개, PNG 12개 |
| `datasets/manually_labeled/ssl` | 156 | 13144035 | subject별 unlabeled SSL window CSV 150개와 metadata JSON 6개 |

## CSV 스키마

모든 CSV 816개는 동일한 24개 header를 가진다.

`timestamp`, `seq`, `millis`, `q0`, `q1`, `q2`, `s0_ax`, `s0_ay`, `s0_az`, `s0_gx`, `s0_gy`, `s0_gz`, `s1_ax`, `s1_ay`, `s1_az`, `s1_gx`, `s1_gy`, `s1_gz`, `s2_ax`, `s2_ay`, `s2_az`, `s2_gx`, `s2_gy`, `s2_gz`

Target model 입력에 사용할 수 있는 값 채널은 `s0_*`, `s1_*`, `s2_*`의 18개 IMU 채널이다. `timestamp`, `seq`, `millis`, `q0`, `q1`, `q2`는 식별/동기화 또는 보조 정보로 남기되 기본 모델 입력에서는 제외하는 것이 맞다.

센서 위치 이름은 데이터 파일 안에서 발견되지 않았다. 현재 파일만으로는 `s0`, `s1`, `s2`가 lower back 또는 waist/torso, right thigh, right calf 중 어느 위치인지 확정할 수 없다. 공식 conversion 전에 sensor map을 별도 config로 고정해야 한다.

## 행수와 window 상태

| 그룹 | CSV 수 | row min | row max | row mean | 해석 |
|---|---:|---:|---:|---:|---|
| `raw/labeled` | 60 | 2194 | 4347 | 3084.73 | class/subject/set 단위 continuous session |
| `raw/unlabeled` | 6 | 4983 | 7088 | 6024.00 | subject 단위 SSL용 continuous session |
| `manually_labeled/class*` | 600 | 184 | 461 | 294.51 | 수동 라벨링된 squat repetition/window |
| `manually_labeled/ssl` | 150 | 167 | 316 | 225.51 | SSL용 수동 window |

중요하게도, 현재 발견된 supervised window CSV는 512행이 아니다. 따라서 `manually_labeled/class*/*_window*.csv`는 이미 잘린 supervised window이지만, target task의 `(512, 18)` 입력으로 쓰려면 resampling 또는 padding/cropping 정책이 필요하다.

## Supervised Coverage

`raw/labeled`와 `manually_labeled/class0~4` 모두 다음 coverage를 가진다.

| 항목 | 값 |
|---|---:|
| class 수 | 5 |
| subject 수 | 6 |
| set/repetition group | subject-class마다 set1, set2 |
| raw labeled session 수 | 60 |
| manually labeled supervised window 수 | 600 |
| class별 supervised window 수 | 120 |
| subject별 supervised window 수 | 100 |
| class-subject-set별 supervised window 수 | 10 |

subject-wise LOSO는 subject1부터 subject6까지 6개 fold로 구성 가능하다. class label은 `class0`부터 `class4` 폴더명에서 안정적으로 복원 가능하고, subject와 set은 파일명 `subject{n}_set{m}`에서 안정적으로 복원 가능하다.

## Target Dataset 후보

### 후보 A: `datasets/raw/labeled` + `datasets/manually_labeled/class*/subject*_set*_metadata.json`

이 후보를 1순위로 추천한다.

장점:

- raw continuous session을 원천 데이터로 사용하므로 논문에서 데이터 생성 과정을 설명하기 쉽다.
- metadata JSON에 `source_file`, `sampling_rate=100`, `method=manual`, `windows` 경계가 들어 있다.
- raw 파일의 18 IMU 채널을 명확히 선택한 뒤 metadata의 `start_sample`, `end_sample`로 window를 재구성할 수 있다.
- class, subject, set, window_id를 모두 안정적으로 복원할 수 있다.
- 수동 window CSV와 raw slicing 결과를 대조해 conversion QA가 가능하다.

한계와 불확실성:

- metadata window 길이는 167~461 sample 범위로 다양하며 512가 아니다.
- 512x18 입력 생성을 위해 resampling 또는 padding/cropping 정책을 새 코드베이스에서 명시해야 한다.
- sensor location mapping, 즉 `s0`, `s1`, `s2`가 lower back/right thigh/right calf 중 무엇인지는 데이터 파일만으로 확인되지 않는다.

누수 위험:

- raw session 전체를 먼저 표준화하거나 subject 전체 통계로 fit하면 held-out subject leakage가 생긴다.
- conversion 단계에서는 per-window resampling만 수행하고, scaler fit은 LOSO training fold 안에서만 해야 한다.

### 후보 B: `datasets/manually_labeled/class0~4/*_window*.csv`

2순위 후보이다.

장점:

- 이미 supervised repetition/window 단위로 잘려 있다.
- path만으로 class, subject, set, window_id를 복원할 수 있다.
- 600개 supervised sample이 균형 있게 존재한다.

한계:

- 이미 처리된 파생 window이므로 raw에서 어떤 방식으로 잘렸는지 설명하려면 metadata와 raw 파일을 함께 참조해야 한다.
- row length가 512가 아니므로 바로 `X.npy shape=(N,512,18)`이 되지 않는다.
- raw slicing과 window CSV가 완전히 일치하는지 conversion 전에 검증해야 한다.

누수 위험:

- window CSV 자체에는 통계 fit은 없지만, 변환 시 전체 데이터 기준 normalization을 적용하면 leakage가 된다.
- 같은 subject의 set1/set2 및 모든 class가 train/test에 섞이지 않도록 반드시 subject 기준 LOSO split을 먼저 적용해야 한다.

### 후보 C: `datasets/raw/unlabeled` 및 `datasets/manually_labeled/ssl`

현재 supervised target classification 후보로는 부적합하다.

장점:

- subject별 unlabeled IMU sequence와 SSL용 window가 있다.
- 향후 self-supervised pretraining이나 representation learning 후보로 쓸 수 있다.

한계:

- 5-class posture label이 없다.
- 논문 공식 supervised target classification 결과에는 직접 포함하면 안 된다.
- SSL 실험은 별도 계획과 adapter/manifest가 필요하다.

### 제외 대상

- PNG 파일: plot/visualization sidecar로 보이며 모델 입력이 아니다.
- `.DS_Store`: macOS 폴더 metadata로 실험에서 제외한다.
- `manually_labeled/ssl`: 현재 단계의 5-class supervised target에는 제외한다.

## 최종 추천 후보

공식 target squat classification dataset은 `datasets/raw/labeled`를 원천 데이터로 삼고, `datasets/manually_labeled/class0~4/*_metadata.json`의 window boundary를 적용해 생성하는 것을 추천한다. `datasets/manually_labeled/class0~4/*_window*.csv`는 raw slicing 결과를 검증하는 reference artifact로 사용한다.

이 방식이 가장 논문 친화적이다. 원천 raw session, manual boundary, class/subject/set/window id가 모두 설명 가능하고, 향후 conversion script가 동일한 source와 metadata에서 재현 가능한 `X.npy`, `y.npy`, `subject_id.npy`, `metadata.csv`를 만들 수 있다.

## 아직 불확실한 점

- `s0`, `s1`, `s2`의 실제 센서 위치 매핑이 데이터 안에 없다.
- 현재 window 길이가 512가 아니므로 resampling과 padding/cropping 중 어떤 정책을 공식으로 채택할지 정해야 한다.
- `q0`, `q1`, `q2`가 무엇을 의미하는지 데이터 파일만으로는 명확하지 않다. 현재 target IMU-only 18채널 입력에서는 제외하는 것이 안전하다.
- 일부 metadata JSON에는 `is_ssl` key가 없고, 일부에는 `False`가 있다. path 기준 classification metadata로 해석하면 실험에는 문제 없지만, manifest 생성 시 key 부재를 허용해야 한다.

## 데이터 누수 위험

- scaler를 전체 `X`에 fit하면 held-out subject leakage가 발생한다.
- resampling 후 per-channel normalization을 conversion 단계에서 전체 데이터 기준으로 적용하면 leakage가 발생한다.
- augmentation은 training fold에만 적용해야 한다.
- SSL/unlabeled window를 supervised target 평가 fold 안에 섞으면 실험 목적이 달라진다.
- 같은 subject의 set이나 class가 train/test 양쪽에 섞이지 않도록 split key는 반드시 `subject_id`여야 한다.

## 다음 Conversion 계획

승인 후 다음 단계에서만 conversion을 수행한다.

1. sensor map config를 만든다: `s0`, `s1`, `s2`를 lower back/right thigh/right calf 중 어느 위치로 볼지 확정한다.
2. `datasets/raw/labeled/class*/subject*_set*.csv`를 read-only로 읽는다.
3. 대응 metadata JSON의 `start_sample`, `end_sample`, `window_id`로 raw session을 slice한다.
4. 18개 IMU 채널만 선택한다.
5. 각 variable-length window를 512 time steps로 deterministic resampling한다.
6. `X.npy shape=(N,512,18)`, `y.npy shape=(N,)`, `subject_id.npy shape=(N,)`, `metadata.csv`를 `squat_imu_experiments/data/target/processed/`에 새로 저장한다.
7. window CSV와 raw-sliced conversion 결과의 row count 및 일부 numeric equality를 검증한다.
8. normalization/scaler는 conversion에서 적용하지 않고, 향후 LOSO training fold 안에서만 fit한다.
