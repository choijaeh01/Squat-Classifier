# Codex Operating Notes

## Clean-room 원칙

- 기존 old repository 코드는 import하지 않는다.
- 기존 old repository 산출물은 명시 승인 없이는 실험 입력으로 사용하지 않는다.
- 공식 실험 결과는 이 `squat_imu_experiments/` 코드베이스에서 생성한다.

## 원본 데이터 보호

- 상위 `datasets/` 폴더의 원본 파일은 read-only로 취급한다.
- `datasets/` 안의 파일을 삭제, 이동, 수정, 덮어쓰기 하지 않는다.
- 변환 산출물은 이 프로젝트 내부 `data/target/processed/` 아래에만 생성한다.

## 자원 사용

- 인공지능 학습, 대규모 전처리, 장시간 평가처럼 컴퓨터 성능을 필요로 하는 작업은 `ssh CAU` 서버를 활용한다.
- 단, 사용자가 로컬 실행, CAU 금지, GPU 금지를 명시한 작업에서는 로컬 CPU에서 가벼운 검증만 수행한다.
- full training, epoch 단위 training, hyperparameter tuning, SSL, external adapter 구현은 명시 승인 후에만 수행한다.

## 언어와 문서

- 모든 문서와 사용자 요약은 한국어로 작성한다.
- 실험 결정, 데이터 변환, 누수 방지 규칙, 불확실성은 문서에 명확히 남긴다.

## LOSO 및 누수 방지

- subject-independent 평가는 Leave-One-Subject-Out 기준으로 수행한다.
- scaler fit은 각 LOSO fold 내부의 train subject만 사용한다.
- held-out subject는 scaler, augmentation, validation tuning, SSL pretraining에 사용하지 않는다.
- augmentation은 training window에만 적용한다.
- 성능 향상을 위해 preprocessing, label, split을 임의 변경하지 않는다.

## 재현성 기록

- 모든 실험과 변환은 config snapshot, random seed, git commit hash 또는 git 상태, manifest/data checksum을 저장한다.
- 데이터 변환 단계에서는 z-score, StandardScaler, clipping, augmentation을 적용하지 않는다.
