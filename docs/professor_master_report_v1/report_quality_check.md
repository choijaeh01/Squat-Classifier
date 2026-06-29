# Report Quality Check

| check | status | evidence |
| --- | --- | --- |
| 새 학습 실행 없음 | PASS | script는 CSV/Markdown/PNG 생성만 수행한다. |
| CAU training 실행 없음 | PASS | CAU 접속 또는 training command를 실행하지 않았다. |
| 기존 result directory 수정 없음 | PASS | output은 docs/professor_master_report_v1 내부에만 생성된다. |
| internal prefix 노출 제한 | PASS | allowed mapping files 외 위반 0건 |
| 모든 figure link 유효 | PASS | missing links: 0 |
| Obsidian report와 repo mirror 생성 | PASS | repo mirror 생성 후 macbook rsync 검증 대상 |
| tables/figures/diagrams 존재 | PASS | expected artifact count 확인 |
| Statistical Summary MLP feature 72개 설명 | PASS | mean/std/min/max x 18 channels로 설명 |
| RF/XGBoost feature 162개 구분 | PASS | signal-derived 162 features로 설명 |
| feature importance causal evidence 금지 | PASS | feature importance를 인과 증거로 권장하지 않음 |
| identity core claim 금지 | PASS | identity를 core로 두지 않음 |
| attention core claim 금지 | PASS | attention을 core로 두지 않음 |
