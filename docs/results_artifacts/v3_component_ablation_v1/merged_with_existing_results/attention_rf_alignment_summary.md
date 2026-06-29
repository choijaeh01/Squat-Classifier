# Attention-RF Alignment Summary

이 분석은 기존 v3 attention weight와 RandomForest feature importance를 사용한 inference-only/post-hoc 비교다. Attention weight와 feature importance는 같은 의미가 아니므로 정성적 alignment로만 해석한다.

- channel-level Pearson correlation: 0.6361
- channel-level Spearman correlation: 0.3870
- RF by-fold importance file: available

## Group Aggregate

| group type | group name | v3 attention | RF sum importance |
|---|---|---:|---:|
| axis | x | 0.0815 | 0.4766 |
| axis | y | 0.0619 | 0.2597 |
| axis | z | 0.0232 | 0.2636 |
| modality | acc | 0.1076 | 0.6507 |
| modality | gyro | 0.0035 | 0.3493 |
| sensor | s0_lower_back | 0.0602 | 0.2453 |
| sensor | s1_right_thigh | 0.0761 | 0.5240 |
| sensor | s2_right_calf | 0.0304 | 0.2306 |

## Class-wise Top Attention Channels

| class | rank | channel | attention |
|---:|---:|---|---:|
| 0 | 1 | s1_ax | 0.3717 |
| 0 | 2 | s0_ay | 0.2541 |
| 0 | 3 | s1_az | 0.0782 |
| 0 | 4 | s2_ay | 0.0705 |
| 0 | 5 | s0_ax | 0.0644 |
| 1 | 1 | s1_ax | 0.3333 |
| 1 | 2 | s0_ay | 0.2680 |
| 1 | 3 | s2_ay | 0.0791 |
| 1 | 4 | s1_az | 0.0647 |
| 1 | 5 | s0_ax | 0.0629 |
| 2 | 1 | s1_ax | 0.3565 |
| 2 | 2 | s0_ay | 0.2352 |
| 2 | 3 | s1_ay | 0.1039 |
| 2 | 4 | s1_az | 0.0741 |
| 2 | 5 | s0_ax | 0.0682 |
| 3 | 1 | s1_ax | 0.2972 |
| 3 | 2 | s0_ay | 0.2421 |
| 3 | 3 | s0_ax | 0.1796 |
| 3 | 4 | s1_az | 0.0921 |
| 3 | 5 | s2_ax | 0.0576 |
| 4 | 1 | s0_ay | 0.2709 |
| 4 | 2 | s1_ax | 0.2546 |
| 4 | 3 | s2_ax | 0.1342 |
| 4 | 4 | s1_az | 0.0817 |
| 4 | 5 | s2_ay | 0.0731 |

## RF Top Features

| rank | feature | channel | importance |
|---:|---|---|---:|
| 1 | s1_ax_std | s1_ax | 0.0612 |
| 2 | s1_ax_energy | s1_ax | 0.0505 |
| 3 | s1_ax_rms | s1_ax | 0.0479 |
| 4 | s1_ax_max | s1_ax | 0.0432 |
| 5 | s1_gx_mean | s1_gx | 0.0409 |
| 6 | s1_ax_mean | s1_ax | 0.0374 |
| 7 | s1_ax_ptp | s1_ax | 0.0347 |
| 8 | s1_gy_mean | s1_gy | 0.0248 |
| 9 | s2_gy_mean | s2_gy | 0.0221 |
| 10 | s0_ax_mean | s0_ax | 0.0220 |
