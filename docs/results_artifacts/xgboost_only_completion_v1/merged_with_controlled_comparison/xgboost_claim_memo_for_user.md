# XGBoost Completion Claim Memo For User

이 메모는 수치와 관찰만 정리한다. 논문 포함 여부와 최종 해석은 사용자가 판단한다.

## Merged Ranking

- controlled_stats_mlp: Macro F1 0.817383971028595, Accuracy 0.825
- controlled_shared_1d_residual: Macro F1 0.8004316129351925, Accuracy 0.8094444444444444
- controlled_all_channel_1d_cnn: Macro F1 0.7993945208514475, Accuracy 0.8250000000000002
- controlled_shared_1d_residual_identity: Macro F1 0.7973192046953161, Accuracy 0.8072222222222222
- feature_xgboost_v1: Macro F1 0.7961438852637317, Accuracy 0.8155555555555556
- feature_random_forest_v1: Macro F1 0.7845249924781911, Accuracy 0.8055555555555555
- rescnn_bigru_attention_lite_v1: Macro F1 0.7691308145395062, Accuracy 0.7944444444444444
- controlled_all_channel_1d_cnn_small: Macro F1 0.7561545162134669, Accuracy 0.7805555555555557
- lee_style_cnn_lstm_2d_v1: Macro F1 0.6268939297423978, Accuracy 0.6622222222222222

## Paired Differences

- vs feature_random_forest_v1: delta 0.011618892785540525, CI [-0.014546471186839204, 0.039839656253243615], n=18
- vs controlled_stats_mlp: delta -0.021240085764863265, CI [-0.059398476264448496, 0.01708112242491516], n=18
- vs controlled_shared_1d_residual: delta -0.004287727671460832, CI [-0.03845682566427537, 0.030439809736415187], n=18
- vs controlled_shared_1d_residual_identity: delta -0.0011753194315845847, CI [-0.03280006573543378, 0.030984108845500603], n=18
- vs controlled_all_channel_1d_cnn: delta -0.00325063558771589, CI [-0.05707363066002744, 0.05334039576134643], n=18

## Class 3 Excessive Lean

- recall: 0.6444444444444444
- F1: 0.6614163032203315

## XGBoost Top Features

- s1_ax_std: 0.0674416720867157
- s1_gx_mean: 0.06564743485715654
- s1_ax_energy: 0.06006597375704183
- s0_ax_mean: 0.03207558059754471
- s0_az_std: 0.030333431178910866
- s0_ax_median: 0.02245824470527522
- s2_gy_max: 0.019720920371279742
- s2_az_median: 0.019524071913717005
- s0_ay_median: 0.018223141190699406
- s1_az_mean: 0.016955283027426857
