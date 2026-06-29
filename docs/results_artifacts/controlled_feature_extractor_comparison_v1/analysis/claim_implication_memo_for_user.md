# Controlled Feature Extractor Claim Implication Memo

이 메모는 수치 중심 관찰만 정리한다. 최종 학술 해석과 main model 판단은 사용자가 수행한다.

## Macro F1 상위 모델

- controlled_stats_mlp: Macro F1 0.817383971028595, Accuracy 0.825
- controlled_shared_1d_residual: Macro F1 0.8004316129351925, Accuracy 0.8094444444444444
- controlled_all_channel_1d_cnn: Macro F1 0.7993945208514475, Accuracy 0.8250000000000002
- controlled_shared_1d_residual_identity: Macro F1 0.7973192046953161, Accuracy 0.8072222222222222
- feature_random_forest_v1: Macro F1 0.7845249924781911, Accuracy 0.8055555555555555
- rescnn_bigru_attention_lite_v1: Macro F1 0.7691308145395062, Accuracy 0.7944444444444444
- controlled_all_channel_1d_cnn_small: Macro F1 0.7561545162134669, Accuracy 0.7805555555555557
- feature_linear_svm_v1: Macro F1 0.721326918228006, Accuracy 0.746111111111111
- controlled_flatten_mlp: Macro F1 0.7045902580171839, Accuracy 0.7344444444444443
- lee_style_cnn_lstm_2d_v1: Macro F1 0.6268939297423978, Accuracy 0.6622222222222222

## Class 3 Excessive Lean 상위 모델

- controlled_shared_1d_residual: recall 0.7222222222222223, F1 0.7541546836616831
- controlled_shared_1d_residual_identity: recall 0.7416666666666666, F1 0.750862937346277
- controlled_stats_mlp: recall 0.6916666666666667, F1 0.7149652856162004
- feature_random_forest_v1: recall 0.65, F1 0.6847691300101206
- controlled_all_channel_1d_cnn_small: recall 0.663888888888889, F1 0.6835277746676404
- controlled_all_channel_1d_cnn: recall 0.6805555555555556, F1 0.6691669832041227
- rescnn_bigru_attention_lite_v1: recall 0.6833333333333332, F1 0.655228315956381
- feature_linear_svm_v1: recall 0.6055555555555556, F1 0.5799837240659473
- controlled_flatten_mlp: recall 0.5416666666666666, F1 0.5571366220114449
- controlled_2d_cnn: recall 0.5333333333333333, F1 0.4920252601482644

## subjectwise row count

- rows: 78
