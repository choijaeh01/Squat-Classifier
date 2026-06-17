# Paper Claim Audit v1

## 1. 논문에서 안전하게 주장 가능한 내용

- Clean-room codebase에서 target squat IMU dataset을 `X.npy=(600,512,18)`, `y.npy=(600,)`, `subject_id.npy=(600,)`로 재구성했다.
- Evaluation은 subject-independent LOSO이며, final protocol에서는 held-out test subject를 train, validation, scaler fitting에서 완전히 제외했다.
- Full supervised matrix v1은 3 seeds x 6 folds x 7 models = 126 runs로 완료됐고 실패 run은 없었다.
- 모든 split leakage 및 scaler leakage check가 통과했다.
- `channel_shared_posres_attention_v3`는 full matrix에서 가장 높은 평균 Macro F1 0.8108을 기록했다.
- naive shared v2 모델들은 v3보다 크게 낮았고, 이는 channel identity 손실 및 pooling bottleneck 가능성을 뒷받침한다.
- v3는 all-channel Conv1D baseline들과 경쟁 가능한 성능을 보였다.

## 2. 통계적으로 조심해야 하는 내용

- v3의 평균 Macro F1은 all-channel baselines보다 높지만 paired confidence interval이 0을 포함한다.
- 따라서 v3가 all-channel Conv1D보다 통계적으로 유의하게 우수하다고 주장하면 안 된다.
- 현재 confidence interval은 18 paired observations에 대한 bootstrap CI이므로, small-subject dataset의 불확실성을 함께 명시해야 한다.

## 3. 주장하면 안 되는 내용

- SSL이 효과적이라는 주장
- external IMU dataset transfer가 가능하다는 주장
- augmentation, focal loss, balanced sampling이 성능을 높인다는 주장
- 1 seed literature temporal baseline screening 결과만으로 CNN-LSTM, CNN-GRU, TCN, Transformer, classical feature baseline 대비 우월성을 주장하는 내용
- v3가 모든 IMU posture dataset에서 일반적으로 우수하다는 주장
- hyperparameter tuning을 통해 최적 성능을 얻었다는 주장

## 4. 리뷰어가 물어볼 가능성이 높은 질문

1. Subject 수가 6명뿐인데 일반화 주장이 가능한가?
2. Validation split이 test subject와 독립적인가?
3. Scaler fitting에서 test subject가 섞이지 않았는가?
4. v3가 all-channel baseline보다 유의하게 좋은가?
5. all_channel_conv1d_small이 강한 baseline인데 v3의 의미는 무엇인가?
6. naive shared v2가 낮은 이유가 구현 문제는 아닌가?
7. Class 3 Excessive Lean은 충분히 해결됐는가?
8. transfer learning 가설을 왜 본 논문에서 실험하지 않았는가?

## 5. 질문별 답변 초안

1. 본 연구는 small-subject IMU setting을 명시적으로 다루며, subject-independent LOSO로 평가했다. 일반화 범위는 6-subject dataset 내부의 held-out subject generalization으로 제한한다.
2. Test subject는 train/validation/scaler fitting에서 완전히 제외했다. Validation은 candidate train subjects 내부에서 subject-class stratified로 구성했다.
3. Scaler는 fold별 train indices 400개에만 fit했고, scaler audit에서 validation/test index 사용 여부를 검사했다.
4. 평균 Macro F1은 v3가 높지만 paired CI가 0을 포함하므로 유의한 우월성은 주장하지 않는다.
5. all_channel_conv1d_small은 parameter-efficient strong baseline이다. v3의 의미는 shared encoder 구조가 channel identity를 보존하면 all-channel baseline과 경쟁 가능하다는 점이다.
6. v2는 overfit diagnostic에서도 낮았고 class 3에서 붕괴가 컸다. 이는 단순 parameter 문제가 아니라 channel identity 손실 및 pooling bottleneck으로 해석한다.
7. Class 3은 여전히 어려운 class지만 v3는 naive shared v2 대비 class 3 recall/F1을 크게 개선했다.
8. Transfer learning은 외부 dataset adapter와 protocol 설계가 별도로 필요하다. 본 논문은 먼저 clean supervised LOSO baseline과 architecture claim을 고정한다.

## 6. v3를 proposed model로 둘 때의 장점

- Shared encoder의 논리와 IMU channel identity 보존을 함께 만족한다.
- Parameter count가 all-channel v1과 유사한 범위다.
- Full matrix에서 가장 높은 평균 Macro F1을 보였다.
- naive shared v2 대비 개선 폭이 커서 architecture motivation을 설명하기 쉽다.

## 7. all_channel_conv1d_small이 강한 baseline인 점 설명

`all_channel_conv1d_small`은 낮은 parameter budget에서도 Macro F1 0.7740을 기록했다. 이는 small dataset에서 단순 parameter reduction 자체도 중요하다는 점을 보여준다. 따라서 v3의 claim은 “작은 모델보다 무조건 우수”가 아니라 “channel identity를 보존한 shared encoder가 strong all-channel baselines와 경쟁 가능한 성능을 보였다”로 제한한다.

## 8. naive shared v2가 낮은 결과 해석

Mean pooling 및 단순 attention pooling v2는 channel token을 지나치게 압축해 sensor 위치와 axis identity를 충분히 보존하지 못한 것으로 해석한다. 특히 Class 3 Excessive Lean에서 recall이 크게 낮았다. v3는 channel/sensor/modality/axis embedding과 residual branch로 이 문제를 완화했다.

## 9. Class 3 Excessive Lean 결과 활용

Class 3은 여러 모델에서 어려운 class다. v3의 class 3 F1은 0.7184로 naive shared v2보다 크게 높았다. Error analysis에서는 Class 3 confusion을 중심으로 posture-specific sensor information의 중요성을 설명할 수 있다.

## 10. transfer learning을 후속 연구로 남기는 이유

External IMU dataset은 sensor count, channel layout, task label, sampling rate가 target dataset과 다를 가능성이 높다. Transfer learning은 adapter와 pretraining protocol 자체가 별도 연구 질문이므로, 본 논문에서는 supervised clean-room target protocol을 먼저 확정한다.

## 11. literature temporal baseline screening 관련 claim 제한

`literature_baseline_screening_v1`은 문헌 기반 temporal baseline을 clean-room codebase에 추가하고 1 seed x 6 folds로 screening하는 단계다. 이 결과는 full supervised matrix v1 lock을 대체하지 않으며, 기존 locked result directory를 수정하지 않는다.

Screening에 포함되는 neural family는 CNN-LSTM, CNN-GRU, lite Residual CNN-BiGRU-Attention, TCN, LSTM-only, GRU-only, lite Transformer encoder이다. Classical feature RandomForest와 LinearSVM은 scikit-learn이 설치된 환경에서만 실행하고, 없으면 skipped로 기록한다.

CAU screening에서는 54개 run이 모두 성공했고 skipped run은 없었다. 1 seed screening macro F1은 `feature_random_forest_v1` 0.7900, `rescnn_bigru_attention_lite_v1` 0.7485, `feature_linear_svm_v1` 0.7164, `tcn_literature_v1` 0.7049 순이었다.

논문 claim에는 다음 제한을 둔다.

- 3 seed full extension 전에는 literature baseline 대비 우월성을 main result로 주장하지 않는다.
- 1 seed screening과 3 seed locked full matrix를 같은 통계 단위로 비교하지 않는다.
- Screening 결과는 full extension 후보 선별 및 reviewer 대응 준비용으로만 사용한다.

## 12. literature baseline full extension 관련 claim 제한

`literature_baseline_full_extension_v1`은 3 seed x 6 folds x 8 models = 144 runs로 완료됐고, 실패 및 skipped run은 없었다. 모든 split leakage와 scaler leakage check가 통과했다.

추가로 구현한 `lee_style_cnn_lstm_2d_v1`은 Lee et al.의 exact reproduction이 아니라, 현재 18채널 IMU squat dataset과 final LOSO protocol에 맞춘 adapted clean-room CNN-LSTM baseline이다. 따라서 Lee et al. 논문 결과 재현이라고 주장하면 안 된다.

수치상 `feature_random_forest_v1`은 Macro F1 0.7845로 literature/classical extension 내 가장 높았고, `rescnn_bigru_attention_lite_v1`은 Macro F1 0.7691로 neural temporal baseline 중 가장 높았다. 기존 locked `channel_shared_posres_attention_v3`의 Macro F1은 0.8108이다.

안전한 문장:

- Literature/classical baseline full extension을 동일 final validation policy에서 추가 평가했다.
- RandomForest와 ResCNN-BiGRU-Attention-lite는 강한 reference baseline으로 관찰됐다.
- Lee-style adapted CNN-LSTM은 simple CNN-LSTM보다 높은 Macro F1을 보였다.
- 기존 proposed v3는 merged reference ranking에서 가장 높은 평균 Macro F1을 유지했다.

조심해야 할 문장:

- RandomForest보다 neural proposed model이 통계적으로 우수하다는 주장
- Literature extension 결과 때문에 기존 main model을 반드시 바꿔야 한다는 주장
- Lee-style 결과를 Lee et al. exact reproduction 결과로 표현하는 문장
- 결과를 보고 tuning하지 않았는데도 최적 literature baseline 성능이라고 표현하는 문장

## 13. v3 component ablation 관련 claim 제한

`v3_component_ablation_v1`은 original `channel_shared_posres_attention_v3`를 새로 튜닝하지 않고, component 제거 variant 4개를 같은 final validation policy에서 3 seeds x 6 folds로 실행한 분석이다. 기존 locked full matrix와 literature full extension 결과는 read-only reference로만 사용했다.

결과 요약:

- `channel_shared_posres_attention_v3_no_residual`: Macro F1 0.5937
- `channel_shared_posres_attention_v3_residual_only_mlp`: Macro F1 0.8036
- `channel_shared_posres_attention_v3_no_identity`: Macro F1 0.7675
- `channel_shared_posres_meanpool_v3_no_attention`: Macro F1 0.8082
- original `channel_shared_posres_attention_v3`: locked Macro F1 0.8108

Original v3 대비 paired delta는 no residual -0.2171, residual-only MLP -0.0072, no identity -0.0434, no attention -0.0026이었다. no residual과 no identity의 confidence interval은 0을 포함하지 않았고, residual-only MLP와 no attention은 0을 포함했다.

안전한 문장:

- Component ablation에서 residual summary branch 제거는 큰 성능 하락과 연결됐다.
- Identity embedding 제거도 original v3 대비 낮은 결과를 보였다.
- Attention pooling을 mean pooling으로 바꾼 ablation은 original v3와 매우 가까운 결과를 보였다.
- Residual-only MLP가 강하게 관찰되어, small-subject IMU dataset에서는 raw channel summary statistics도 강한 signal을 제공할 수 있다.

조심해야 할 문장:

- Attention pooling이 v3 성능의 주된 원인이라는 주장
- Shared temporal encoder만으로 v3 성능을 설명하는 주장
- Residual-only MLP 결과를 근거로 proposed model을 반드시 바꿔야 한다는 결론
- Attention weight와 RandomForest feature importance를 동일한 설명량으로 해석하는 문장

Attention-RF alignment는 channel-level Pearson 0.6361, Spearman 0.3870으로 기록됐다. 둘 다 `s1_ax` 관련 정보를 상위로 보았지만, attention weight와 feature importance는 산출 의미가 다르므로 정성적 참고 분석으로만 사용한다.

## 13. v3 component ablation 관련 claim 제한

`v3_component_ablation_v1`은 기존 locked full supervised matrix와 literature baseline full extension을 수정하지 않고, 별도 결과 디렉토리에서 4개 component ablation 모델을 3 seeds x 6 folds로 실행한 실험이다. 총 72개 run이 모두 성공했고 split/scaler leakage check는 모두 통과했다.

관찰된 Macro F1은 다음과 같다.

| model | macro F1 | CI |
|---|---:|---|
| channel_shared_posres_attention_v3 | 0.8108 | [0.7644, 0.8537] |
| channel_shared_posres_meanpool_v3_no_attention | 0.8082 | [0.7471, 0.8603] |
| channel_shared_posres_attention_v3_residual_only_mlp | 0.8036 | [0.7488, 0.8530] |
| channel_shared_posres_attention_v3_no_identity | 0.7675 | [0.7112, 0.8195] |
| channel_shared_posres_attention_v3_no_residual | 0.5937 | [0.4998, 0.6842] |

안전한 문장:

- Residual branch를 제거하면 Macro F1이 크게 낮아졌다.
- Residual-only MLP가 original v3와 가까운 성능을 보였으므로, v3 성능의 상당 부분은 channel-wise summary statistics와 관련될 수 있다.
- Identity embedding 제거 모델은 original v3보다 낮았지만, residual branch가 남아 있어 모든 position cue를 제거한 실험은 아니다.
- Attention pooling을 mean pooling으로 바꾼 모델은 original v3와 가까운 성능을 보였고 paired CI가 0을 포함했다.
- Attention-RF alignment는 정성적 post-hoc 비교이며, attention weight와 RF feature importance는 같은 의미가 아니다.

피해야 할 문장:

- Attention pooling이 v3 성능 향상의 핵심 원인이라고 단정하는 문장
- Shared encoder 자체가 residual summary보다 더 중요한 성능 원인이라고 단정하는 문장
- Residual-only MLP 결과를 RandomForest와 동일한 모델 또는 동일한 해석 단위로 표현하는 문장
- `no_identity` 결과만으로 channel/sensor/modality/axis identity 전체가 필수라고 단정하는 문장
- Ablation 결과를 보고 모델 구조를 바꾸거나 hyperparameter tuning을 수행했다는 인상을 주는 문장
