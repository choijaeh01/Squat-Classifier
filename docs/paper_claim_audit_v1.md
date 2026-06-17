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
