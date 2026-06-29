# Locked Claim Audit

## Safe claims

- The full supervised matrix completed 126/126 runs without failed runs.
- All split and scaler leakage checks passed.
- The proposed position-aware shared encoder v3 achieved the highest mean Macro F1 in this matrix.
- Naive shared v2 models performed substantially worse than v3.
- The paired differences between v3 and all-channel baselines were positive on average, but the confidence intervals included zero.

## Unsafe claims

- Do not claim statistically significant superiority over all-channel Conv1D.
- Do not claim transfer learning effectiveness.
- Do not claim SSL, augmentation, focal loss, or balanced sampling benefits.
- Do not claim external dataset generalization.
