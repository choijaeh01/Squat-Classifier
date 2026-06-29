# Figure Captions v3

## fig_identity_residual_2x2_grid.png
- Korean caption: Identity embedding은 residual branch가 없을 때 channel identity 손실을 일부 보완하지만, residual branch가 들어가면 channel-wise 통계 정보가 이미 위치 정보를 강하게 제공하기 때문에 추가 이득이 작다.
- English caption: Identity embeddings help when residual statistics are absent, but their incremental benefit becomes small once the residual branch provides fixed-order channel-wise statistics.
- 핵심 메시지: claim은 identity 중심이 아니라 residual branch의 shared encoder bottleneck 완화로 두는 것이 안전하다.

## fig_identity_residual_effect_arrows.png
- Korean caption: 18개 seed-fold paired observation에서 계산한 identity/residual effect와 bootstrap CI.
- English caption: Identity and residual effects with bootstrap confidence intervals from 18 seed-fold paired observations.
- 핵심 메시지: residual effect가 identity effect보다 훨씬 크고, residual 이후 identity effect는 거의 0에 가깝다.

## fig_shared_encoder_information_bottleneck.png
- Korean caption: Shared encoder의 정보 병목과 identity/residual branch가 보완하는 정보의 차이.
- English caption: Information bottleneck view of the shared encoder and the different cues restored by identity embeddings and the residual branch.
- 핵심 메시지: identity는 token origin을 알려주지만 residual branch는 signal-derived statistics를 직접 제공한다.

## diagram_architecture_comparison_four_models.png
- Korean caption: All-Channel CNN, Shared 1D, Shared 1D + Identity, Residual Channel-Shared 구조의 정보 흐름 비교.
- English caption: Information-flow comparison of All-Channel CNN, Shared 1D, Shared 1D + Identity, and Residual Channel-Shared architectures.

## diagram_residual_branch_detailed.png
- Korean caption: train-scaled tensor에서 mean/std/min/max 72개 channel-wise statistics를 계산하는 residual branch.
- English caption: Residual branch that computes 72 channel-wise statistics from the train-scaled tensor.

## diagram_identity_embedding_detailed.png
- Korean caption: channel/sensor/modality/axis embedding을 token에 더해 token origin을 제공하는 identity embedding 구조.
- English caption: Identity embedding structure that adds channel, sensor, modality, and axis embeddings to each token.
