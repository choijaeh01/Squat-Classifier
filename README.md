# Squat IMU Experiments

Clean-room experimental codebase for IMU-only squat posture classification and future external IMU transfer experiments.

This project intentionally does not import modules from older squat repositories. The current supervised IMU paper-candidate results are generated from this clean-room codebase.

Current scope:

- Target dataset conversion from raw CSV and manual window metadata
- LOSO evaluation with within-train stratified validation
- Train-only scaler fitting and leakage audit outputs
- Controlled feature extractor comparison with a shared classifier head
- XGBoost, Random Forest, SVM, literature temporal baseline comparisons
- v3 component ablation and professor-facing report assets
- Professor report v2 with architecture, normalization/scaler, parameter count, and confusion matrix refinements
- Curated, GitHub-friendly result artifacts under `docs/results_artifacts/`

Documentation entry points:

- `docs/README.md`: documentation index
- `docs/experiment_protocol.md`: experiment protocol
- `docs/results_tracking.md`: run history and metric summaries
- `docs/results_artifacts/README.md`: selected CSV/MD/PNG artifacts copied from generated `results/`
- `docs/professor_report_v1/Professor Report - Residual Channel-Shared Encoder.md`: professor-facing report mirror with tables, figures, and diagrams
- `docs/professor_report_v2/Professor Report v2 - Architecture and Protocol.md`: refined professor report mirror focused on architecture and protocol evidence

Generated outputs policy:

- Full `results/`, processed target arrays, checkpoints, and dependency caches are not committed.
- Selected lightweight result tables and figures are mirrored under `docs/results_artifacts/` for review.
- Professor-facing Obsidian report assets are mirrored under `docs/professor_report_v1/` and `docs/professor_report_v2/`; the personal Obsidian vault copy remains outside this repository.

Run the smoke test:

```bash
cd squat_imu_experiments
python scripts/smoke_test.py
```

Run unit tests from the parent workspace:

```bash
python -m unittest discover -s squat_imu_experiments/tests -v
```
