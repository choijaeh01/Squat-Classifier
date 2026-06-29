# Squat IMU Experiments

Clean-room experimental codebase for IMU-only squat posture classification and future external IMU transfer experiments.

This project intentionally does not import modules from older squat repositories. Official paper results should be generated from this codebase after the full training pipeline is implemented.

Current scope:

- Project skeleton
- Data manifest system
- Target dataset loader scaffold
- Preprocessing scaffold with LOSO-safe scaler fitting
- Model registry scaffold
- Metrics and result-saving scaffold
- Synthetic-data smoke test
- Experiment protocol documentation
- Curated, GitHub-friendly result artifacts under `docs/results_artifacts/`

Documentation entry points:

- `docs/README.md`: documentation index
- `docs/experiment_protocol.md`: experiment protocol
- `docs/results_tracking.md`: run history and metric summaries
- `docs/results_artifacts/README.md`: selected CSV/MD/PNG artifacts copied from generated `results/`

Generated outputs policy:

- Full `results/`, processed target arrays, checkpoints, and dependency caches are not committed.
- Selected lightweight result tables and figures are mirrored under `docs/results_artifacts/` for review.

Run the smoke test:

```bash
cd squat_imu_experiments
python scripts/smoke_test.py
```

Run unit tests from the parent workspace:

```bash
python -m unittest discover -s squat_imu_experiments/tests -v
```
