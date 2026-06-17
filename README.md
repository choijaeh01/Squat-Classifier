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

Run the smoke test:

```bash
cd squat_imu_experiments
python scripts/smoke_test.py
```

Run unit tests from the parent workspace:

```bash
python -m unittest discover -s squat_imu_experiments/tests -v
```
