# Exercises: 1. Basics - 13-Day Sprint Version

Use this folder for practice exercises based on the university basics lesson.

Source material is in `../sources/`.

Lesson 1 is support work, not the main event. Use it to become faster at data
processing, validation, metrics, and basic debugging while spending most study
time on later ML topics.

## Time Budget

Total target: **2 focused sessions + light reuse during other lessons**.

- **Day 1 session, 60-75 min:** `01_vectorization_shapes.md` and the first half of `02_pandas_features.md`.
- **Day 2 session, 75-90 min:** finish `02_pandas_features.md`, then do `03_metrics_plots.md`.
- **Later lessons, 10-15 min as needed:** use `04_fast_baseline_playbook.md` before any new dataset or model.
- **Skip unless needed:** old PyTorch/GPU drills are useful only when a later notebook fails or runs slowly.

The goal is not to finish every possible basics drill. The goal is to build the
minimum reflexes that save time in competition:

- Python fluency
- NumPy vectorization
- broadcasting and shapes
- pandas basics
- metrics
- train/validation split
- simple sklearn baselines
- basic model-debugging sanity checks

No solution code should be generated here unless explicitly requested.

## Sprint Problem Sets

Work through these in order. Each file is a practice prompt, not a solution.

1. `01_vectorization_shapes.md` - one dense NumPy drill: axes, broadcasting, row/column logic, boolean indices, and pairwise distances.
2. `02_pandas_features.md` - one dense pandas drill: inspect, clean, group features, merge checks, missingness, and final feature table.
3. `03_metrics_plots.md` - one dense validation drill: metric choice, threshold/error inspection, leakage checks, and splitter choice.
4. `04_fast_baseline_playbook.md` - reusable 30-minute baseline workflow for any later lesson or contest dataset.

## Deferred Reference

These older prompts are intentionally no longer part of the sprint path:

- `04_leakage_adversarial_validation.md`
- `05_pipelines_cross_validation.md`
- `06_pytorch_tensors_training.md`
- `07_gpu_memory_debugging.md`
- `08_training_playbook_missing_data.md`

Use them only as references when a later task exposes that exact weakness.
