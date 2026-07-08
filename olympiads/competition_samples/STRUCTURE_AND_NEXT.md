# Competition Samples Structure And Next Problem

Updated: 2026-07-08

Use this file as the entry point when you open `olympiads/competition_samples/`.
The directory is a practice archive, not a reading backlog. A task counts only
when it produces visible competition evidence: score, prediction table,
submission-like file, clustering labels, checked output shape, model-output
check, or value/Q/policy table.

## What Exists Here

Top-level files:

- `README.md`: purpose of the archive and fast-use rule.
- `source_index.csv`: source-of-truth index with 94 rows. Use this when you
  need to find a task by source, year, region, CEOAI tag, local path, or download
  status.
- `ceoai_task_map.md`: grouped map from tasks to CEOAI syllabus sections.
- `problem_pattern_analysis.md`: summary of what IOAI/regional/national AI
  tasks repeatedly ask for.
- `practice_queue.md`: ranked order of problems to solve during the CEOAI
  sprint.
- `task_cards/`: compact high-value task cards.
- `raw/`: downloaded public sources and direct official zip assets.

Raw archive status:

- Total raw files: 23147.
- Main raw file types: 19348 `.pt` files, 2402 `.png` files, 380 notebooks,
  210 `.sample` files, 159 markdown files, and 100 CSV files, plus zip/model/data
  assets.
- Raw source folders currently present:
  - `awesome-ioai-tasks`
  - `ceoai-2026-practice-rounds`
  - `hungary-haio-sparse`
  - `ioai-2024-official-zips`
  - `IOAI-2024-sparse`
  - `IOAI-2025-sparse`
  - `kazakhstan-tst-day1`
  - `kazakhstan-tst-day2`
  - `kazakhstan-tst-day3`
  - `kazakhstan-tst-day4`
  - `malaysia-china-ioai-tsp-2025-sparse`
  - `neoai-2025-sparse`
  - `polish-oai-2024-sparse`
  - `polish-oai-2025-sparse`
  - `polish-oai-2026-sparse`
  - `roai-2026-selection-camp-cpu-practical`
  - `romania-onia-examples`
  - `romania-roai-solved`

Attempt status:

- `task_attempts/kazakhstan_player_clustering/` exists.
- Do not count additional progress unless the attempt folder contains visible
  competition evidence: checked labels, predictions, metrics, submission-like
  files, or policy/value outputs.

## How To Navigate

Use this order:

1. Open `practice_queue.md`.
2. Open the matching file in `task_cards/`.
3. Open the local source under `raw/`.
4. Create an attempt folder under `task_attempts/`.
5. Stop only after the attempt folder contains visible evidence.

Do not start from `raw/` and browse randomly. The raw archive is large enough to
waste the whole study block.

## Task Cards Present

High-value task cards currently available:

- `task_cards/romania_onia_examples.md`
- `task_cards/kazakhstan_day2_player_clustering.md`
- `task_cards/ioai_2024_help_bobai.md`
- `task_cards/ioai_2025_chicken_counting.md`
- `task_cards/ioai_2025_concepts.md`
- `task_cards/ceoai_2026_practice1_project_kraken.md`
- `task_cards/ceoai_2026_practice1_star_observatory.md`
- `task_cards/ceoai_2026_practice1_stochastic_rift.md`
- `task_cards/ceoai_2026_practice2_panda_mnist.md`
- `task_cards/ceoai_2026_practice2_trace_twins.md`
- `task_cards/roai_2026_polyglot.md`
- `task_cards/roai_2026_too_easy_fairy.md`
- `task_cards/roai_2026_smart_warehouse.md`
- `task_cards/romania_markov_maze.md`
- `task_cards/poland_2024_pruning.md`
- `task_cards/poland_2025_hallucination.md`
- `task_cards/poland_2025_source_extraction.md`
- `task_cards/neoai_broken_bert.md`

## Problem To Solve Next

Solve `romania_onia_examples` first.

Why this first:

- It is the closest simple train/eval tabular workflow.
- It maps directly to CEOAI `2(a)` classical ML and `2(d)` feature engineering.
- If the selected part is unsupervised, it also touches CEOAI `2(b)` clustering.
- It trains the most common competition pattern: inspect data, identify the
  metric/output contract, build a baseline, and validate the submission shape.

Open these files first:

- `task_cards/romania_onia_examples.md`
- `raw/romania-onia-examples/Problema.md`
- `raw/romania-onia-examples/Dataset si evaluator/Dataset/Dataset.ipynb`
- `raw/romania-onia-examples/Dataset si evaluator/Evaluator.ipynb`
- `raw/romania-onia-examples/Dataset si evaluator/Dataset/dataset_train.csv`
- `raw/romania-onia-examples/Dataset si evaluator/Dataset/dataset_eval.csv`
- `raw/romania-onia-examples/Dataset si evaluator/Dataset/dataset_eval_t.csv`

Do not open the solution notebooks first unless you are doing post-attempt
review:

- `raw/romania-onia-examples/Solutii/Candidat_incepator/PredictPrices.ipynb`
- `raw/romania-onia-examples/Solutii/Candidat_avansat/PredictPrices.ipynb`

## Required Attempt Folder

Create:

```text
olympiads/competition_samples/task_attempts/romania_onia_baseline/
```

Minimum files to save there:

- `attempt.md`: short notes with source path, CEOAI tag, metric/output contract,
  and what you tried.
- `baseline_predictions.csv`: submission-like output or checked prediction
  dataframe export.
- Optional `notebook.ipynb`: your runnable baseline notebook, if you work in a
  notebook.

## Pass/Fail For The Next Attempt

Pass only if the attempt contains all of these:

- Exact source path.
- CEOAI syllabus tag.
- Train/eval row counts and column names.
- Dtype and missing-value inspection.
- Target column or output contract.
- Metric or evaluator rule, if visible.
- Baseline model or baseline rule.
- Prediction table.
- Row count and column validation for the submission-like output.

Fail if you only read the problem, inspect files, copy a solution, or write notes
without producing predictions or a checked output format.

## After Romania ONIA

Only after `romania_onia_baseline` passes, move to:

```text
kazakhstan_day2_player_clustering
```

Entry files:

- `task_cards/kazakhstan_day2_player_clustering.md`
- `raw/kazakhstan-tst-day2/batyr-yerdenov-2-2.ipynb`
- `raw/kazakhstan-tst-day2/README.md`

Required evidence:

- Selected features.
- Imputation/scaling decision.
- Chosen cluster count.
- Cluster labels.
- One sanity table or plot.
- Output/submission-format check.

This second task matters because the current error journal shows a real weak
spot: evaluating K-Means like a supervised classifier. Fix that with a proper
clustering artifact, not more theory.
