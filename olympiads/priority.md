# CEOAI Priority Handoff

## Status

Current date: 2026-07-12. CEOAI starts 2026-07-14. Calendar days left: 2. Effective work days left excluding 2026-07-13: 1.

Pace since previous run: FAST. Overall status: still behind because there is only one usable work day left, but the last interval was a real recovery push. Target schedule: ahead for the immediate 2026-07-08 onward mixed-practice slot; behind overall because several artifacts are still shallow or low-score.

Baseline used: automation memory entry `2026-07-12T08:06:55+03:00` plus prompt last-run timestamp `2026-07-12T05:03:52.306Z`, checked against files modified after `2026-07-12T08:03:52+03:00` Athens time.

Cumulative position: counted competition evidence now exists for Stochastic Rift, Trace Twins Part A, Panda MNIST, Broken BERT, Hungary model extension, Romania ONIA, Help BOBAI, Star Observatory local fixture, Project KRAKEN official-size baseline, IOAI Chicken Counting format/score baseline, IOAI Concepts zip baseline, and a Markov Maze RL drill. The biggest remaining direct gap is clustering/classical unsupervised practice, especially `kazakhstan_day2_player_clustering`.

## New Since Previous Run

- `olympiads/competition_samples/raw/ceoai-2026-practice-rounds/round-1/project_kraken/Project_KRAKEN_Baseline.ipynb`
  - Evidence: 18 executed code cells, no saved notebook errors, train shapes `(12000, 3, 128, 128)` and `(12000, 1024, 2)`, subtask 1 MSE `0.01477`, subtask 2 macro F1 `0.7937`, subtask 3 RMSE `0.11397`, and 3,000 test items predicted.
  - Syllabus: CEOAI `3(c)`, `5(a)`, `2(a)`.
  - Competition pattern: multimodal feature extraction, baseline, metric routing, strict CSV generation.
  - Verdict: counts. This passes the previous handoff gate.

- `olympiads/competition_samples/raw/ceoai-2026-practice-rounds/round-1/project_kraken/submission.csv`
  - Evidence: 9,000 rows, columns `subtaskID,datapointID,answer`, exactly 3,000 rows per subtask, 3,000 datapoints, no missing answers, subtask 1 semicolon-vector format checked, subtask 2 integer-like labels checked, subtask 3 numeric answers checked.
  - Syllabus: CEOAI `3(c)`, `5(a)`, `2(a)`.
  - Competition pattern: submission format, row-count validation, file validation.
  - Verdict: counts strongly.

- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/Chicken_Counting.ipynb`
  - Evidence: 14 executed code cells, no saved notebook errors, training/eval ran, logged test score `0.368`, MSE `858.5911`, MAE `28.1888`.
  - Syllabus: CEOAI `5(a)`, `5(b)`, `3(c)`.
  - Competition pattern: CV model pipeline, metric, submission artifact.
  - Verdict: counts only as weak baseline/format practice. Predictions are all-zero, so model quality is not competition-ready.

- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/submission.npz`
  - Evidence: arrays `pred_a` and `pred_b`, each shaped `(100, 1, 180, 320)`, dtype `float32`, no shape failure.
  - Syllabus: CEOAI `5(a)`, `5(b)`.
  - Competition pattern: array submission format validation.
  - Verdict: counts for output contract, not for score improvement.

- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb`
  - Evidence: 16 executed code cells, no saved errors, generated clue files and zip; saved score output is `0.0`.
  - Syllabus: CEOAI `4(b)`, `4(c)`, `3(c)`.
  - Competition pattern: NLP/embedding workflow, JSONL/zip submission.
  - Verdict: counts as output-format exposure only. It is not a useful scoring solution yet.

- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/submission.zip`
  - Evidence: contains `clues_a.jsonl` with 50 lines and `clues_b.jsonl` with 100 lines.
  - Syllabus: CEOAI `4(b)`, `4(c)`.
  - Competition pattern: JSONL/zip output contract.
  - Verdict: counts for file validation.

- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/markov_maze_production_drill.ipynb`
  - Evidence: 12 executed code cells, no saved errors, transition checks passed, value table checks passed, policy checks passed, and a submission-like CSV was written.
  - Syllabus: CEOAI `1(d)`, `1(e)`, `1(f)`.
  - Competition pattern: value/policy table, constraints, submission-like output.
  - Verdict: counts. The source-copy notebook under `competition_samples/raw/.../markov_maze_production_drill.ipynb` still has a saved `NotImplementedError`, so use the clean exercise copy as the counted artifact.

## Study Next

Final-session override from 2026-07-12 evening: do not start a new full exercise by default tomorrow morning. The user has enough solved-task volume; the remaining highest-risk bottleneck is API discovery under time pressure.

1. Run the final API survival session.
   - Target file: `olympiads/notes/ceoai_final_api_survival_session.md`
   - Supporting file: `olympiads/recommended_materials_2026/MODEL_API_SURVIVAL.md`
   - Syllabus pattern trained: library use across CEOAI `2`, `3`, `4`, and `5`.
   - Competition pattern trained: unfamiliar object -> signature/docs -> tiny input -> output fields -> loss/metric/submission contract.
   - Required visible evidence: open 2-3 old solved notebooks; for each, probe one unfamiliar sklearn/PyTorch/transformers/torchvision object with `dir`, `inspect.signature`, `help` or docstring, one tiny input, printed output keys/shapes, and one reload-style artifact check.
   - Why highest-value next move: another exercise is unlikely to change readiness; a repeatable API-discovery protocol can prevent the most unpredictable failure mode.
   - Stop condition: end when the protocol feels automatic, not when a new score is produced.

2. Optional only if calm and the API session is complete: Kazakhstan Day 2 Player Clustering.
   - Target file: `olympiads/competition_samples/raw/kazakhstan-tst-day2/solution.ipynb`
   - Reason to defer: it remains the clearest artifact gap, but starting it tomorrow morning is lower value than API survival if time or focus is limited.

## Pass/Fail Check Before Next Run

PASS: the user has rehearsed the API-survival protocol against 2-3 old solved notebooks, including `dir`, `inspect.signature`, `help` or docstring, tiny input, output keys/shapes, and a reload-style artifact check.

STRETCH PASS: Kazakhstan Day 2 has an executed clustering notebook, saved cluster labels or submission CSV, cluster-count justification, cluster-size sanity table/plot, and disk-level file validation.

FAIL: the final session becomes passive reading, broad theory review, or a new exercise that does not improve API discovery.

## Avoid Until This Is Done

- Do not start another new official IOAI 2025 task.
- Do not tune Project KRAKEN unless a validation bug appears; it now passes the previous gate.
- Do not polish Concepts.
- Do not redo Markov Maze in the raw source-copy notebook.
- Do not touch audio; CEOAI excludes IOAI-only audio.
- Do not collect more links or task cards.
- Do not turn the final morning into passive documentation reading.

## Evidence To Recheck

- `olympiads/competition_samples/problem_pattern_analysis.md`
- `olympiads/notes/ceoai_final_api_survival_session.md`
- `olympiads/recommended_materials_2026/MODEL_API_SURVIVAL.md`
- `olympiads/competition_samples/practice_queue.md`
- `olympiads/competition_samples/source_index.csv`
- `olympiads/competition_samples/task_cards/kazakhstan_day2_player_clustering.md`
- `olympiads/competition_samples/task_cards/ceoai_2026_practice1_project_kraken.md`
- `olympiads/competition_samples/task_cards/ioai_2025_chicken_counting.md`
- `olympiads/competition_samples/task_cards/ioai_2025_concepts.md`
- `olympiads/competition_samples/task_cards/romania_markov_maze.md`
- `olympiads/competition_samples/raw/kazakhstan-tst-day2/`
- `olympiads/competition_samples/raw/ceoai-2026-practice-rounds/round-1/project_kraken/Project_KRAKEN_Baseline.ipynb`
- `olympiads/competition_samples/raw/ceoai-2026-practice-rounds/round-1/project_kraken/submission.csv`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/Chicken_Counting.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/submission.npz`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/submission.zip`
- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/markov_maze_production_drill.ipynb`
- `olympiads/ceoai_syllabus.md`
- `olympiads/ioai_syllabus.md`
- `olympiads/schedule.csv`
- `olympiads/reviews/error_journal.jsonl`
