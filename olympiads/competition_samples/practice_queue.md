# CEOAI Practice Queue

Current rule: produce visible evidence, then move on. Do not polish the archive.

## Official CEOAI Practice Tasks

Do these before inferred regional samples when the goal is direct CEOAI format rehearsal.

1. `ceoai_2026_practice1_stochastic_rift`
   - Why: official CEOAI Round 1 RL/MDP value-estimation task.
   - Artifact: `predictions.csv` for query-state values plus MSE/validation note.
   - CEOAI: `1(d)`, `1(e)`, `1(f)`.

2. `ceoai_2026_practice2_panda_mnist`
   - Why: official CEOAI/EUROAI Round 2 CV/domain-shift task with TorchScript model submission.
   - Artifact: `submission.zip` containing `model_sub1.pt` and `model_sub2.pt`, with parameter-count and per-scanner accuracy notes.
   - CEOAI: `5(a)`, `3(b)`, `3(c)`.

3. `ceoai_2026_practice2_trace_twins`
   - Why: official CEOAI/EUROAI Round 2 sequence-similarity task with ROC-AUC scoring.
   - Artifact: `submission.pkl` with `Submission.score_A` and `Submission.score_B`, plus ROC-AUC validation notes.
   - CEOAI: `4(a)`, `4(b)`, `2(d)`.

4. `ceoai_2026_practice1_star_observatory`
   - Why: official CEOAI Round 1 CV/regression task with two-subtask CSV formatting.
   - Artifact: 600-row submission CSV with center tuples and flux predictions.
   - CEOAI: `5(a)`, `2(a)`, `3(c)`.

5. `ceoai_2026_practice1_project_kraken`
   - Why: official CEOAI Round 1 multimodal task with three target types.
   - Artifact: one submission CSV with all three subtask outputs and metric notes.
   - CEOAI: `3(c)`, `5(a)`, `2(a)`.

## First 15 Regional/IOAI Support Tasks

1. `romania_onia_examples`
   - Why: closest simple train/eval tabular workflow.
   - Artifact: baseline notebook result plus prediction CSV shape check.
   - CEOAI: `2(a)`, `2(b)`, `2(d)`.

2. `kazakhstan_day2_player_clustering`
   - Why: direct clustering practice.
   - Artifact: cluster labels, cluster-count explanation, and one sanity plot/table.
   - CEOAI: `2(b)`, `2(d)`.

3. `ioai_2024_help_bobai`
   - Why: official IOAI feature-engineering/tabular task.
   - Artifact: baseline score plus one feature-improvement note.
   - CEOAI: `2(a)`, `2(d)`.

4. `poland_2024_imbalanced_classification`
   - Why: imbalanced data and metric discipline.
   - Artifact: confusion matrix and chosen metric explanation.
   - CEOAI: `2(a)`, `2(c)`.

5. `ioai_2025_chicken_counting`
   - Why: official CV counting/classification workflow.
   - Artifact: visible prediction examples and metric/submission check.
   - CEOAI: `5(a)`, `5(b)`.

6. `roai_2026_too_easy_fairy`
   - Why: Romanian IAIO/CEOAI selection-camp CV task with DINOv2 features and one-shot segmentation.
   - Artifact: binary mask CSV, 256-value answer validation, and Dice-score or mask sanity-check note.
   - CEOAI: `5(a)`, `5(c)`, `3(c)`.

7. `poland_2025_coin_counting`
   - Why: another counting/detection task, national selection style.
   - Artifact: baseline detector/counting output.
   - CEOAI: `5(a)`, `5(b)`.

8. `ioai_2025_concepts`
   - Why: official NLP/embedding/LLM-style task.
   - Artifact: baseline output table and error cases.
   - CEOAI: `4(b)`, `4(c)`.

9. `roai_2026_polyglot`
   - Why: Romanian IAIO/CEOAI selection-camp NLP embedding-alignment task.
   - Artifact: submission CSV mapping rows for both subtasks plus anchor/held-out accuracy notes.
   - CEOAI: `4(b)`, `4(c)`, `2(b)`.

10. `poland_2025_hallucination`
   - Why: very likely style for modern NLP competitions.
   - Artifact: classifier score and three inspected mistakes.
   - CEOAI: `4(b)`, `4(c)`.

11. `poland_2025_source_extraction`
   - Why: retrieval/embedding task.
   - Artifact: similarity/ranking output and validation metric.
   - CEOAI: `4(b)`, `4(c)`.

12. `romania_markov_maze`
    - Why: rare direct RL/MDP sample.
    - Artifact: value/Q/policy evidence and one explanation of convergence or policy choice.
    - CEOAI: `1(d)`, `1(e)`, `1(f)`.

13. `roai_2026_smart_warehouse`
    - Why: Romanian IAIO/CEOAI selection-camp RL task with TD(0), Q-learning, SARSA, and hidden policy evaluation.
    - Artifact: `solution.py`, `solution.pkl`, update-formula test output, and trained-policy evidence.
    - CEOAI: `1(d)`, `1(e)`, `1(f)`.

14. `poland_2024_pruning`
    - Why: DL optimization/model architecture reasoning.
    - Artifact: before/after model-size or score table.
    - CEOAI: `3(b)`, `3(c)`.

15. `neoai_underfitting_cv`
    - Why: underfitting/regularization debugging under competition pressure.
    - Artifact: one baseline and one corrected training run.
    - CEOAI: `3(b)`, `5(b)`.

## If Time Is Almost Gone

Do only these ten:

1. `ceoai_2026_practice1_stochastic_rift`
2. `ceoai_2026_practice2_panda_mnist`
3. `ceoai_2026_practice2_trace_twins`
4. `ceoai_2026_practice1_star_observatory`
5. `ceoai_2026_practice1_project_kraken`
6. `kazakhstan_day2_player_clustering`
7. `ioai_2025_chicken_counting`
8. `ioai_2025_concepts`
9. `romania_markov_maze`
10. `roai_2026_smart_warehouse`

This covers ML, clustering, CV, NLP, and RL with the strongest direct CEOAI-selection signal.

## Do Not Prioritize For CEOAI

- Audio tasks.
- IOAI team/practical generative-media tasks.
- Huge Kaggle tasks that require account setup unless the next block explicitly targets that skill.
- Full solution reading before you attempt a baseline.
