# CEOAI Priority Handoff

## Status

Current date: 2026-07-08. CEOAI starts 2026-07-14.

Calendar days left: 6. Effective work days left excluding 2026-07-13: 5.

Pace since previous automation run: SLOW. Baseline used: prompt-provided last run timestamp `2026-07-07T14:01:58.793Z` / local `2026-07-07 17:01:58`, plus the previous `priority.md` because the automation memory file was missing.

Overall verdict: behind overall. Target schedule status: behind for the 2026-07-08 onward timed mixed-practice phase. Since the last run, the repo shows useful problem-description correction work, but no new score, regenerated Concepts zip, classifier score, clustering labels, submission-like file, value/Q/policy table, or confusion matrix.

Cumulative position already counted before this interval: Search/RL comparison, minimax/A*/MDP/Q-table lesson work, NLP TF-IDF/BERT/language-model drills, classical ML and clustering lessons, MNIST/CV benchmark, PyTorch regression/two-moons notebooks, Romania ONIA outputs, Polish imbalanced-classification score, Help BOBAI router/submission artifacts, IOAI 2025 Chicken Counting `submission.npz` and `score.json`, IOAI 2025 Concepts valid JSONL/zip with `0.0` score, archive indexing, task cards, translations, queue setup, and Markov Maze value/policy/submission-style evidence.

Highest live weakness: IOAI 2025 Concepts still has valid file format but saved score evidence remains `hit@10: 0.0`, `NDCG@10: 0.0`, and `Final Score: 0.0`. The required repair artifact from the previous handoff was not completed.

## New Since Previous Run

Counted as limited study evidence, not competition-artifact progress:

- `olympiads/competition_samples/problem_description_analysis_log.md`
  - New corrected entries for Polish OAI 2025 Hallucination Detection, Polish OAI 2025 Source Extraction, Polish OAI 2024 Pruning, and NEOAI 2025 Broken BERT.
  - Syllabus mapping:
    - Hallucination Detection: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures.
    - Source Extraction: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures.
    - Pruning: CEOAI `3(b)` neural-network optimization, CEOAI `3(c)` model architecture / parameters.
    - Broken BERT: CEOAI `4(b)`, `4(c)`, and CEOAI `3(c)` transformer/encoder recognition.
  - Competition pattern trained: statement parsing, metric identification, baseline route selection, constraints, output-contract awareness.
  - Verdict: useful, but insufficient. It did not produce a score, prediction table, confusion matrix, zip/jsonl regeneration, model-parameter artifact, or checked submission.

- `olympiads/competition_samples/problem_description_analysis_protocol.md`
  - Syllabus mapping: none directly.
  - Competition pattern trained: statement-analysis workflow.
  - Verdict: process scaffolding only. Does not count as competition readiness by itself.

- Translated notebooks:
  - `olympiads/competition_samples/raw/polish-oai-2025-sparse/1_etap/2_wykrywanie_halucynacji/2_wykrywanie_halucynacji_translated_en.ipynb`
  - `olympiads/competition_samples/raw/polish-oai-2025-sparse/2_etap/ekstrakcja_zrodel/ekstrakcja_zrodel_translated_en.ipynb`
  - `olympiads/competition_samples/raw/polish-oai-2024-sparse/first_stage/pruning/pruning_translated_en.ipynb`
  - `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/broken_bert_solution_translated_en.ipynb`
  - Syllabus mapping: NLP `4(b)/4(c)`, DL `3(b)/3(c)`, transformer recognition `3(c)`.
  - Competition pattern trained: source accessibility and statement comprehension only.
  - Verdict: not enough. Hallucination and Pruning translated notebooks have zero executed cells and no outputs. Source Extraction has no outputs. Broken BERT has old/inherited solution outputs, not a new local repaired-model attempt.

Not counted:

- `olympiads/IOAI Material/2. (Mostly) Linear models/exercises/README.md` update: organizational note only.
- Any raw archive/source collection after the cutoff: no score/output evidence.
- Kazakhstan clustering remains blocked unless `train.csv` and `sample_submission.csv` are found locally.
- The new blank IOAI Concepts log template at the bottom of `problem_description_analysis_log.md`: incomplete and not counted.

## Study Next

1. Target file: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb`
   - Syllabus tag: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures, CEOAI `3(c)` transformer/encoder recognition.
   - Competition pattern trained: metric-driven repair, model-output inspection, discriminative feature/clue strategy, JSONL/zip submission validation.
   - Required visible evidence: regenerate `out/clues_a.jsonl`, `out/clues_b.jsonl`, and `out/submission.zip`; save either a score above `0.0` or three inspected weak cases with label/options/current clues/guesser output and one changed clue-generation strategy.
   - Why this is highest value: the task already has a valid file contract but zero useful score. Repairing a zero-score official NLP/embedding task is more valuable than opening another notebook.
   - Target schedule slot: 2026-07-08 timed mixed-practice repair block.

2. Target file: `olympiads/competition_samples/raw/polish-oai-2025-sparse/1_etap/2_wykrywanie_halucynacji/2_wykrywanie_halucynacji_translated_en.ipynb`
   - Syllabus tag: CEOAI `4(b)` embeddings / representations, CEOAI `4(c)` related NLP architectures.
   - Competition pattern trained: binary classification metric, allowed-library feature engineering, ROC AUC, baseline-first NLP modeling.
   - Required visible evidence: one executed baseline producing validation ROC AUC, plus three inspected false positives or false negatives.
   - Why second: use this only if Concepts gets real evidence quickly. It converts the new statement-analysis work into a scored NLP artifact.
   - Target schedule slot: 2026-07-08 stretch only after Concepts pass.

## Pass/Fail Check Before Next Run

Pass if `Concepts.ipynb` has new post-2026-07-08 execution evidence and `out/clues_a.jsonl`, `out/clues_b.jsonl`, and `out/submission.zip` are regenerated after this handoff, with either score above `0.0` or three explicit weak-case inspections tied to a changed strategy.

Stretch pass if Hallucination Detection has an executed ROC AUC baseline and three inspected mistakes.

Fail if the next interval produces only more translations, descriptions, archive curation, raw downloads, task-card edits, or an unchanged Concepts `submission.zip` with no weak-case diagnosis.

## Avoid Until This Is Done

- Opening another new competition sample.
- More translations.
- More archive curation or source indexing.
- Kazakhstan clustering recovery unless the official CSVs are already present.
- Old MNIST/CV cleanup.
- Audio.
- IOAI team/generative-media tasks.
- From-scratch standard algorithm reimplementation.
- Any task that does not produce a score, confusion matrix, prediction table, submission-like file, clustering labels, checked model output, value/Q/policy table, or explicit format validation.

## Evidence To Recheck

- `olympiads/competition_samples/problem_pattern_analysis.md`
- `olympiads/competition_samples/practice_queue.md`
- `olympiads/competition_samples/source_index.csv`
- `olympiads/competition_samples/problem_description_analysis_log.md`
- `olympiads/competition_samples/problem_description_analysis_protocol.md`
- `olympiads/competition_samples/task_cards/ioai_2025_concepts.md`
- `olympiads/competition_samples/task_cards/poland_2025_hallucination.md`
- `olympiads/competition_samples/task_cards/poland_2025_source_extraction.md`
- `olympiads/competition_samples/task_cards/kazakhstan_day2_player_clustering.md`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_a.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_b.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/submission.zip`
- `olympiads/competition_samples/raw/polish-oai-2025-sparse/1_etap/2_wykrywanie_halucynacji/2_wykrywanie_halucynacji_translated_en.ipynb`
- `olympiads/competition_samples/raw/polish-oai-2025-sparse/2_etap/ekstrakcja_zrodel/ekstrakcja_zrodel_translated_en.ipynb`
- `olympiads/competition_samples/raw/polish-oai-2024-sparse/first_stage/pruning/pruning_translated_en.ipynb`
- `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/broken_bert_solution_translated_en.ipynb`
- `olympiads/ceoai_syllabus.md`
- `olympiads/ioai_syllabus.md`
- `olympiads/schedule.csv`
- `olympiads/reviews/error_journal.jsonl`
- `C:\Users\Manos\.codex\automations\ceoai-dynamic-study-coach-2\memory.md`
