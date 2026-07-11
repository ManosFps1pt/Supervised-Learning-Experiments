# CEOAI Priority Handoff

## Status

Current date: 2026-07-11. CEOAI starts 2026-07-14. Calendar days left: 3. Effective work days left excluding 2026-07-13: 2.

Pace since previous run: ON SCHEDULE, but only narrowly. Overall status: behind. Target schedule: behind overall; the immediate Broken BERT repair target passed, but the Hungary 55-class artifact regressed.

Baseline used: prompt last-run timestamp `2026-07-11T04:00:10.402Z`. The automation memory file at `$CODEX_HOME/automations/ceoai-dynamic-study-coach-2/memory.md` was missing, so this run compared against the previous `priority.md` handoff and files modified after that timestamp.

Cumulative position: official CEOAI practice evidence exists for Stochastic Rift, Trace Twins, and Panda MNIST. Broken BERT now has a validation classification report and a valid 2,499-row submission CSV. The user is still not safe because the current Hungary model-extension submission file has been overwritten back to base-class-only predictions, and Star Observatory / Project Kraken remain unfinished official CEOAI-format tasks.

## New Since Previous Run

- `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/broken_bert_solution_translated_en.ipynb`
  - Evidence: 21 executed code cells, no saved notebook errors, validation/test CSVs loaded, embedding shape `(30522, 768)` inspected, corrupted embedding repair loop ran, and a classification report printed on 2,500 validation rows.
  - Metric evidence: validation accuracy `0.40`, macro F1 `0.37`, weighted F1 `0.36`.
  - Syllabus: CEOAI `4(b)` embeddings/transformers, `4(c)` pretrained text encoders, `3(c)` pretrained-model use/repair.
  - Competition pattern: pretrained model debugging, model-output checks, validation metric, submission format.
  - Verdict: counts as real progress. This converted the previous failed NLP attempt into a scored/submission artifact.

- `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/submit_bf534d97.csv`
  - Evidence: exactly 2,499 rows, columns `labels,id`, label distribution `neutral=2147`, `negative=289`, `positive=63`.
  - Syllabus: CEOAI `4(b)`, `4(c)`.
  - Competition pattern: submission format and file validation.
  - Verdict: counts. Not a strong model, but it is a valid competition output.

- `olympiads/recommended_materials_2026/07_sst2_dataset/sst2_competition_exercise.ipynb`
  - Evidence: 7 executed cells, train/validation/test parquet shapes and label checks printed.
  - Syllabus: CEOAI `4(b)`, `4(c)`.
  - Competition pattern: data inspection and file/shape validation only.
  - Verdict: weak support only. It does not count as completed practice because there is no metric or submission artifact.

- `olympiads/IOAI Material/12. Computer Vision/Code/01_classical_cv.py` and `classical_cv_result.png`
  - Evidence: classical CV script/result created after the baseline timestamp.
  - Syllabus: CEOAI `5(a)` image processing basics.
  - Competition pattern: preprocessing/feature extraction.
  - Verdict: partial. Useful recognition practice, but not a contest artifact unless it feeds a metric or required output file.

- `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites_translated_en.ipynb`
  - Evidence: translated notebook exists with 8 executed code cells, but it has a saved error and only shows the base model output shape `[16, 30]`.
  - Syllabus: CEOAI `3(c)`, `5(b)`.
  - Competition pattern: pretrained-model adaptation attempt.
  - Verdict: does not count as a completed repair.

- `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/submission.csv`
  - Evidence: exactly 1,100 rows and columns `Id,Class`, but current labels are only `0..29`; `above29=0`.
  - Syllabus: CEOAI `3(c)`, `5(b)`.
  - Competition pattern: submission format check.
  - Verdict: regression. The file is structurally valid but fails the 55-class model-extension contract because it contains no new-class predictions.

## Study Next

1. Repair Hungary model-extension output and prove the 55-class contract.
   - Target file: `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites.ipynb`
   - Syllabus tag: CEOAI `3(b)` neural-network heads, `3(c)` pretrained-model fine-tuning, `5(b)` image classification/transfer learning.
   - Competition pattern trained: pretrained head replacement, constraint-aware fine-tuning, subgroup metric checking, submission-format validation.
   - Required visible evidence: final model output shape `[batch, 55]`; printed accuracy or proxy counts separately for old classes `0..29`, S1 classes `30..44`, and S2 labeled classes `45..54`; regenerated `submission.csv` with 1,100 rows, columns `Id,Class`, labels within `0..54`, and at least one prediction above `29`.
   - Why highest-value next move: the current CSV silently reverted to base-class-only predictions; that is exactly the kind of output-contract failure that loses contest points.
   - Target schedule slot: 2026-07-08 onward targeted repair of weakest mixed-practice artifact.

2. Only after Hungary passes, start official Star Observatory baseline.
   - Target file: `olympiads/competition_samples/raw/ceoai-2026-practice-rounds/round-1/star_observatory/solution.ipynb`
   - Syllabus tag: CEOAI `5(a)` image processing, `2(a)` regression, `3(c)` pretrained/feature reuse.
   - Competition pattern trained: official CEOAI submission format, image preprocessing, regression metric, shape/file validation.
   - Required visible evidence: 600-row submission CSV with center tuples and flux predictions; print MAE or RMSE on a validation split; print row count and required columns before stopping.
   - Why highest-value after repair: it is an unfinished official CEOAI-format task and covers CV/regression under a strict output contract.
   - Target schedule slot: 2026-07-08 onward timed mixed official CEOAI samples.

## Pass/Fail Check Before Next Run

PASS: Hungary `modellbovites.ipynb` has post-2026-07-11 execution output showing `[batch, 55]`, separate old/S1/S2 metric or proxy checks, and `submission.csv` has exactly 1,100 rows, columns `Id,Class`, min/max within `0..54`, and at least one label above `29`.

FAIL: current `submission.csv` still has only labels `0..29`, the translated notebook remains the only fresh work, or no split/check output is printed.

Stretch PASS: Star Observatory has a new `solution.ipynb` and a 600-row submission CSV with validation MAE/RMSE plus format checks.

## Avoid Until This Is Done

- Do not open another NLP dataset or SST-2 exercise before Hungary is repaired.
- Do not keep translating notebooks unless the translated notebook is the one being executed to a metric/submission.
- Do not tune Broken BERT further; it is weak but now counts.
- Do not work on audio; CEOAI scope excludes IOAI-only audio.
- Do not collect more sources or task cards unless Star Observatory files are missing.

## Evidence To Recheck

- `olympiads/competition_samples/problem_pattern_analysis.md`
- `olympiads/competition_samples/practice_queue.md`
- `olympiads/competition_samples/source_index.csv`
- `olympiads/competition_samples/task_cards/neoai_broken_bert.md`
- `olympiads/competition_samples/task_cards/ceoai_2026_practice1_star_observatory.md`
- `olympiads/competition_samples/task_cards/ceoai_2026_practice2_trace_twins.md`
- `olympiads/competition_samples/task_cards/ceoai_2026_practice2_panda_mnist.md`
- `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/broken_bert_solution_translated_en.ipynb`
- `olympiads/competition_samples/raw/neoai-2025-sparse/5_Broken_BERT/submit_bf534d97.csv`
- `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites.ipynb`
- `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/modellbovites_translated_en.ipynb`
- `olympiads/competition_samples/raw/hungary-haio-sparse/2026/nyari-tabor/feladatok/submission.csv`
- `olympiads/competition_samples/raw/ceoai-2026-practice-rounds/round-1/star_observatory/`
- `olympiads/ceoai_syllabus.md`
- `olympiads/ioai_syllabus.md`
- `olympiads/schedule.csv`
- `olympiads/reviews/error_journal.jsonl`
