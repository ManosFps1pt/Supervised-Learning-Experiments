# IOAI Priority Handoff

## Status

Current date: 2026-07-27. Departure/logistics date is 2026-07-31, so there are 5 calendar days remaining through 2026-07-31, including today. The last full study day is 2026-07-30, leaving 4 full study days including today. IOAI starts on 2026-08-02.

Pace since previous run: FAST. Overall verdict: still behind for IOAI readiness by departure, but the main blocker from the previous run was cleared. Biggest current bottleneck: official past IOAI task completion is still too thin, especially score-useful CV/counting and NLP submission work.

Baseline used for comparison: previous automation run at 2026-07-27T07:12:00.773Z / local 2026-07-27 10:12. That run had Home Task 3 as the mandatory blocker because the then-current `MySolution` scored `0.0000`, below the random baseline `0.0161`. Already counted before that run: Home Task 1 audio adaptation, Home Task 2 full prediction validation, IOAI 2025 Concepts valid-format zero-score baseline, and AICC work through the imported 11-problem set.

This run inspected recent modifications, `priority.md`, `ioai_syllabus.md`, `practice_queue.md`, `source_index.csv`, Home Task folders, IOAI 2025 Chicken Counting notebooks and disk outputs, AICC folders, task cards, error journal tail, notebook outputs, and git status.

## New Since Previous Run

- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb` was modified on 2026-07-27 after the previous run.
- New Home Task 3 evidence: greedy information-gathering `MySolution` precomputes `1472 animals x 70 questions`, ran 805 animal-question batches, and completed all 150 `dev.csv` rows.
- Home Task 3 result: mean score `0.7764`, solved rate `99.3%`, mean queries `10.93 / 15`, wall time `172.3s`.
- It now beats the random baseline `0.0161` by a wide margin and closes the previous mandatory blocker.
- IOAI syllabus mapping: NLP; Pre-trained Language Models; LLM inference; information gathering/search; Model Evaluation; Data Processing.
- Contest pattern trained: constrained query budget, baseline-first improvement, metric validation, runtime control, model-output interpretation, and failure inspection.
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/Chicken_Counting.ipynb` and `Chicken_Counting_Solution.ipynb` were touched on 2026-07-27, but visible outputs are old. Do not count this as new study progress unless the next run shows a new timestamped metric or regenerated disk output.

Already counted before:

- Home Task 1: audio loading, waveform/spectrogram inspection, AST input contract `(1, 1024, 128)`, AST adaptation on 920 combined rows, validation accuracy `78.51%`, old/new retention score `0.7839285714285715`.
- Home Task 2: 400 demonstrations, 200 validation scenarios, 1600 test scenarios, 5327 state-action samples, MLP training, rollout GIFs, `success_rate: 0.92`, and disk-validated `predictions.jsonl` / `predictions.zip`.
- IOAI 2025 Concepts: local offline TF-IDF clue generator with valid row counts and valid zip packaging, but `Final Score: 0.0`.
- IOAI 2025 Chicken Counting: older `submission.npz` and `score.json` exist with `public_a = 0.36787944117144233` and `private_b = 0.36787944117144233`; this is useful but was not new in this interval.
- AICC: 11 imported / 27; about 9 attempted / 27 if partial/debug work counts; about 6 completed / 27 with executed notebook plus submission-like CSV evidence.

## Mandatory Coverage Buckets

- IOAI syllabus: incomplete. Evidence exists for Python/NumPy/Pandas, data processing, scikit-learn, PyTorch basics, tensor manipulation, supervised learning, MLPs, model evaluation, audio processing, pretrained audio encoders, model finetuning, imitation learning, NLP/text encoders, LLM-style inference/search, vision-text embeddings, sequence tagging, and baseline time-series features. Weak/missing as score-useful contest artifacts: object detection, segmentation, completed pretrained vision classifier workflow, autoencoders/GANs/diffusion, RL/search beyond Home Task 2 behavior cloning and Home Task 3 query search, and broader official past IOAI task completion.
- Past IOAI tasks: still underdone. Home Task 3 is now score-useful. IOAI 2025 Chicken Counting has an older low-score submission artifact, IOAI 2025 Concepts has a valid-format zero-score baseline, and IOAI 2025 Radar, Restroom, Antique, Pixel plus IOAI 2024 Help BOBAI, Lost in Hyperspace, and Madarian Cow remain unclosed or not recently advanced.
- IOAI 2026 home tasks: mandatory bucket now much healthier. Home Task 1 has credible audio evidence. Home Task 2 has valid full prediction artifacts. Home Task 3 now has a score-useful all-dev run: mean `0.7764`, solved `99.3%`, mean queries `10.93 / 15`.
- AICC progress out of 27: 11 imported / 27 (`buried-fault`, `deceptive-points`, `essay-gap`, `face-matching`, `find-brain-tumors`, `massive-problem`, `oriented-ship`, `polarity`, `shuffled`, `the-defected-nuts`, `word-lookups`). About 9 attempted / 27 if counting partial/debugging work. About 6 completed / 27 with executed notebook plus submission-like CSV evidence (`buried-fault`, `deceptive-points`, `essay-gap`, `polarity`, `shuffled`, `word-lookups`). `find-brain-tumors` remains partial unless the next run confirms its newer notebook/submission has no shape/index errors and has a validated score/submission contract.
- Audio coverage: acceptable minimum exists through Home Task 1: loading, waveform/spectrogram inspection, AST input contract, adaptation/finetuning, validation metric, and old-vs-new retention. Add only a short saved when-to-use audio note if time allows; do not make audio the main block before official past IOAI gaps.

## Study Next

1. Close IOAI 2025 Chicken Counting as a score-useful official CV/counting artifact.
   - Target file/folder: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/Chicken_Counting.ipynb`, `submission.npz`, and `score.json`.
   - Syllabus tag: Computer Vision; Image Classification / Counting; Convolutional Layers; PyTorch Basics; Model Evaluation Metrics; Data Processing.
   - Competition pattern trained: official past-task submission format, metric validation, disk artifact validation, baseline-first modeling, output interpretation.
   - Required visible evidence: rerun a contestant-side notebook cell with current timestamp; print validation score/MSE/MAE or official-style score; reload `submission.npz` from disk and print its array keys/shapes; keep or improve above the current score `0.36787944117144233`.
   - Why highest-value now: Home Task 3 is fixed, and the biggest IOAI syllabus gap is score-useful official CV execution under a real past-task contract.
   - Target schedule slot: 2026-07-27 next study block. No heavy work on 2026-07-31.

2. Optional stretch only after Chicken Counting has fresh validated evidence: improve or formally close IOAI 2025 Concepts.
   - Target file/folder: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb` and `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/`.
   - Syllabus tag: NLP; Text Encoders / embeddings; Language Modeling; Model Evaluation Metrics; Data Processing.
   - Competition pattern trained: official past-task submission format, metric validation, offline fallback, error analysis.
   - Required visible evidence: improve the validation probe above `0.0` or save a concise failure-analysis note explaining why the offline TF-IDF clue generator fails; keep valid `clues_a.jsonl`, `clues_b.jsonl`, and `submission.zip`.
   - Why second: official past IOAI work is mandatory and this task already has the file contract wired.
   - Target schedule slot: 2026-07-27 later block or 2026-07-28 first block only after Chicken Counting passes.

## Pass/Fail Check Before Next Run

PASS: `Chicken_Counting.ipynb` has a new visible run after 2026-07-27 16:00 local, `score.json` or printed metrics show a score at least `0.36787944117144233`, and `submission.npz` is reloaded from disk with array names/shapes printed.

STRETCH PASS: Concepts improves above `0.0` on the validation probe or has a saved failure-analysis close plus valid output files.

FAIL: work goes into new AICC imports, CEOAI-only review, passive syllabus reading, archive cleanup, Home Task polishing, or Chicken Counting notebook edits without a fresh metric and disk-level submission check.

## Avoid Until This Is Done

- Do not import more AICC problems before Chicken Counting has fresh validated evidence.
- Do not continue AICC `find-brain-tumors` before the next official IOAI task is closed.
- Do not continue Chameleon before Concepts is score-useful or formally closed.
- Do not polish Home Task 1 beyond a short audio model note unless a saved metric breaks.
- Do not redo Home Task 2 unless `predictions.zip` stops validating.
- Do not spend another main block on Home Task 3 unless the saved `0.7764` result is missing or broken.
- Do not study CEOAI syllabus as the primary target. CEOAI overlap counts only when tied to an IOAI artifact.
- Do not schedule heavy study on 2026-07-31; keep that date for departure, packing, offline files, account/platform checks, and rest.

## Evidence To Recheck

- `olympiads/ioai_syllabus.md`
- `olympiads/priority.md`
- `olympiads/competition_samples/practice_queue.md`
- `olympiads/competition_samples/source_index.csv`
- `olympiads/competition_samples/task_cards/`
- `olympiads/competition_samples/task_cards/ioai_2025_chicken_counting.md`
- `olympiads/competition_samples/task_cards/ioai_2025_concepts.md`
- `olympiads/competition_samples/task_cards/ioai_2024_help_bobai.md`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem1/Home-Task-1.ipynb`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/Home-Task-2.ipynb`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/predictions.jsonl`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/predictions.zip`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/dataset/`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/Chicken_Counting.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/submission.npz`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/score.json`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_a.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_b.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/submission.zip`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/`
- `olympiads/competition_samples/raw/IOAI-2024-sparse/`
- `olympiads/aicc/`
- `olympiads/aicc/aicc_problem_corpus.md`
- `olympiads/aicc/aicc_recommended_problem_order.md`
- `olympiads/aicc/buried-fault/`
- `olympiads/aicc/deceptive-points/`
- `olympiads/aicc/essay-gap/`
- `olympiads/aicc/find-brain-tumors/`
- `olympiads/aicc/polarity/`
- `olympiads/aicc/shuffled/`
- `olympiads/aicc/word-lookups/`
- `olympiads/reviews/error_journal.jsonl`
- `olympiads/reviews/ioai_error_journal.jsonl`
