# IOAI Priority Handoff

## Status

Current date: 2026-07-26. Departure/logistics date is 2026-07-31, so there are 6 calendar days remaining through 2026-07-31, including today. The last full study day is 2026-07-30, leaving 5 full study days including today. IOAI starts on 2026-08-02.

Pace since previous run: NO MEANINGFUL PROGRESS. Overall verdict: behind for IOAI readiness by departure. Biggest current bottleneck: IOAI 2026 Home Task 3 still has only random-baseline evidence and an interrupted strategy cell; it has not been converted into a serious bounded all-dev attempt.

Baseline used for comparison: the 2026-07-25 handoff already counted Home Task 1 audio adaptation, Home Task 2 full prediction validation, IOAI 2025 Concepts valid-format zero-score baseline, AICC corpus cataloging, and the three imported AICC folders. This run inspected recent file modifications, notebooks, outputs, task folders, AICC state, error journals, and git status. A study block counts only with visible IOAI-relevant evidence: metrics, predictions, saved submission files, checked model outputs, validated shape/file contracts, audio plots/features, masks/boxes, value/policy tables, or syllabus rows tied to runnable artifacts.

## New Since Previous Run

- No new visible IOAI-relevant artifact was found after the previous automation run at 2026-07-25T07:24:33.840Z.
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb` is still last modified on 2026-07-24 and still shows the random baseline on all 150 dev rows: mean score `0.0161`, solved rate `2.0%`, mean queries `14.89 / 15`. The later strategy work still ends with `KeyboardInterrupt`.
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb` and `out/submission.zip` were modified on 2026-07-24 before the previous handoff. They remain valid-format but not score-useful: 30-row dev probe `Final Score: 0.0`, 50 `clues_a.jsonl` rows, 100 `clues_b.jsonl` rows, and zip members `clues_a.jsonl` / `clues_b.jsonl`.
- No new AICC folder, notebook completion, submission zip, CSV, pkl, or metric table was found since the previous run.

Already counted before:

- Home Task 1: audio loading, waveform/spectrogram inspection, AST input contract `(1, 1024, 128)`, AST adaptation on 920 combined rows, validation accuracy `78.51%`, `acc_old = 0.8`, `acc_new = 0.7678571428571429`, weighted score `0.7839285714285715`.
- Home Task 2: 400 demonstrations, 200 validation scenarios, 1600 test scenarios, 5327 state-action samples, MLP training, rollout GIFs, `MLP {'success_rate': 0.92, 'avg_steps': 22.585, 'avg_invalid_pickup_or_dropoff': 0.0}`, and disk-validated `predictions.jsonl` / `predictions.zip`.
- IOAI 2025 Concepts: local offline TF-IDF clue generator with 118 hint descriptions, embedding shape `(118, 3415)`, valid row counts, valid zip packaging, but `Final Score: 0.0`.
- AICC: 27-task corpus exists; `deceptive-points`, `face-matching`, and `massive-problem` are imported. Only `deceptive-points` has submission-like CSV evidence. `face-matching` remains a CLIP/OOM debugging attempt, and `massive-problem` still has a `FileNotFoundError` path issue.
- Data preprocessing fluency drill: executed notebook with tabular/text/image preprocessing checks and tiny CNN logits, useful for basic shape discipline but not the current scoring bottleneck.

## Mandatory Coverage Buckets

- IOAI syllabus: incomplete. Stronger artifact evidence exists for Python/NumPy/Pandas, data processing, scikit-learn, PyTorch basics, tensor manipulation, supervised learning, neural networks/MLP, model evaluation, audio processing, pretrained audio encoder use, model finetuning, imitation learning, and baseline NLP embeddings. Weak or missing as contest artifacts: object detection, segmentation, stronger pretrained vision/text encoders, completed CLIP/vision-text encoder workflow, autoencoders/GANs/diffusion, RL/search beyond Home Task 2 behavior cloning, and broader official past IOAI task completion.
- Past IOAI tasks: underdone. IOAI 2025 Concepts has a valid-format but zero-score baseline. IOAI 2025 Chicken Counting, IOAI 2025 Radar/Restroom/Antique/Pixel, IOAI 2024 Help BOBAI, IOAI 2024 Lost in Hyperspace, and IOAI 2024 Madarian Cow remain unclosed or not recently advanced.
- IOAI 2026 home tasks: mandatory and still not closed. Home Task 1 has credible audio adaptation evidence. Home Task 2 has valid full prediction artifacts. Home Task 3 is the blocking gap: random baseline only, no completed improved all-dev strategy.
- AICC progress out of 27: 3 imported / 27; 3 attempted / 27 if counting setup/debugging; 1 completed / 27 with submission-like CSV evidence (`deceptive-points`). The corpus covers all 27 tasks but cataloging does not count as solved practice.
- Audio coverage: acceptable minimum exists through Home Task 1: loading, waveform/spectrogram inspection, AST input contract, finetuning/adaptation, validation metric, and old-vs-new retention. A short AST-vs-Whisper/HuBERT when-to-use note is still useful but lower priority than Home Task 3.

## Study Next

1. Finish a serious IOAI 2026 Home Task 3 strategy.
   - Target file/folder: `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb`.
   - Syllabus tag: NLP; Pre-trained Language Models; LLM inference; information gathering; search/decision strategy; Model Evaluation.
   - Competition pattern trained: constrained query budget, metric validation, baseline-first improvement, runtime control, model-output interpretation, failure analysis.
   - Required visible evidence: remove or bypass the interrupted all-animal precompute, run a deterministic or bounded strategy on all 150 `dev.csv` rows, print mean score, solved rate, mean queries, wall time, and at least five inspected failures, and beat the random baseline mean score `0.0161`.
   - Why this is the highest-value next move: all IOAI 2026 home/platform tasks are mandatory, and Home Task 3 is the only one still without a serious score-useful attempt.
   - Target schedule slot: 2026-07-26 first study block. No heavy work on 2026-07-31.

2. Only after Home Task 3 beats random, close one official past IOAI task with score usefulness or a tactical failure analysis.
   - Target file/folder: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb` and `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/`.
   - Syllabus tag: NLP; Text Encoders / embeddings; Language Modeling; Model Evaluation Metrics; Data Processing.
   - Competition pattern trained: official past-task submission format, metric validation, local offline fallback, error analysis.
   - Required visible evidence: improve the 30-example validation probe above `0.0` or save a concise failure-analysis note explaining why the offline TF-IDF clue generator fails; keep a valid `submission.zip` with 50 `clues_a.jsonl` rows and 100 `clues_b.jsonl` rows.
   - Why this is the second highest-value next move: official past IOAI work is mandatory, and Concepts already has the file contract wired; the missing part is score usefulness or a deliberate close.
   - Target schedule slot: 2026-07-26 second study block, only if Home Task 3 has a completed all-dev run beating random.

## Pass/Fail Check Before Next Run

PASS: Home Task 3 saved notebook has a completed all-150-dev run with mean score greater than `0.0161`, solved rate and mean queries printed, no `KeyboardInterrupt` as the latest strategy output, and at least five inspected failures. Stretch pass: Concepts improves above `0.0` on the validation probe or has a saved failure-analysis close plus valid output files.

FAIL: work goes into new AICC imports, Chameleon exploration, CEOAI-only review, passive syllabus reading, archive cleanup, or another Home Task 3 precompute that does not finish inside a contest-sized block.

## Avoid Until This Is Done

- Do not import more AICC problems before Home Task 3 has a serious bounded strategy.
- Do not continue Chameleon before Concepts is score-useful or formally closed.
- Do not polish Home Task 1 beyond a short AST-vs-Whisper/HuBERT note unless a bug breaks the saved metric.
- Do not redo Home Task 2 unless the existing `predictions.zip` stops validating.
- Do not use data preprocessing drills as the main study block now; they are useful fluency work but not the current scoring bottleneck.
- Do not study CEOAI syllabus as a primary target. CEOAI overlap counts only when tied to an IOAI artifact.
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
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline-Copy1.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_a.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_b.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/submission.zip`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/`
- `olympiads/competition_samples/raw/IOAI-2024-sparse/`
- `olympiads/aicc/`
- `olympiads/aicc/aicc_problem_corpus.md`
- `olympiads/aicc/deceptive-points/`
- `olympiads/aicc/face-matching/`
- `olympiads/aicc/massive-problem/`
- `olympiads/reviews/error_journal.jsonl`
- `olympiads/reviews/ioai_error_journal.jsonl`
