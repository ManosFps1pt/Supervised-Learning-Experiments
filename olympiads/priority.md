# IOAI Priority Handoff

## Status

Current date: 2026-07-29. Departure/logistics date is 2026-07-31: 2 calendar days remain until that date. Including today there are 3 calendar dates left, but 2026-07-31 is travel/logistics only. The last full study day is 2026-07-30. IOAI starts on 2026-08-02.

Pace since previous run: SLOW. Overall verdict: behind for IOAI readiness by departure. Biggest bottleneck: official NLP/embedding/LLM practice is still not converted into a useful artifact because IOAI 2025 Concepts has no prompt contract and its previous output scored 0.0.

Baseline used for comparison: previous automation run at 2026-07-29T05:54:47.809Z and the previous `priority.md`, where Pixel needed `score.json` or `validation_note.md`, Concepts needed a prompt contract, Home Tasks 1-3 were already counted, IOAI 2025 Radar and Chicken Counting were already scored, and AICC progress was 11 imported / 27.

Git state at this run: `olympiads/priority.md` was modified before this run and has been overwritten for the current handoff. This run also added `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/validation_note.md`. No commit was made.

## New Since Previous Run

- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/Pixelv2.ipynb` was saved at 2026-07-29 13:34:47 UTC. It has 17/17 executed code cells, no saved errors, and outputs showing CLIP use, predicted label inspection, `698/698` export, and `Masks saved to submission.jsonl`.
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/submission.jsonl` was regenerated at 2026-07-29 13:33:36 UTC. Disk validation found 698 rows, 698 unique IDs, 0 duplicate IDs, 0 invalid coordinate rows, all coordinates within bounds, all coordinate orderings valid, and all crop areas equal to the 3136-pixel limit.
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/validation_note.md` was added by this run to preserve the validation result and metric blocker. The official metric did not produce `score.json`; system Python lacked `transformers`, and the repo venv metric command exited 1 without writing a score.
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb` was saved at 2026-07-29 08:58:48 UTC. New outputs show a weaker run with mean score `0.0161`, solved rate `2.0%`, mean queries `14.89 / 15`, then a greedy precompute interrupted by `KeyboardInterrupt`. Do not count this as progress; keep the older Home Task 3 artifact as the meaningful evidence.

Already counted before:

- IOAI 2026 Home Task 1: audio loading, waveform/spectrogram inspection, AST input contract `(1, 1024, 128)`, AST adaptation on old/new classes, validation accuracy `78.51%`, retention score `0.7839285714285715`.
- IOAI 2026 Home Task 2: 400 demos, 200 validation scenarios, 1600 test scenarios, 5327 state-action samples, MLP rollout, `success_rate: 0.92`, disk-validated `predictions.jsonl` and `predictions.zip`.
- IOAI 2026 Home Task 3: earlier useful greedy/query artifact with mean score `0.7764`, solved rate `99.3%`, and mean queries `10.93 / 15`.
- IOAI 2025 Chicken Counting: valid `submission.npz`, public/private score `0.36787944117144233`, prediction arrays `pred_a`/`pred_b` each `(100, 1, 180, 320)` float32.
- IOAI 2025 Radar: scored segmentation/signal run with weighted score `0.7482325087504285`, `submission_val.csv`, `submission_test.csv`, and `submission.zip`.
- IOAI 2025 Concepts: valid-format offline clue output exists from 2026-07-24, but score remained `0.0`; no prompt contract exists yet.
- AICC: unchanged since last run. 11 imported / 27; about 9 attempted / 27 if partial work counts; about 6 completed / 27 with executed notebook plus submission-like CSV evidence.

## Mandatory Coverage Buckets

- IOAI syllabus: partial and still behind. Strong runnable coverage exists for Python/NumPy/Pandas, data processing, classical ML basics, PyTorch basics, supervised learning, metrics, audio processing, pretrained audio encoders, model finetuning, imitation learning, query/search strategy, CV counting, segmentation/signal workflows, and CLIP/vision-text inference. Weak score-useful artifacts remain for official NLP/text encoders/LLM workflows, object detection, autoencoders/GANs/diffusion, and broader RL/search beyond the home tasks.
- Past IOAI tasks: mandatory and not complete enough. Scored evidence exists for IOAI 2025 Radar and Chicken Counting. Pixel now has a full notebook run and a saved format-valid/constraint-valid 698-row submission, but no `score.json`. Concepts remains the highest-value official-task gap because prior output scored 0.0 and no task contract was written. IOAI 2025 Restroom/Antique and IOAI 2024 Help BOBAI/Lost in Hyperspace/Madarian Cow remain unfinished or not recently closed.
- IOAI 2026 home tasks: acceptable minimum, but Home Task 3 was accidentally degraded in a recent notebook run. Keep the previous high-scoring Home Task 3 evidence; do not spend time retuning unless a platform/travel smoke check fails.
- AICC progress out of 27: unchanged. 11 imported / 27: `buried-fault`, `deceptive-points`, `essay-gap`, `face-matching`, `find-brain-tumors`, `massive-problem`, `oriented-ship`, `polarity`, `shuffled`, `the-defected-nuts`, `word-lookups`. About 9 attempted / 27; about 6 completed / 27 with executed notebook plus submission-like CSV evidence. Do not import more before Concepts has a usable contract.
- Audio coverage: acceptable minimum exists through Home Task 1 and portable smoke: audio loading, waveform/spectrogram inspection, AST preprocessing/input contract, adaptation/finetuning, validation metric, retention metric, and 16 kHz smoke clip. Add only a short when-to-use note if Concepts is closed early.

## Study Next

1. Decode IOAI 2025 Concepts into a task contract before writing more solution code.
   - Target file/folder: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/concepts_prompt_contract.md`, with reference to `olympiads/competition_samples/task_cards/ioai_2025_concepts.md` and `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb`.
   - Syllabus tag: NLP; Pre-trained Text Encoders; Language Modeling; Pre-trained Language Models; Model Evaluation Metrics; Data Processing.
   - Competition pattern trained: prompt-to-input/output contract extraction, metric identification, submission format recognition, baseline-first modeling.
   - Required visible evidence: a concise contract listing inputs, outputs, scoring, allowed tools, required files, existing failed baseline, next baseline route, and one remaining unclear point.
   - Why highest-value now: Concepts is the main unclosed official IOAI NLP artifact, and the current blocker is understanding the task contract rather than model choice.
   - Target schedule slot: 2026-07-29 evening or 2026-07-30 first block. No heavy work on 2026-07-31.

2. If and only if the Concepts contract is written, run one minimal Concepts baseline check.
   - Target file/folder: `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb`, `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/`, and either `score.json` or `validation_note.md`.
   - Syllabus tag: NLP; text embeddings; pretrained text encoders; LLM-style reasoning; metrics.
   - Competition pattern trained: baseline-first modeling, output-file validation, metric execution, error analysis from zero-score output.
   - Required visible evidence: regenerated clue files or submission zip plus a saved note explaining whether the new output improved over the previous 0.0 route or why it still fails.
   - Why second: a weak but understood NLP artifact is more valuable for IOAI readiness than another CV polish block.
   - Target schedule slot: 2026-07-30 main block after the contract exists.

## Pass/Fail Check Before Next Run

PASS: `concepts_prompt_contract.md` exists and clearly states Concepts inputs, outputs, scoring, allowed tools, submission files, baseline route, and one remaining unclear point.

STRONG PASS: Concepts also has regenerated output files plus `score.json` or `validation_note.md` comparing the new result to the previous 0.0 attempt.

FAIL: the next study block goes to new AICC imports, CEOAI-only review, passive syllabus reading, archive cleanup, Home Task 3 retuning, Radar/Chicken retuning, another Pixel notebook, or generic NLP reading instead of the Concepts contract.

## Avoid Until This Is Done

- Do not import more AICC problems before Concepts has a contract and one minimal rerun plan.
- Do not create another Pixel notebook; Pixel is format-valid but unscored, and more Pixel polish is lower value than Concepts.
- Do not retune Home Task 3 unless the travel/platform smoke check fails; preserve the older high-score evidence.
- Do not retune Chicken Counting or Radar before Concepts; both already have score-useful official evidence.
- Do not continue AICC `find-brain-tumors` before Concepts is unblocked.
- Do not study CEOAI syllabus as the primary target. CEOAI overlap counts only when tied to an IOAI artifact.
- Do not schedule heavy study on 2026-07-31; that date is for departure, packing, offline files, account/platform checks, and rest.

## Evidence To Recheck

- `olympiads/ioai_syllabus.md`
- `olympiads/priority.md`
- `olympiads/competition_samples/practice_queue.md`
- `olympiads/competition_samples/source_index.csv`
- `olympiads/competition_samples/task_cards/`
- `olympiads/competition_samples/task_cards/ioai_2025_concepts.md`
- `olympiads/competition_samples/task_cards/ioai_2025_chicken_counting.md`
- `olympiads/competition_samples/task_cards/ioai_2024_help_bobai.md`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem1/Home-Task-1.ipynb`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/Home-Task-2.ipynb`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/predictions.jsonl`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/predictions.zip`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts_baseline.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/concepts_prompt_contract.md`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_a.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/clues_b.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/out/submission.zip`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/Pixelv2.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/submission.jsonl`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/validation_note.md`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Pixel/score.json`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/Chicken_Counting.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/submission.npz`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Chicken_Counting/score.json`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Radar/Radar.ipynb`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Radar/submission_val.csv`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Radar/submission_test.csv`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Radar/submission.zip`
- `olympiads/competition_samples/raw/IOAI-2024-sparse/On-Site-Round/Help_BOBAI/Help_BOBAI.ipynb`
- `olympiads/portable_ioai/PORTABILITY_REPORT.md`
- `olympiads/portable_ioai/smoke-results/notebook-smoke-latest.json`
- `olympiads/portable_ioai/tasks/concepts_cpu/Concepts_CPU.ipynb`
- `olympiads/portable_ioai/tasks/help_bobai/Help_BOBAI.ipynb`
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
