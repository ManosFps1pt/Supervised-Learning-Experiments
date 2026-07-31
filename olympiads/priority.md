# IOAI Priority Handoff
## Status
Date: 2026-07-30.
Preparation deadline: 2026-07-31. There is 1 calendar day remaining until departure, but 2026-07-31 is travel/logistics only. Treat 2026-07-30 as the last full study day. IOAI starts on 2026-08-02.

Pace since previous run: SLOW. Overall verdict: BEHIND for IOAI readiness by departure. Biggest bottleneck: IOAI 2026 Home Task 3 is edited for a bounded run, but there is not yet a fresh executed smoke/dev/test score artifact.

Baseline used for comparison: automation last run at 2026-07-30T05:36:24.853Z, current priority.md handoff content, git state, and visible working-tree changes. Git state at this run: olympiads/portable_ioai/tasks/home_task_3/Home-Task-3.ipynb modified, olympiads/priority.md modified, and scripts/monitor_gpu.bat untracked. Several non-git file-read commands failed with Windows process-start error -1073741502, so this handoff is deliberately conservative and does not claim any unverified notebook execution.

## New Since Previous Run
- Home Task 3 remains the active IOAI 2026 platform task. The portable notebook is modified and the current handoff indicates it was changed to keep the full animal pool while bounding the question pool.
- Current Home Task 3 plan uses questions from questions_pool.txt, not a hard-coded PRIORITY_QUESTIONS list.
- Current Home Task 3 plan keeps HOME3_MAX_QUESTIONS=32, reducing initial precompute from 1472 x 559 = 823,648 calls to about 1472 x 32 = 47,104 calls.
- Current Home Task 3 plan expects compressed cache reuse via home3_answer_table_1472x32.npz and a dev_smoke_10.csv smoke path before full dev/test scoring.
- No fresh executed Home Task 3 score table was verified in this run. Code/notebook edits count as setup, not completed study progress.
- Untracked scripts/monitor_gpu.bat may support runtime monitoring, but it is not IOAI evidence by itself.

Already counted before:
- IOAI 2026 Home Task 1: audio loading, waveform/spectrogram inspection, AST input contract, model adaptation, validation accuracy, and retention metric.
- IOAI 2026 Home Task 2: demos, rollout baseline, predictions.jsonl, predictions.zip, and validated JSONL-style output.
- Earlier IOAI 2026 Home Task 3: useful greedy/query artifact existed before, but the current bounded-question notebook still needs fresh executed evidence.
- IOAI 2025 Radar and Chicken Counting: score-useful official artifacts already counted.
- IOAI 2025 Pixel: format-valid submission evidence already counted, but scoring was blocked.
- IOAI 2025 Concepts: still weak; prior output scored 0.0 and remains a mandatory NLP/text gap after Home Task 3.
- AICC: previous count was 11 imported / 27, about 9 attempted / 27 if partials count, and about 6 completed / 27 with executed notebook plus submission-like CSV evidence.

## Mandatory Coverage Buckets
- IOAI syllabus: partial and behind. Home Task 3 maps to interactive inference, LLM-based judging, greedy decision/search strategy, output interpretation, metric validation, and runtime constraints. Existing counted artifacts cover some audio, imitation learning, CV segmentation/counting, CLIP-style inference, and classical workflow, but NLP/text encoders, object detection, autoencoders/GANs/diffusion, and broader RL/search remain weak.
- Past IOAI tasks: mandatory and incomplete. IOAI 2025 Radar and Chicken Counting have useful score evidence; Pixel has valid-format output but no score; Concepts remains the most important official NLP gap after Home Task 3.
- IOAI 2026 home tasks: high-priority. Home Task 3 must produce a bounded smoke/dev/test score artifact today; edits alone are not enough.
- AICC progress out of 27: previous evidence says 11 imported / 27, about 9 attempted / 27, about 6 completed / 27. Do not import more before Home Task 3 scoring evidence exists.
- Audio coverage: minimum evidence exists through Home Task 1, but audio is explicitly mandatory before departure. Add a tiny runnable audio recap only after Home Task 3 passes.

## Study Next
1. Finish the bounded Home Task 3 execution artifact.
   - Target file/folder: olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb or olympiads/portable_ioai/tasks/home_task_3/Home-Task-3.ipynb.
   - Syllabus tag: interactive inference; LLM-based judging; greedy search; model-output interpretation; evaluation metrics.
   - Competition pattern trained: constraints/runtime control, baseline-first modeling, metric validation, submission/platform evidence, cache reuse.
   - Required visible evidence: executed dev_smoke_10.csv result, created or loaded home3_answer_table_1472x32.npz, and full dev.csv plus test1.csv score table or a saved validation note explaining the exact blocker.
   - Why highest value: this is current IOAI 2026 home/platform work and directly reduces departure risk.
   - Target schedule slot: 2026-07-30 immediately. No heavy work on 2026-07-31.

2. Optional only after Home Task 3 passes: write one tiny audio recap artifact.
   - Target file/folder: an existing runnable audio notebook under olympiads/, preferably tied to Home Task 1 evidence.
   - Syllabus tag: audio loading; waveform/spectrogram or embeddings; model input contract; metric/output check.
   - Competition pattern trained: preprocessing/features, shape validation, model input/output interpretation.
   - Required visible evidence: loaded audio, waveform/spectrogram or embedding, tensor/dataframe shape, metric/output check, and a short when-to-use note.
   - Why highest value: audio was explicitly required before departure, and a compact runnable recap protects recall under contest pressure.
   - Target schedule slot: late 2026-07-30 only if Home Task 3 has a score artifact.

## Pass/Fail Check Before Next Run
- PASS: Home Task 3 has executed smoke output plus dev/test score tables from the bounded 32-question strategy.
- PASS: the answer cache home3_answer_table_1472x32.npz is created or loaded, and the notebook/output shows questions came from questions_pool.txt.
- PASS: if scoring fails, there is a saved blocker note with command, traceback/error, dataset split, and next concrete fix.
- FAIL: evidence is only code edits, cleared notebook outputs, or runtime setup.
- FAIL: the run starts full 1472 x 559 brute-force generation.
- FAIL: the next block goes to new AICC import, archive cleanup, CEOAI-only work, or broad reading before Home Task 3 score evidence exists.

## Avoid Until This Is Done
- Do not reduce the animal pool for final dev/test; hidden animals make that unsafe.
- Do not increase HOME3_MAX_QUESTIONS above 32 until smoke and at least one full score table finish.
- Do not reintroduce hand-coded question lists unless there is a contest-legal rule for deriving them from the provided pool.
- Do not precompute all 823,648 animal-question pairs today.
- Do not switch to AICC, Pixel polish, Concepts, or archive cleanup before Home Task 3 produces score evidence or a precise blocker note.
- Do not schedule heavy study on 2026-07-31; that date is for packing, offline files, account/platform sanity checks, travel, and rest.

## Evidence To Recheck
- olympiads/ioai_syllabus.md
- olympiads/priority.md
- olympiads/competition_samples/practice_queue.md
- olympiads/competition_samples/source_index.csv
- olympiads/competition_samples/task_cards/
- olympiads/competition_samples/task_cards/ioai_2025_concepts.md
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem1/Home-Task-1.ipynb
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/Home-Task-2.ipynb
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/predictions.jsonl
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem2/predictions.zip
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/Home-Task-3.ipynb
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/dataset/evaluate.py
- olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/problem3/dataset/interactor.py
- olympiads/portable_ioai/tasks/home_task_3/Home-Task-3.ipynb
- home3_answer_table_1472x32.npz
- dev_smoke_10.csv
- any Home Task 3 score tables, validation notes, outputs, screenshots, prediction files, and notebook execution counts
- olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb
- olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/concepts_prompt_contract.md
- olympiads/aicc/
- olympiads/reviews/error_journal.jsonl
