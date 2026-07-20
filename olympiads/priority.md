# IOAI Priority Handoff

## Status

Current date: 2026-07-19. IOAI 2026 arrival is 2026-08-02 in Astana. The last
real study day is 2026-07-30 because 2026-07-31 is a no-study day and the user
flies to Astana on 2026-08-01. Effective study days remaining after today: 11
calendar mornings/evenings if used well.

Primary target: official IOAI 2026 Individual Contest preparation. CEOAI and
regional tasks are now supporting practice only when they map clearly to
`olympiads/ioai_syllabus.md`.

Competition constraints to keep active:

- Main development environment: web-based JupyterLab.
- Secondary editor: VS Code offline, without direct GPU access.
- Platform: Yandex Contest for statements, datasets, submissions, and scores.
- Language/runtime: Python 3.13.
- Individual Contest LLM: officially provided Gemma 3, at most 1000 output
  tokens per query. External LLMs, coding agents, copilots, browser assistants,
  and external APIs are prohibited unless a task statement explicitly allows
  them.
- Hardware expectation: Ubuntu laptops without local GPUs; GPU training and
  evaluation through JupyterLab-backed training machines. Technical appendix
  currently mentions NVIDIA H200 MIG slices with an 18GB VRAM limit.
- Practice limit to simulate: 20-minute notebook runtime per task unless the
  task says otherwise, and up to 60 submissions per task.

## Study Next

1. Official IOAI 2026 at-home / platform-familiarization task.
   - Target: `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/`
     if assets are available.
   - Syllabus: direct IOAI task-style practice across the official syllabus.
   - Evidence: one executed notebook section, model/input/output shape checks,
     one metric or scorer call, and one submission-format or artifact check.
   - Why first: Contest 1 is connected to at-home tasks, so platform and task
     pattern familiarity has unusually high value.

2. Official IOAI past task replay.
   - Targets: IOAI 2025 `Chicken_Counting` or `Concepts`, then IOAI 2024
     `Help_BOBAI`.
   - Evidence: baseline that is not only file-format-valid but has one measured
     improvement or three inspected failure cases.
   - Why second: past IOAI tasks teach the exact baseline -> validation ->
     submission workflow better than generic tutorials.

3. Missing IOAI syllabus practicals.
   - Priority order: transformers/text encoders, object detection, segmentation,
     pretrained vision encoders, CLIP/vision-text encoders, audio encoders,
     autoencoders/GANs/diffusion.
   - Evidence: a small notebook cell block per topic with input contract,
     output contract, metric or visual sanity check, and when-to-use note.

4. Gemma 3 contest-assistance rehearsal.
   - Use a local Gemma model only as an approximation; the real contest LLM is
     integrated into the IOAI platform.
   - Practice with `max_new_tokens=1000`, short prompts, and manual
     verification. Do not let the local model write full solutions during mock
     contests.
   - Evidence: one prompt, the checked answer, and a note on what was useful or
     misleading.

## Daily Pass/Fail Check

PASS: the day produced an IOAI-visible artifact: executed notebook cells,
metric table, prediction/submission file, model-output sanity check, 20-minute
runtime check, syllabus coverage row, or checked Gemma-3 prompt reflection.

STRETCH PASS: the artifact maps to a named row in `olympiads/ioai_syllabus.md`
and includes a reusable debugging/reflex note.

FAIL: the day became passive reading, broad theory review, link collection, or
unsaved experimentation with no artifact.

## Avoid Until The Daily Artifact Exists

- Do not reorganize historical PDTN or CEOAI files.
- Do not read full official solutions before attempting a baseline.
- Do not use Codex/Copilot/external LLM help during strict mock-contest blocks.
- Do not tune models before validating data shape, metric direction, runtime,
  and submission format.
- Do not spend the last days implementing standard internals from scratch when
  the IOAI syllabus marks a method as practice/library use.

## Evidence To Recheck

- `olympiads/ioai_syllabus.md`
- `olympiads/notes/ioai_ceoai_environment.md`
- `olympiads/notes/ioai_contest_strategy.md`
- `olympiads/competition_samples/practice_queue.md`
- `olympiads/competition_samples/source_index.csv`
- `olympiads/competition_samples/raw/IOAI-2026-sparse/Home Task/`
- `olympiads/competition_samples/raw/IOAI-2025-sparse/`
- `olympiads/competition_samples/raw/IOAI-2024-sparse/`
- `olympiads/reviews/error_journal.jsonl`
