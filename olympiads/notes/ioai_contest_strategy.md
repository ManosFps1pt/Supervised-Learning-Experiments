# IOAI Contest Strategy

Date: 2026-07-19

## Default Loop

1. Read the task and identify the required output: code, predictions, model
   file, zip, CSV, JSONL, or notebook result.
2. Load the smallest slice of data and print shape, dtype, labels, missing
   values, and one raw example.
3. Build the simplest baseline that reaches the evaluator or submission format.
4. Validate metric direction, row count, column names, output dtype, and reload
   behavior before tuning.
5. Improve one lever at a time: feature processing, model family, pretrained
   encoder, augmentation, learning rate, regularization, or threshold.
6. Stop early enough to rerun the final notebook path and save the exact
   requested artifact.

## JupyterLab Discipline

- Practice in JupyterLab first because it is the official main development
  environment.
- Keep cells small: imports/setup, data load, sanity checks, baseline, metric,
  submission, final validation.
- Restart-and-run important notebooks during practice. A notebook that only
  works because hidden state exists is not contest-ready.
- Keep VS Code as a secondary offline editor for larger `.py` files, not as the
  main GPU training interface.

## Runtime And Submission Checks

- Simulate the expected 20-minute notebook runtime limit for serious practice.
- Save checkpoints or intermediate features only when the task permits and the
  final artifact can be regenerated.
- Before every submission, check:
  - expected filenames;
  - row count or array shape;
  - no missing IDs;
  - metric direction;
  - class label encoding;
  - numeric precision and dtype;
  - zip contents if a zip is required.
- Spend the last 10-15 minutes on validation, not on a new model idea.

## Limited Docs Practice

- Assume whitelist-only docs: Python, NumPy, pandas, scikit-learn, PyTorch,
  Matplotlib, SciPy, and Hugging Face documentation for approved models.
- Train the reflex: `dir(obj)`, `help(obj)`, `inspect.signature`, tiny input,
  output keys/shapes, then integration.
- Do not depend on Stack Overflow, GitHub search, random blog posts, Copilot, or
  external agents during mock contests.

## Gemma 3 Prompt Discipline

The real Individual Contest assistant is the official platform-integrated Gemma
3 with at most 1000 output tokens per query. Local Gemma practice is only an
approximation.

Good practice prompts:

- "Here is my traceback and tensor shapes. What is the most likely mismatch?"
- "Given this sklearn estimator and metric, what sanity checks should I run?"
- "Summarize the API contract of this model output from these printed keys."
- "List three likely causes of validation improving while leaderboard worsens."

Rules:

- Cap local practice with `max_new_tokens=1000`.
- Ask for debugging probes, not complete contest solutions.
- Verify every answer manually in the notebook.
- Keep a short note on what the model helped with and what it missed.
