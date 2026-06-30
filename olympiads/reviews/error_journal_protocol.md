# Olympiad Error Journal Protocol

Purpose: turn every practice error into a reusable contest reflex without taking solution ownership away from the student.

## Core Rule

Codex may explain the error, suggest debugging probes, and help the user reason. Codex should not write solution code unless the user explicitly asks.

The logged fix must describe what the user actually did, not what Codex would have done.

## Workflow

1. Capture the error from the traceback, notebook output, script output, or user description.
2. Ask what the user was trying to do if the intent is unclear.
3. Explain the error meaning and likely cause.
4. Suggest small checks before fixes: shapes, dtypes, columns, nulls, target leakage, metric direction, tensor dimensions, device, file paths, and submission format.
5. Let the user attempt the fix.
6. After the user reports the fix, append one JSON object to `error_journal.jsonl`.
7. If the fix is unknown, set `status` to `open` and leave `user_fix` empty.

## JSONL Schema

Each line in `error_journal.jsonl` is one JSON object.

Required fields:

```json
{
  "schema_version": "1.0",
  "status": "open|resolved",
  "date": "YYYY-MM-DD",
  "competition_context": "IOAI|CEOAI|EUROAI|practice|unknown",
  "exercise": "",
  "file": "",
  "cell_or_location": "",
  "error_type": "",
  "error_message": "",
  "category": "",
  "pressure_tag": "",
  "what_user_was_trying": "",
  "real_cause": "",
  "user_diagnosis_steps": "",
  "user_fix": "",
  "memory_rule": "",
  "next_drill": ""
}
```

Recommended categories:

- syntax
- import
- path
- shape
- dtype
- pandas
- sklearn
- pytorch
- metric
- leakage
- submission
- logic
- environment
- pressure

Recommended pressure tags:

- rushed
- tired
- overconfident
- confused
- stuck
- unknown
- none

## Pattern Reviews

After roughly 5 to 10 entries, update `error_pattern_review.md` with:

- recurring issue
- likely underlying cause
- contest reflex to memorize
- next concrete drill

Keep reviews short and actionable.
