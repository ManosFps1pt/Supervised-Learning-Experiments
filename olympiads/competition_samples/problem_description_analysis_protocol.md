# Problem Description Analysis Protocol

Use this during the emergency CEOAI practice block when a problem statement feels unfamiliar.

## Non-Negotiable Loop

1. Read one real problem description.
2. Write your own analysis before looking for a solution.
3. Identify input, output, metric, baseline, and validation.
4. State the first 15 minutes of physical work.
5. Get teacher correction.
6. Either run the first baseline step or move to the next description with the correction saved.

## What The Student Must Say

For every problem, answer these in plain language:

```text
What is the task asking me to produce?
What files/data do I expect to receive?
What exact file/table/object do I need to output?
How is it scored?
What dumb baseline could produce a valid output?
What part of the statement is unclear?
What would I do in the first 15 minutes?
```

## Teacher Correction Standard

The correction should be direct and specific:

```text
Correct:
Missing:
Wrong or risky assumption:
Why it matters:
Better first move:
Reusable reflex:
```

Do not give full solution code unless explicitly requested. The goal is to train recognition and action, not to replace the attempt.

## First-15-Minutes Checklist

Default move for practical ML contest tasks:

```text
1. Locate train/test/sample submission/evaluator files.
2. Print file names, shapes, columns, and two examples.
3. Identify the target and metric.
4. Produce a dumb valid output.
5. Validate row count, column names, file names, array names, and zip/JSON/CSV structure.
6. Save one note about what the baseline ignores.
```

If there is no dataset yet, the first move is statement parsing:

```text
1. Find the required output format.
2. Find the scoring section.
3. Find constraints and hidden traps.
4. Translate the task into input -> model/process -> output.
5. Decide what minimal valid artifact would count.
```

## Today's Starting Point

Start with:

```text
olympiads/competition_samples/raw/IOAI-2025-sparse/Individual-Contest/Concepts/Concepts.ipynb
```

Reason: the format contract already works, but the saved strategy scored `0.0`. This is the right kind of failure to analyze: not "can I create a file?", but "did I understand what the task rewards?"
