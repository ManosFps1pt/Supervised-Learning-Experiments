# Olympiads Repo Review

Date: 2026-06-30

## Summary

The repo is in a good state as a personal competition archive, but the active `olympiads/` folder is still early-stage compared with the amount of solved PDTN material elsewhere in the repository.

The strongest thing about the overall repo is not polish. It is the amount of solved practice and contest-pattern memory: PyTorch loops, image classifiers, adversarial image edits, CAPTCHA segmentation, coordinate-to-RGB regression, embedding ranking, and offline notes for fast retrieval.

For CEOAI / EUROAI / IOAI, the next step is to turn `olympiads/` into a clean coaching layer that references the older PDTN work without being buried by it.

## Official Context Checked

Primary sources checked:

- IOAI official site: https://ioai-official.org/
- IOAI 2026 syllabus page: https://ioai-official.org/republic-of-kazakhstan/syllabus-2026/
- IOAI 2026 syllabus PDF: https://ioai-official.org/wp-content/uploads/2025/10/Syllabus.pdf
- IOAI 2026 contest rules: https://ioai-official.org/republic-of-kazakhstan/2026-contest-rules/
- IOAI regional olympiads page: https://ioai-official.org/regional-oai/

Important current facts from official IOAI pages:

- IOAI 2026 is scheduled for Astana, Kazakhstan, August 2-8, 2026.
- The official syllabus separates knowledge into theory, practice, and both.
- The individual contest is Python/Jupyter-based.
- The core contest AI/ML libraries are `torch` and `scikit-learn`; TensorFlow and Keras are not available according to the IOAI 2026 rules.
- Contest tasks can require code, trained models, predictions, and local inference.
- Internet and pretrained model access are limited during the contest.
- Regional olympiads sit under the IOAI ecosystem; the first European Olympiad in AI is listed by IOAI as EOAI in Serbia in 2027.

Note: the local file `ceoai_syllabus.md` appears useful for Central European Olympiad preparation, but I did not find a matching official CEOAI syllabus page during this pass. Treat it as local/team-provided syllabus material unless a primary source is added.

## Current `olympiads/` Condition

### Strong

- The IOAI 2026 syllabus is present locally.
- There is a concise CEOAI topic list locally.
- University material exists in multiple formats: notebooks, PDF, DOCX, PPTX, JSONL, and model files.
- `1. Basics/sources/L01.ipynb` looks especially valuable: it covers practical vectorization, NumPy, pandas, metrics, sklearn, and small model utilities.
- The neural-network folder has classification/regression/structure notebooks plus slides.
- The NLP folder has a substantial language-model presentation and an initialization-strategy DOCX.
- Every lesson directory should now keep original material in `sources/` and coach-generated practice prompts in `exercises/`.

### Partial

- The `2. (Mostly) Linear models` folder is underdeveloped. The SVM document is currently only `TBA`.
- There is no clean local index explaining the order in which the university material should be studied.
- There are no obvious local notebooks yet for search/RL topics from the CEOAI syllabus.
- Vision, NLP, transformers, transfer learning, and adversarial examples are strongly represented in PDTN history, but not yet distilled into clean `olympiads/` templates.
- The folder contains many formats, but no material inventory that says which file teaches which syllabus item.

### Missing

High-priority missing topics:

- A* search and heuristic design.
- Minimax, alpha-beta pruning, and game search.
- Monte Carlo simulation/search basics.
- MDPs, value iteration, and policy iteration.
- Temporal Difference learning.
- Q-learning.
- Object detection practical baseline, even if only YOLO-style usage notes.
- Image segmentation practical baseline, especially U-Net.
- Autoencoder notebook.
- CLIP / vision-text encoder exercise.
- Whisper or audio-classification/transcription exercise.
- Diffusion and GAN practical/theory notes.

## Recommendation

Do not try to reorganize the whole repo first. The current archive has value because it preserves the path that got you first place in PDTN.

Instead, build a clean active layer inside `olympiads/`:

1. Add a syllabus tracker.
2. Add clean templates.
3. Add one exercise per missing topic.
4. After every solved problem, write a short reusable note.
5. Link back to PDTN history only when it gives a useful pattern.

## Suggested Next Files To Create

```text
olympiads/notes/syllabus_gap_review.md
olympiads/notes/contest_strategy.md
olympiads/templates/sklearn_tabular_baseline.py
olympiads/templates/pytorch_train_loop.py
olympiads/templates/search_algorithms.py
olympiads/templates/rl_gridworld.py
olympiads/exercises/06_search_rl/README.md
```

## First Practical Sprint

If preparing for CEOAI first, start with search/RL because that is the largest mismatch between the current repo and the CEOAI syllabus.

Recommended sprint:

1. Implement BFS, Dijkstra, and A* on grid mazes.
2. Implement minimax and alpha-beta on tic-tac-toe or a small impartial game.
3. Implement value iteration on a gridworld MDP.
4. Implement Q-learning on the same gridworld.
5. Write one comparison note: when a problem is search, optimization, supervised ML, or RL.

If preparing for IOAI first, start with clean versions of existing PDTN patterns:

1. `torch` training loop template.
2. `sklearn` tabular baseline.
3. CNN/transfer-learning notebook.
4. BERT/transformers text-classification notebook.
5. Embedding retrieval/ranking notebook.

## Risks

- The repo can become hard to navigate if PDTN history and active olympiad preparation are mixed.
- Notebook exports can create duplicated and noisy `.py` files.
- Generated datasets and answer files may distract from reusable learning material.
- Some local markdown has encoding artifacts from notebook/PDF conversion.
- Missing search/RL could become costly for CEOAI even if IOAI preparation is strong.

## Working Principle

The repo should become a memory palace for solved exercises:

- problem statement
- baseline
- bug encountered
- final trick
- reusable template
- syllabus topic covered

That is exactly the kind of structure an AI coach can use well across machines and future sessions.

## Coaching Constraint

The AI coach should not write solution code unless explicitly asked.

For generated exercises, prefer task statements, expected behavior, hints, and self-checks. The user should write the implementation.
