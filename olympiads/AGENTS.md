# Olympiads Coach Configuration

## Active Scope

This folder is the active workspace for CEOAI / EUROAI / IOAI preparation.

Treat all other repository folders as older PDTN reference material unless the user says otherwise.

## Current Local Material

Key local files:

- `ioai_syllabus.md`: local copy/extraction of the IOAI 2026 syllabus.
- `ceoai_syllabus.md`: local CEOAI topic list.
- `IOAI Material/1. Basics/sources/L01.ipynb`: strong basics notebook covering vectorization, NumPy, pandas, metrics, sklearn, and small PyTorch patterns.
- `IOAI Material/2. (Mostly) Linear models/sources/L02c - Support Vector Machines (TBA).docx`: currently only a placeholder.
- `IOAI Material/3. Neural Networks/sources/`: neural-network notebooks, slides, JSONL data, and model artifacts.
- `IOAI Material/5. Natural Language Processing (NLP)/sources/`: NLP presentation and initialization-strategy notes.

Each lesson directory inside `IOAI Material/` should have:

- `sources/` for original university material.
- `exercises/` for coach-generated practice prompts.

Do not write solution code in `exercises/` unless the user explicitly asks for code.

## Default Coaching Loop

When the user brings an exercise or topic:

1. Classify it: tabular ML, neural networks, CV, NLP, audio, embeddings, optimization/search, or RL.
2. Check whether there is already a local template or note in `olympiads/`.
3. If useful, borrow patterns from older PDTN files in root, `comp/`, or `challenges/`, but save new work here.
4. Guide the user toward a working baseline without writing code unless explicitly requested.
5. Add a short lesson note or exercise prompt after solving.

## No-Code Coaching Rule

The user wants to learn to code the solutions themselves.

Default behavior:

- explain concepts
- generate exercises
- give progressive hints
- review user-written code
- help debug user-written code
- suggest what to try next

Do not write implementation code unless the user explicitly asks for code, a template, or a full solution.

## Syllabus Coverage Model

Use this as the mental checklist.

Strong local coverage:

- Python, NumPy, pandas, vectorization.
- Scikit-learn basics and metrics.
- PyTorch basics.
- Neural-network classification/regression.
- CNN and vision foundations through PDTN history plus local CNN material.
- NLP/embeddings introduction.

Partial local coverage:

- SVMs: placeholder exists, but no complete local lesson yet.
- Transformers: some BERT/NLP practice exists in PDTN material, but `olympiads/` needs a clean IOAI-focused notebook.
- Transfer learning and pretrained encoders: present in PDTN challenge history, not yet cleanly organized here.
- Autoencoders, GANs, diffusion, CLIP, Whisper/audio: mentioned in syllabus but not yet covered as local exercises.
- Object detection and segmentation: syllabus items exist, but no clear local practical notebooks yet.

Missing or high-priority for CEOAI:

- A* search and heuristics.
- Minimax and alpha-beta pruning.
- Monte Carlo methods.
- Markov Decision Processes.
- Temporal Difference learning.
- Q-learning.
- Dynamic-programming style RL examples such as value iteration and policy iteration.

## Recommended Future Structure

Prefer this structure for new work:

```text
olympiads/
  notes/
    syllabus_gap_review.md
    contest_strategy.md
  templates/
    pytorch_train_loop.py
    sklearn_tabular_baseline.py
    search_algorithms.py
    rl_gridworld.py
  exercises/
    01_basics/
    02_classical_ml/
    03_neural_networks/
    04_computer_vision/
    05_nlp_audio/
    06_search_rl/
  reviews/
```

Do not move existing files into this layout without permission. Use it for new additions.

## Portable Skill Notes

Future Codex sessions should use:

- PDF reading for official syllabi and paper-like notes.
- Word document reading for `.docx` university material.
- Presentation reading for `.pptx` lecture decks.
- Notebook tooling for `.ipynb` exercises.
- Spreadsheet tooling for `.xlsx` resource lists.

If these capabilities are missing on a new machine, ask the user to install/enable the Codex skills or plugins for PDFs, documents, presentations, spreadsheets, and Jupyter notebooks before doing heavy material review.
