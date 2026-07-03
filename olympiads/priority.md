# CEOAI Priority Handoff

Updated at: 2026-07-03

This file tells future agents what the user should learn next in strict
CEOAI exam mode.

Use current workspace evidence first. Count only executed notebooks, saved
metrics, submission-style outputs, explicit comparison tables, or checked model
outputs. Do not count reading, resource collection, or vague notes as progress.

## Overall Status

The user is improving but is still behind the CEOAI sprint target.

Why:

- NLP now has real artifacts.
- Basic PyTorch training/evaluation loops now have real artifacts.
- Search/RL is no longer empty because A* and tabular Q-learning now exist.
- The highest-value remaining gaps are still uncovered or only partially
  covered: minimax, Monte Carlo, explicit MDP reasoning, and any competition
  style CV artifact.

## Current Evidence Snapshot

These already count:

- `olympiads/IOAI Material/1. Basics/exercises/solution_5.ipynb`
  - CEOAI `1(a)` A* Algorithm and heuristics
  - visible evidence: valid path, path cost, expanded count
- `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/solution.ipynb`
  - CEOAI `1(f)` Q-learning
  - partial CEOAI `1(e)` Temporal Difference Learning
  - visible evidence: Q-table, rewards, final best actions
- `olympiads/IOAI Material/5. Natural Language Processing (NLP)/exercises/caramains_dril1.ipynb`
  - CEOAI `4(b)` TF-IDF style embeddings
  - IOAI text classification and metrics
- `olympiads/IOAI Material/5. Natural Language Processing (NLP)/exercises/caramanis_dril_2.ipynb`
  - CEOAI `3(c)` Transformer/BERT workflow
- `olympiads/IOAI Material/5. Natural Language Processing (NLP)/exercises/caramanis_dril_3.ipynb`
  - CEOAI `4(c)` T5 / language-model workflow
- `olympiads/IOAI Material/5. Natural Language Processing (NLP)/exercises/caramanis_drill4.ipynb`
  - submission-style prediction artifact
- `olympiads/IOAI Material/3. Neural Networks/exercises/solution2.ipynb`
  - CEOAI `3(a)` and `3(b)` baseline DL workflow
- `olympiads/IOAI Material/3. Neural Networks/exercises/solution3.ipynb`
  - CEOAI `3(a)` and `3(b)` baseline DL workflow with visible accuracy and boundary plot

These do not count enough yet:

- `olympiads/IOAI Material/3. Neural Networks/exercises/solution4.ipynb`
  - too thin; setup exists but not the intended finished probe artifact
- `olympiads/IOAI Material/1. Basics/exercises/copy_numpy_array_muscle_memory_drills.ipynb`
  - useful foundation practice, but low CEOAI sprint leverage now

## What The User Should Learn Next

### Priority 1: Minimax

Reason:

- It is still missing from the highest-priority CEOAI block:
  `1(b) Minimax and variations`.
- Search/RL is the sprint bottleneck. Closing this gap is worth more than
  adding another NLP or foundation artifact.
- It is fast to verify: one executed notebook can prove understanding.

Required artifact:

- one executed minimax notebook in
  `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/`
- visible move scores
- chosen move
- one example position
- one short sentence explaining why that move was selected

Pass condition:

- an agent can open the notebook and immediately see the game state, candidate
  scores, and final chosen action

### Priority 2: Explicit MDP and TD Learning

Reason:

- Current Q-learning evidence is real, but MDP and TD are still only partially
  evidenced.
- This is an easy way to turn partial Search/RL coverage into solid Search/RL
  coverage without starting a new large project.

Required artifact:

- extend
  `olympiads/IOAI Material/7. Reinforcement Learning and AI Search/exercises/solution.ipynb`
- add one explicit state-action-reward-next_state example
- add one written TD-style update using actual values from the run
- add one short note identifying the state space, action space, reward, and
  terminal condition

Syllabus tags:

- CEOAI `1(d)` Markov Decision Processes
- CEOAI `1(e)` Temporal Difference Learning
- CEOAI `1(f)` Q-learning

Pass condition:

- the notebook contains an explicit numeric update and clear MDP components,
  not just a trained table

### Priority 3: Monte Carlo vs TD vs Q-Learning Recognition

Reason:

- CEOAI includes `1(c) Monte Carlo method`, but there is no direct artifact yet.
- A short comparison artifact is cheap and improves recognition speed for
  contest questions.

Required artifact:

- one filled comparison table in the Search/RL area covering:
  - A*
  - minimax
  - Monte Carlo
  - TD learning
  - Q-learning
- for each method: problem type, whether it plans or learns, and what output
  it produces

Pass condition:

- another agent can classify a toy problem with that table without reading
  external notes

### Priority 4: One Competition-Style CV Baseline

Reason:

- CV is still empty in visible sprint artifacts.
- Once Search/RL is no longer the main hole, CV is the next biggest uncovered
  area by breadth.

Required artifact:

- one executed image-classification baseline notebook
- visible prediction batch or accuracy metric
- use libraries and pretrained components if they speed up output

Syllabus tags:

- CEOAI `5(b)` CNN architectures
- IOAI image classification / pre-trained vision encoders

Pass condition:

- a notebook shows real predictions or metrics, not only imports or dataset
  loading

## What Agents Should Not Push Next

Do not push these unless they directly unblock the priorities above:

- more broad reading
- more PDF or slide ingestion
- more NumPy muscle-memory drills
- from-scratch reimplementation of standard algorithms beyond the minimum
  needed artifact
- deep RL
- long theoretical derivations
- audio before Search/RL and CV are stronger

## Coaching Rule For Future Agents

If the user asks what to do next, give the smallest artifact that closes the
largest uncovered syllabus gap.

Current answer:

1. Finish minimax.
2. Make MDP and TD explicit in the RL notebook.
3. Add the Search/RL comparison table.
4. Only then move to CV.
