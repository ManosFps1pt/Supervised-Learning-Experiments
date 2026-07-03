# Two-Hour RL Learning Sprint

## Goal

Learn enough reinforcement learning to build your first tiny RL notebook while
covering the CEOAI search/RL syllabus rows efficiently.

This is a learning block, not a test. The target is to understand the Python
imports, the environment API, the Q-table idea, and the difference between RL,
search, and game-tree methods.

## Syllabus Rows Covered

From `olympiads/ceoai_syllabus.md`:

- A* Algorithm and heuristics
- Minimax and variations
- Monte Carlo method
- Markov Decision Processes (MDPs)
- Temporal Difference Learning
- Q-learning

## Learning Milestones

By the end of the block, aim to produce these learning artifacts in
`solution.ipynb`:

- Import `numpy` and `matplotlib.pyplot`.
- Understand what an optional RL environment library such as `gymnasium` gives
  you.
- Create or inspect a tiny gridworld/FrozenLake-style environment.
- Create a Q-table with visible shape: `states x actions`.
- Run a small training loop, even if you need TODO scaffolding at first.
- Plot or print reward/success information.
- Print a final learned policy or best action per state.
- Write a short comparison table: A*, minimax, Monte Carlo, TD learning,
  Q-learning.

If something does not run, the learning artifact is the debugging note: what was
missing, what import failed, what shape was wrong, or what part of the RL loop
was unclear.

## Python Library Setup

Start in `solution.ipynb`.

The first code cell should be about imports only:

```python
import numpy as np
import matplotlib.pyplot as plt
```

What these libraries do:

- `numpy` gives you arrays. In this sprint, the Q-table is a NumPy array.
- `matplotlib.pyplot` lets you plot rewards or success rates.
- `random` can help with exploration, but NumPy can also do random choices.
- `gymnasium` is optional. It provides ready-made RL environments, but you do
  not need it if setup becomes slow.

Recommended rule:

- If `gymnasium` imports cleanly, you may use `FrozenLake-v1`.
- If it does not import cleanly, use a tiny hand-made gridworld in the notebook.

Do not spend more than 10 minutes fighting package installation. The concept is
more important than the library today.

## 0-15 Min: Imports And Syllabus Map

First run the import cell in `solution.ipynb`.

Then create the checklist:

| Topic | What you must recognize | Artifact |
| --- | --- | --- |
| MDP | state, action, reward, transition, discount | one gridworld diagram |
| Monte Carlo | learns from full episodes | one sentence comparison |
| TD learning | learns after each step | one update explanation |
| Q-learning | learns action values | trained Q-table |
| A* | shortest-path planning with a heuristic | classify one maze problem |
| Minimax | adversarial game-tree planning | classify one game problem |

Stop condition:

- You know what each imported library is for.
- You can explain why Q-learning is RL but A* is not.

## 15-35 Min: RL Contract

Learn only this contract first:

1. The agent observes a state.
2. The agent chooses an action.
3. The environment returns a next state, a reward, and whether the episode is
   done.
4. The agent updates its estimate of which actions are good.

Plain-language meaning:

- Supervised learning asks: "What label should this input have?"
- Reinforcement learning asks: "What action should I take now to get more
  future reward?"
- `Q[state, action]` means: "How good is this action from this state, including
  future consequences?"

Artifact:

- Write your own one-sentence definition of `Q[state, action]`.

## 35-75 Min: First RL Model

Build a small tabular Q-learning agent with scaffolded code.

Recommended environment:

- Best: a tiny 4x4 gridworld that you define yourself.
- Also acceptable: a local FrozenLake-style environment if already available.

Keep it simple:

- No PyTorch.
- No neural networks.
- No deep RL.
- No custom graphics.
- No complicated reward design.

Your notebook should show:

- number of states
- number of actions
- Q-table shape
- episode count
- learning rate
- discount factor
- exploration rate
- total reward or success rate
- final policy

Stop condition:

- You can run the notebook cells up to Q-table creation.
- If training is not finished, you know which TODO is blocking you.

## 75-95 Min: Q-Learning Update

Understand this update conceptually:

```text
new estimate = old estimate + learning_rate * (better target - old estimate)
```

The target is:

```text
reward + discount * best future Q-value
```

What this means:

- If the outcome was better than expected, increase the Q-value.
- If the outcome was worse than expected, decrease the Q-value.
- The discount controls how much future reward matters.
- The learning rate controls how quickly the table changes.

Artifact:

- Pick one transition from your environment and explain one Q-value update in
  words.

## 95-110 Min: Monte Carlo vs TD

Do not implement both unless you are already ahead.

Create this comparison:

| Method | When it learns | What it needs |
| --- | --- | --- |
| Monte Carlo | after the full episode ends | final return from the episode |
| TD learning | after each step | reward plus an estimate of the future |
| Q-learning | after each step | best estimated future action value |

Contest-level interpretation:

- Monte Carlo is easier to reason about but can be slower.
- TD learning is more immediate and usually more practical.
- Q-learning is a TD method for learning which action is best.

Stop condition:

- You can answer: "Why is Q-learning temporal-difference learning?"

## 110-120 Min: Search vs RL Recognition

Classify these:

| Situation | Best category |
| --- | --- |
| You know the maze map and need the shortest route. | A* |
| You play a small deterministic game against an opponent. | Minimax |
| You learn good actions through trial, error, and rewards. | RL |

Write three examples of your own:

- one A* problem
- one minimax problem
- one RL problem

Stop condition:

- You can decide whether a problem is search, game search, or RL before coding.

## Final Learning Check

Mark each row:

| Learning output | Done? |
| --- | --- |
| `numpy` imported and used | |
| `matplotlib.pyplot` imported or explained | |
| Optional `gymnasium` checked, or skipped intentionally | |
| Q-table created | |
| Agent training loop started | |
| Reward/success metric printed or planned | |
| Final policy printed, visualized, or scaffolded | |
| MDP terms explained | |
| Monte Carlo vs TD distinguished | |
| Q-learning update explained | |
| A* vs minimax vs RL classified | |

## What To Skip Today

Skip these unless explicitly required later:

- DQN
- policy gradients
- actor-critic methods
- neural-network approximators
- OpenAI Gym wrappers if setup becomes slow
- long Bellman-equation derivations
- from-scratch A* or minimax implementations

The efficient target is beginner-comfortable tabular Q-learning plus problem
classification.

## After The Sprint

If the first artifact works, the next useful block is:

1. Implement A* on a tiny grid maze.
2. Implement minimax or alpha-beta on tic-tac-toe.
3. Compare all three approaches in one note:
   search, adversarial search, and reinforcement learning.
