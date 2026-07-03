# CEOAI Section 1 Finish Plan: Search and RL

Date: 2026-07-03

## Verdict

You are still behind on CEOAI Section 1 until these rows have visible artifacts:

- `1(b)` Minimax and variations
- `1(c)` Monte Carlo method
- `1(d)` Markov Decision Processes
- `1(e)` Temporal Difference Learning

Existing evidence already helps:

- `1(a)` A* has a notebook artifact in the Basics exercises.
- `1(f)` Q-learning has a notebook artifact in this Search/RL folder.
- `1(e)` TD learning is partially covered by Q-learning, but it needs an explicit numeric update.

Today is not a theory day. The goal is to connect the whole first CEOAI section through one fixed game, one fixed gridworld, one comparison table, and one short research note.

Required route:

- Minimax uses tic-tac-toe.
- MDP, TD learning, and Q-learning use the same deterministic 4x4 gridworld.
- Monte Carlo uses the fixed three-step episode written below.
- Do not switch to another game, library environment, or dataset until these artifacts pass.

## Target Artifacts

Finish these in order.

| Artifact | File | Syllabus rows | Counts only if |
| --- | --- | --- | --- |
| Minimax notebook | `minimax_tictactoe.ipynb` | `1(b)` | shows the fixed tic-tac-toe board, candidate move scores, chosen move, and one explanation sentence |
| MDP/TD extension | `solution.ipynb` | `1(d)`, `1(e)`, `1(f)` | uses the fixed 4x4 gridworld and shows state/action/reward/next_state plus one numeric TD update |
| Monte Carlo note/table | `search_rl_comparison.md` or inside `solution.ipynb` | `1(c)` | distinguishes Monte Carlo from TD and Q-learning using the fixed three-step episode |
| Section 1 recognition table | `search_rl_comparison.md` | `1(a)`-`1(f)` | lets you classify A*, minimax, Monte Carlo, MDP, TD, and Q-learning problems fast |

If time is tight, do not polish. A rough executed notebook with visible outputs is better than a clean unread note.

## Time-Boxed Plan

### Block 1: Minimax Recognition and Artifact, 75 Minutes

Goal: understand adversarial search well enough to choose a move in one fixed tic-tac-toe position.

Required implementation contract:

- Board representation: length-9 tuple or list.
- Empty cell marker: `" "`.
- Maximizing player: `"X"`.
- Minimizing player: `"O"`.
- Terminal scores: `X` win = `+1`, draw = `0`, `O` win = `-1`.
- Candidate move output: one row per legal move with the resulting minimax score.
- Fixed starting board:

```text
X | O | X
O | X |  
  |   | O
```

Use cell indices:

```text
0 | 1 | 2
3 | 4 | 5
6 | 7 | 8
```

Research window, 15 minutes:

- Read only enough to answer:
  - What is a game tree?
  - What does the maximizing player do?
  - What does the minimizing player do?
  - What is alpha-beta pruning, and why does it return the same move faster?
- Stop reading when you can explain: "Minimax assumes the opponent also plays optimally."

Build/check artifact, 45 minutes:

- Use only the tic-tac-toe contract above.
- Show the fixed board position.
- List legal moves.
- Score each candidate move with minimax.
- Print the selected move.
- Add one sentence: why this move was selected.

Reflection, 15 minutes:

- Write three problem-recognition rules:
  - Use minimax when there is an opponent.
  - Use A* when the map/model is known and the target is shortest path.
  - Use RL when the agent must learn from rewards.

Pass condition:

- Opening the notebook immediately shows board state, scores, chosen action, and explanation.

### Block 2: Explicit MDP Contract, 35 Minutes

Goal: make the existing Q-learning notebook count more strongly for `1(d)`.

Required gridworld contract:

- Grid size: 4x4.
- State representation: `(row, col)` coordinates. Also show the integer mapping `state_id = row * 4 + col` for the Q-table.
- Start state: `(0, 0)`.
- Goal state: `(3, 3)`.
- Hole/blocked terminal states: `(1, 1)` and `(2, 1)`.
- Actions: `0=up`, `1=right`, `2=down`, `3=left`.
- Movement: deterministic; moving into a wall leaves the agent in the same state.
- Rewards: `+1` for reaching `(3, 3)`, `0` otherwise.
- Episode ends at the goal or a hole.
- Discount factor: `gamma = 0.95`.

Research window, 10 minutes:

- Find or write the definitions of:
  - state space
  - action space
  - transition
  - reward
  - terminal state
  - discount factor

Artifact work, 20 minutes:

In `solution.ipynb`, add one compact section:

| MDP component | Fixed 4x4 gridworld value |
| --- | --- |
| State | `(row, col)` plus `state_id = row * 4 + col` |
| Action | `0=up`, `1=right`, `2=down`, `3=left` |
| Reward | `+1` at `(3, 3)`, `0` otherwise |
| Transition | deterministic movement, wall keeps same state |
| Terminal condition | goal `(3, 3)` or holes `(1, 1)`, `(2, 1)` |
| Discount factor | `gamma = 0.95` |

Then add one concrete transition:

```text
state = ...
action = ...
reward = ...
next_state = ...
done = ...
```

Pass condition:

- The notebook names the MDP pieces using your actual environment, not generic definitions only.

### Block 3: TD Learning From One Numeric Update, 40 Minutes

Goal: turn Q-learning from "I ran a loop" into "I understand the update."

Research window, 10 minutes:

- Read only enough to answer:
  - What is the old estimate?
  - What is the target?
  - What is the temporal-difference error?
  - Why does Q-learning count as TD learning?

Artifact work, 25 minutes:

In `solution.ipynb`, use this exact transition:

```text
state = (3, 2)
action = right
reward = 1
next_state = (3, 3)
done = True
```

Fill this with actual Q-table values from the notebook:

```text
old_q = ...
reward = ...
best_next_q = ...
learning_rate = ...
discount = ...
target = reward + discount * best_next_q
td_error = target - old_q
new_q = old_q + learning_rate * td_error
```

Then write:

```text
Q-learning is TD learning because it updates after one step using reward plus an estimate of future value, not after waiting for the full episode return.
```

Pass condition:

- There is a numeric TD update with actual values from your notebook.

### Block 4: Monte Carlo vs TD vs Q-Learning, 35 Minutes

Goal: cover `1(c)` without overbuilding.

Research window, 15 minutes:

- Read about Monte Carlo prediction/control only at recognition level.
- Focus on the contrast:
  - Monte Carlo waits until the episode ends.
  - TD updates after each step.
  - Q-learning is an off-policy TD control method.

Artifact work, 15 minutes:

Use this fixed episode:

```text
S0 --a0,r=0--> S1 --a1,r=0--> S2 --a2,r=1--> terminal
```

For this example, use `gamma = 1.0` so the Monte Carlo return from `S0` is easy to inspect.

Write:

- Monte Carlo would update `S0` from the final observed return `G = 1`.
- TD would update after `S0 -> S1` using `0 + gamma * V(S1)`.
- Q-learning would update `Q(S0, a0)` using `0 + gamma * max_a Q(S1, a)`.

Pass condition:

- You can answer: "What information does this method need before it updates?"

### Block 5: One Section 1 Comparison Table, 30 Minutes

Goal: make the whole first syllabus section searchable in your own words.

Fill this table:

| Method | Syllabus | Problem type | Plans or learns? | Needs model/map? | Output |
| --- | --- | --- | --- | --- | --- |
| A* | `1(a)` | shortest path on a known grid | plans | yes | path, cost, expanded nodes |
| Minimax | `1(b)` | tic-tac-toe position with optimal opponent | plans | yes, game rules | move scores and chosen move |
| Monte Carlo | `1(c)` | episode-return estimation | learns | no transition model required | value estimate from full returns |
| MDP | `1(d)` | fixed 4x4 gridworld contract | describes problem | yes, if solving exactly | states, actions, rewards, transitions |
| TD learning | `1(e)` | step-by-step value update in gridworld | learns | no transition model required | updated value from reward plus estimate |
| Q-learning | `1(f)` | action-value learning in 4x4 gridworld | learns | no transition model required | Q-table and greedy policy |

Then classify these without notes:

| Prompt | Best method/category |
| --- | --- |
| Find the shortest route from `(0, 0)` to `(3, 3)` on the known 4x4 grid. | |
| Choose the best `X` move on the fixed tic-tac-toe board. | |
| Estimate `S0` from the complete `S0 -> S1 -> S2 -> terminal` episode. | |
| Describe the states, actions, rewards, transitions, and terminals of the 4x4 gridworld. | |
| Update a value after `(3, 2) --right,+1--> (3, 3)`. | |
| Learn best actions in the 4x4 gridworld from repeated trial episodes. | |

Pass condition:

- You can choose the right method before writing code.

## How To Research Efficiently

Use research only to unblock artifact creation.

For each topic, search or read until you can fill these four lines:

```text
What problem does it solve?
What input does it need?
What output does it produce?
How do I recognize it in a contest prompt?
```

Do not continue reading after those lines are filled unless the notebook artifact is blocked.

Suggested search phrases:

- `tic tac toe minimax X maximizing O minimizing terminal score`
- `4x4 gridworld MDP state action reward transition terminal discount`
- `temporal difference learning numeric update reward gamma next value`
- `Monte Carlo reinforcement learning full episode return S0 S1 terminal`
- `Q-learning TD target max next action numeric example`

Research stop rule:

- 15 minutes maximum per topic before returning to the artifact.
- Any definition you read must become a row, example, or numeric update in the notebook/table.
- Reading does not count unless it changes a saved artifact.

## Final Self-Test

Answer these from memory after finishing:

1. Why is A* not reinforcement learning?
2. Why is minimax not Q-learning?
3. What are the MDP components in your gridworld?
4. What is the TD target in your numeric update?
5. Why does Monte Carlo wait longer than TD before updating?
6. Why is Q-learning a TD method?
7. Given a new problem, what clue tells you whether it is search, adversarial search, or RL?

## Stop Condition For Today

Stop only when the Search/RL folder contains visible evidence for every CEOAI Section 1 row:

- `1(a)` A*: existing path/cost/expanded-count notebook
- `1(b)` Minimax: board, move scores, chosen move
- `1(c)` Monte Carlo: full-episode return comparison
- `1(d)` MDP: explicit state/action/reward/transition/terminal components
- `1(e)` TD: numeric update
- `1(f)` Q-learning: Q-table, rewards, final policy/actions

High-pressure target: finish the minimax notebook first, then make `solution.ipynb` show one real MDP transition and one numeric TD update before doing any more reading.
