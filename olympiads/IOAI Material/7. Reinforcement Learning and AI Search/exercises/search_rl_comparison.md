# Search and RL Comparison Table

Use this as the final recognition artifact for CEOAI Section 1.

Required route:

- A*: known 4x4 grid shortest-path framing.
- Minimax: tic-tac-toe with `X` maximizing and `O` minimizing.
- Monte Carlo: fixed three-step episode below.
- MDP/TD/Q-learning: fixed 4x4 gridworld from `02_finish_ceoai_section_1_plan.md`.

## Method Table

| Method | Syllabus | Problem type | Plans or learns? | Needs model/map? | Output |
| --- | --- | --- | --- | --- | --- |
| A* | `1(a)` | shortest path from `(0, 0)` to `(3, 3)` on a known 4x4 grid | plans | yes | path, cost, expanded nodes |
| Minimax | `1(b)` | choose the best `X` move on a fixed tic-tac-toe board | plans | yes | candidate move scores and chosen move |
| Monte Carlo | `1(c)` | estimate value from complete `S0 -> S1 -> S2 -> terminal` episodes | learns | no | return-based value estimate |
| MDP | `1(d)` | describe the fixed 4x4 gridworld | describes problem | yes, for exact planning | states, actions, rewards, transitions, terminals |
| TD learning | `1(e)` | update after one gridworld transition | learns | no | value update from reward plus future estimate |
| Q-learning | `1(f)` | learn actions in the fixed 4x4 gridworld | learns | no | Q-table and greedy policy |

## Recognition Drill

| Prompt | Best method/category | Why |
| --- | --- | --- |
| Find the shortest route from `(0, 0)` to `(3, 3)` on a known 4x4 grid. | | |
| Choose the best `X` move on the fixed tic-tac-toe board. | | |
| Estimate `S0` from complete `S0 -> S1 -> S2 -> terminal` episodes. | | |
| Describe the state/action/reward/transition structure of the fixed 4x4 gridworld. | | |
| Improve a value after `(3, 2) --right,+1--> (3, 3)`. | | |
| Learn best actions in the fixed 4x4 gridworld from rewards without knowing the full transition table. | | |

## Monte Carlo vs TD Example

Toy episode:

```text
S0 --a0,r=0--> S1 --a1,r=0--> S2 --a2,r=1--> terminal
```

Use `gamma = 1.0` for this table. Fill this after research:

| Method | When it updates | What information it uses |
| --- | --- | --- |
| Monte Carlo | | |
| TD learning | | |
| Q-learning | | |

## One-Sentence Contracts

- A*:
- Minimax:
- Monte Carlo:
- MDP:
- TD learning:
- Q-learning:
