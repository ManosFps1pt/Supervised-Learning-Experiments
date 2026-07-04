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
| Find the shortest route from `(0, 0)` to `(3, 3)` on a known 4x4 grid. | A* | It plans the shortest path when the map and goal are already known. |
| Choose the best `X` move on the fixed tic-tac-toe board. | Minimax | It looks to the future assuming the opponent is optimal |
| Estimate `S0` from complete `S0 -> S1 -> S2 -> terminal` episodes. | Monte Carlo | It learns from full episodes |
| Describe the state/action/reward/transition structure of the fixed 4x4 gridworld. | MDP | It specifies the world definition: states, actions, rewards, transitions, and terminals. |
| Improve a value after `(3, 2) --right,+1--> (3, 3)`. | TD learning | It updates after one transition using reward plus an estimate of future value. |
| Learn best actions in the fixed 4x4 gridworld from rewards without knowing the full transition table. | Q-learning | It learns action values from trial-and-error rewards without needing the full model. |

## Monte Carlo vs TD Example

Toy episode:

```text
S0 --a0,r=0--> S1 --a1,r=0--> S2 --a2,r=1--> terminal
```

Use `gamma = 1.0` for this table. Fill this after research:

| Method | When it updates | What information it uses |
| --- | --- | --- |
| Monte Carlo | After the whole episode reaches terminal. | The full observed return, such as `G = 1` from the completed episode. |
| TD learning | After each step, such as right after `S0 -> S1`. | The immediate reward plus a bootstrap estimate like `0 + gamma * V(S1)`. |
| Q-learning | After each step, such as right after `S0 -> S1`. | The immediate reward plus the best estimated next action value, `0 + gamma * max_a Q(S1, a)`. |

## One-Sentence Contracts

- A*: Use A* when the map is known and the goal is to plan the shortest path to a target.
- Minimax: Use minimax when there is an optimal opponent and you need the best move under perfect play.
- Monte Carlo: Use Monte Carlo when you want to learn from complete episode returns after the episode finishes.
- MDP: Use an MDP when you need to specify a decision-making world in terms of states, actions, rewards, transitions, and terminals.
- TD learning: Use TD learning when you want to update a value after one step using reward plus an estimate of future value.
- Q-learning: Use Q-learning when you want to learn action values and a greedy policy from rewards without a full transition model.
