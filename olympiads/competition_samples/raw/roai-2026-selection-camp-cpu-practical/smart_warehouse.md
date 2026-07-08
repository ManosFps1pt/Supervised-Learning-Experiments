# Smart Warehouse

- Source: https://judge.nitro-ai.org/competitions/roai-2025/lot-2-2026/3/view
- Competition: ROAI Selection Camp - CPU Practical Round
- Local status: public statement mirrored; test data, starter kit, and pre-judging script are listed on Nitro but not mirrored here.
- CEOAI tags: `1(d)`, `1(e)`, `1(f)`
- Priority: very high

## Task Type

Reinforcement learning / temporal-difference control in a small discrete grid environment called `SmartWarehouse-v0`.

The robot must:

1. move to the package;
2. pick it up;
3. move to the delivery cell;
4. deliver it before the battery or step limit runs out.

The hidden evaluator uses similar unseen warehouse scenarios. Some scenarios may include walls, danger cells, recharge cells, and slip probability.

## Environment Contract

Observation is a discrete integer encoding:

```text
(r, c, b, p)
```

- `r`: robot row;
- `c`: robot column;
- `b`: current battery level;
- `p`: package status, `0` for not carrying and `1` for carrying.

Actions:

```text
0 = up
1 = down
2 = left
3 = right
4 = pick up
5 = deliver
6 = recharge
```

Allowed public interface:

- `env.reset(seed=...)`;
- `env.step(action)`;
- `env.action_space.n`;
- `env.action_space.contains(action)`;
- `env.observation_space.n`;
- `env.max_steps`;
- `env.decode(observation)`.

Do not rely on private environment fields.

## Required Submission Interface

Submit both:

```text
solution.py
solution.pkl
```

`solution.py` must define:

```python
def td_update(v_s, reward, v_next, alpha, gamma, done):
    pass

def q_learning_update(q_s_a, reward, q_next, alpha, gamma, done):
    pass

def sarsa_update(q_s_a, reward, q_next_a_next, alpha, gamma, done):
    pass

def train_agent(env_factory, seed: int, episodes: int):
    pass

def select_action(agent, observation, deterministic: bool = True) -> int:
    pass
```

Terminal transitions must not bootstrap from the next state.

## Evaluation Contract

The evaluator calls:

```python
agent = train_agent(env_factory, seed=..., episodes=8000)
action = select_action(agent, observation, deterministic=True)
```

Runtime constraints from the public statement:

- CPU-only;
- `TRAIN_EPISODES = 8000`;
- `EVAL_EPISODES = 40`;
- training timeout: `12.0` seconds per hidden scenario;
- action timeout: `0.20` seconds per decision.

Scoring includes TD(0), SARSA, Q-learning, terminal-state handling, interface checks, hidden normalized return, and hidden success rate.

## Baseline Route

1. Implement the three update formulas first and unit-test terminal handling.
2. Train a tabular Q-learning agent with epsilon-greedy exploration.
3. Store Q-table and any metadata in `solution.pkl`.
4. Make `select_action(..., deterministic=True)` pure, fast, and exploration-free.
5. Validate returned actions are integers in `[0, 6]`.

## Completion Evidence

Save:

- `solution.py`;
- `solution.pkl`;
- update-formula test output;
- trained-policy evidence such as reward/success-rate table or sample decoded trajectory.
