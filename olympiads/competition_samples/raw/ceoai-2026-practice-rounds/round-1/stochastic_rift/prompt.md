# The Stochastic Rift

- Source: https://judge.nitro-ai.org/competitions/ceoai/ceoai-2026-practice-1/1/view
- Competition: CEOAI 2026 - Practice Round 1
- Local status: public statement mirrored; train data, test data, sample output, and starter kit are listed on Nitro but not mirrored here.
- CEOAI tags: `1(d)`, `1(e)`, `1(f)`
- Priority: very high

## Task Type

Offline reinforcement learning / stochastic MDP value estimation from historical transition logs.

The environment is a stochastic Markov Decision Process:

```text
M = <S, A, P, R, gamma>
```

- States are discrete node IDs.
- Actions are integers `0..3`.
- Transition probabilities and reward distributions are unknown.
- Discount factor is `gamma = 0.99`.

The goal is to estimate the optimal value `V*(s)` for query states using static logs and the provided environment structure.

## Dataset Contract

Public statement lists three files:

- `sector_logs.csv`: roughly 6000 shuffled transitions with columns `current_state, action, reward, next_state`.
- `query_states.csv`: query rows with columns `id, state_id`.
- `env.py`: environment class with state/action space and local simulation API.

The log is sparse and noisy. The same state-action pair may produce different next states and rewards.

## Output Contract

Submit one CSV named `predictions.csv` with exactly as many rows as `query_states.csv`:

```text
subtaskID,datapointID,answer
1,0,24.5012
1,1,-5.100
```

- `subtaskID`: always `1`.
- `datapointID`: the `id` from `query_states.csv`.
- `answer`: float estimate of `V*(state_id)`.

## Scoring

The judge computes MSE between submitted values and the true optimal values from a ground-truth solver. The public thresholds are:

- baseline MSE: `4000.0`;
- optimal MSE: `205.0`.

Score improves sharply only if the estimate beats naive sparse-log statistics.

## Baseline Route

1. Aggregate empirical transition and reward estimates by `(state, action)`.
2. Smooth missing or sparse state-action pairs using `env.py` simulation if available.
3. Run value iteration over the estimated MDP.
4. Export values for `query_states.csv`.
5. Validate row count and numeric float answers.

## Completion Evidence

Save `predictions.csv` plus a short note with MSE proxy/validation strategy, number of known state-action pairs, and value-iteration convergence evidence.
