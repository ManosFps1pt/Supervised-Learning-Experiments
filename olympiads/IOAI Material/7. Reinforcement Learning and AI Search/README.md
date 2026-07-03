# Reinforcement Learning and AI Search

## Purpose

This lesson covers the search and reinforcement-learning items from the CEOAI
syllabus that are not yet covered by the earlier IOAI material.

The priority is practical recognition and fast artifact production:

- decide whether a problem is search, adversarial search, or RL
- build one small tabular RL baseline
- explain the core update rule and metric
- avoid deep-RL theory until the basic syllabus rows are covered

## Core Path

Start with:

1. `exercises/01_two_hour_rl_syllabus_sprint.md`
2. `exercises/02_finish_ceoai_section_1_plan.md`

This first sprint is complete only when it produces a trained Q-table, a visible
reward or success metric, a final policy, and a short comparison of A*, minimax,
Monte Carlo, TD learning, and Q-learning.

The second sprint is complete only when the Search/RL folder has visible
evidence for every CEOAI Section 1 row: A*, minimax, Monte Carlo, MDP, TD
learning, and Q-learning. Use `exercises/search_rl_comparison.md` as the final
recognition table.

## Deferred Follow-Ups

After the first RL artifact works, add separate short drills for:

- A* on a tiny grid maze
- minimax or alpha-beta on tic-tac-toe
- value iteration on the same gridworld used for Q-learning
