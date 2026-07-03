# Sprint Add-On: A* Grid Search

## Why This Exercise Exists

You are not doing this to become a graph-theory expert.

You are doing this because CEOAI explicitly includes:

- `CEOAI 1(a)` A* algorithm and heuristics

And because A* is one of the fastest ways to test whether your current Python
basics are actually useful under competition pressure:

- tuples for states,
- dictionaries for best-known costs and parents,
- a priority queue,
- boundary checks,
- a clean function contract,
- debug prints that tell you where the search went wrong.

This is a good Lesson 1 add-on because the main difficulty is not advanced AI
theory. The main difficulty is basic implementation discipline.

## Time Box

Target: **45-60 minutes**.

If you are still polishing architecture after 60 minutes, you are doing the
wrong exercise. The goal is one working baseline.

## What You Should Build

Build a function that runs **A\*** on a small 2D grid with blocked cells.

Use the exact function contract:

```python
astar_grid(grid, start, goal) -> (path, path_cost, expanded_count)
```

Where:

- `grid` is a 2D structure representing free cells and blocked cells,
- `start` is a tuple `(row, col)`,
- `goal` is a tuple `(row, col)`,
- `path` is the list of visited coordinates from `start` to `goal`,
- `path_cost` is the total number of moves in that path,
- `expanded_count` is how many nodes you popped from the priority queue.

Use **4-direction movement only**:

- up
- down
- left
- right

Each move has cost `1`.

Use **Manhattan distance** as the heuristic:

```text
h(r, c) = abs(r - goal_r) + abs(c - goal_c)
```

## Required Inputs

Start with this exact toy grid:

```text
S . . # .
. # . # .
. # . . .
. . # # .
# . . . G
```

Interpretation:

- `S` = start
- `G` = goal
- `.` = free cell
- `#` = blocked cell

Use:

- `start = (0, 0)`
- `goal = (4, 4)`

You may represent the grid as:

- a list of strings,
- a list of lists,
- or a set of blocked coordinates

But keep the representation simple.

## Required Outputs

Your notebook or script must print:

1. The original grid.
2. The final path as a list of coordinates.
3. The final path cost.
4. The number of expanded nodes.
5. The same grid again, with the final path marked visually.

Example of the kind of output expected:

```text
path = [(0, 0), (0, 1), ... , (4, 4)]
path_cost = 8
expanded_count = 11
```

The exact path may differ if multiple shortest paths exist, but it must be
valid and obstacle-free.

## What Logic Must Exist

Your implementation must include these objects explicitly:

1. **Priority queue / open set**
   Stores candidates ordered by smallest `f = g + h`.

2. **`g_cost` dictionary**
   `g_cost[state]` = best known cost from start to this state.

3. **`parent` dictionary**
   `parent[child] = previous_state` so you can reconstruct the final path.

4. **Neighbor generation**
   From `(r, c)`, generate up/down/left/right and filter out:
   - cells outside the grid
   - blocked cells

5. **Path reconstruction**
   When goal is reached, walk backward through `parent` until `None`.

## Required Debug Prints

While testing, print at least:

- the node currently popped from the heap,
- its `g`,
- its `h`,
- its `f`,
- and any neighbor whose score gets improved.

Why: this turns A* from “magic” into bookkeeping you can inspect.

## Self-Checks

Your result is acceptable only if all of these are true:

- The returned path starts at `start`.
- The returned path ends at `goal`.
- Every consecutive pair of states differs by exactly one grid move.
- No path cell is blocked.
- `path_cost == len(path) - 1`.
- The algorithm returns `None` or a clear failure value if the goal is
  unreachable.

## Stretch Check

After the baseline works, change **one thing only**:

- add one new wall, or
- move the goal, or
- make the goal unreachable

Then rerun and confirm that the output changes in a sensible way.

Do not generalize into a full framework.

## Why This Is Useful For You

This exercise is not just about A*.

It also trains a competition reflex you will need in many CEOAI/IOAI tasks:

- define a clean input/output contract,
- build the smallest working baseline,
- inspect internal state with debug prints,
- verify correctness with explicit checks,
- stop after a valid result instead of endlessly “improving” the code.

If you cannot finish this, the issue is probably not “AI knowledge”.
The issue is usually one of:

- weak Python data-structure handling,
- unclear state representation,
- missing parent reconstruction,
- or not checking whether a better `g` value was found.

## Stop Condition

Stop when:

- the toy grid works,
- the printed path is valid,
- the path overlay is readable,
- and you can explain in one sentence:

```text
A* is Dijkstra-style shortest-path search that chooses the next node using
current cost plus heuristic estimate to the goal.
```

Do not spend extra time on weighted graphs, diagonal movement, or performance
optimizations today.
