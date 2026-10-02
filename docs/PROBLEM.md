# Problem formulation

## Setting
- **Network**: undirected graph `G = (V, E)` (`NodeGraph`). Each edge has a base travel cost
  `c_e` (topology attribute `graph[u][v]["cost"]`, uniform in [10, 50] in the demo; models never see it).
- **Disaster**: every edge takes a discrete, ordinal damage state `s_e ∈ {0, …, K-1}`;
  `0` = undamaged, larger = worse. A full assignment is a `Scenario = dict[Edge, State]`.
- **Realised cost**: `c_e · m(e, s_e)`. The `CostFunction` is a distribution of damage
  multipliers (any scipy distribution or custom pdf, conditioned on `m ≥ 1`); a multiplier's
  **severity** is its quantile in it. State 0 → 1.0; damaged state k draws a severity uniformly
  from the k-th of `K-1` equal bands of [0, 1] and maps it to a multiplier — so higher states
  are strictly costlier. `main.py` uses the upper half of a lognormal(0, σ). Multipliers are
  drawn once per (edge, state) and cached in the model.

## Disaster model (spatial correlation)
Damage is spatially clustered. `MRFModel` places a **centered autologistic MRF** on the
*line graph* (nodes = road edges; distance = hop distance between edges):

```
log p(s_e | rest) ∝ [ log π_e(s_e) + s_e · D_e ] / T  (+ observation log-likelihood)
D_e = mean over neighbours n within cutoff of  β[d(e,n)-1] · (s_n − μ_n)
```

- `π_e`: per-edge prior (`prior_probs`), `μ_n`: prior mean state of neighbour n.
- `β`: coupling per hop distance; `len(β)` is the neighbourhood cutoff.
- `T`: temperature.
- Centering on `μ` (Caragea & Kaiser 2009) keeps β interpretable and avoids the field
  collapsing to one state.
- Inference and sampling are both done by Gibbs sampling (`_run_gibbs`).

## Observations
Held in an agent's `KnowledgeBase`, always as cost multipliers (`cost / c_e`):
- **Fully observed** (`fully_observed_edges: Edge → float`): exact multiplier. A discrete model
  inverts it to a state (the band holding its severity) and clamps it in Gibbs.
- **Partially observed** (`partially_observed_edges: Edge → float`): a *lower bound* on the
  multiplier; enters Gibbs as `log P(multiplier ≥ lower | state)`.

## Agents
- Each agent has an id, source, target, speed, its own **twin** `DisasterModel`, a
  `KnowledgeBase`, per-edge **expected multipliers** (posterior means), and a **heuristic map**
  (one joint multiplier sample consistent with its observations).
- Position is continuous along an edge: `(prev_node, next_node, offset)`.
- Planning (replans only on new information): Dijkstra with expected edge cost `c_e · E[m_e]` (for a discrete model, `E[m_e] = Σ_s b_e(s) · m(e, s)`). When replanning
  mid-edge, the agent compares continuing forward vs turning back to `prev_node`.
- Communication: `Message(source, target = ALL | set of ids, payload)`; payloads are
  `Update(edge, multiplier)` (share an observation) and `Heartbeat(sender)`.

## Goal
All agents reach their targets (`Stelllar.run_stellar` loops until every
`has_completed_goal`). Simulation clock with fixed `DELTA_TIME = 0.1`.

## Decisions
1. **Objective**: every agent wants to reach its own target and minimises its own travel time
   (self-interested). Evaluation reports several metrics: sum of travel costs, makespan
   (last arrival), mean arrival time.
2. **Communication**: global broadcast for now, but the message layer must stay pluggable
   (range-limited, delayed, lossy channels later). `Heartbeat` is only a liveness sanity
   check; it carries no information for the model.
3. **Observation**: an edge's multiplier is observed **exactly** once the whole edge has been
   traversed. While partway along it the agent only knows a **lower bound** (from the time
   spent on it so far), which is a partial observation. An agent can enter the same edge from
   **either end at different times** (go partway from u, turn back, later enter from v), so
   partial progress is tracked per edge **per entry side**, and the lower bound is derived
   from that history.
4. **Replanning**: only when new information arrives (own observation or a received `Update`).
5. **Twin models**: usually differ from the true model; model mismatch is a research question.
6. **Routing**: expected costs by default; the cost source (expected multipliers vs the
   heuristic-map sample) is a parameter.
7. **Costs** are time-based: traversal time of edge e is `c_e · m_e / speed`.

## Open questions
1. How partial progress from the two ends of an edge combines into one lower bound
   (e.g. max of the two sides, or something else).
