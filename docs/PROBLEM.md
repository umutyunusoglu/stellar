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
- Position: at a node, or on edge `(entry, far)` heading to one of its ends, `pos` time units
  from `entry`. Per edge it remembers the furthest time reached from each end (`reach`).
- Planning (replans only on new information): Dijkstra with estimated edge time
  `c_e · m̂_e / speed`, where `m̂_e` is the exact value if known, else the posterior mean
  (`RoutingCost.EXPECTED`, default) or the heuristic-map sample (`RoutingCost.SAMPLE`).
  Mid-edge it compares going back (`pos + d(entry)`, known exactly) with going on
  (`max(T̂_e − pos, 0) + d(far)`); ties keep the heading.
- Replan triggers: received knowledge, an edge fully explored (far end reached or stretches
  meet), an edge overdue (time spent > estimated time → new lower bound). At most one model
  update per batch of news.
- Communication: `Message(source, target = ALL | set of ids, payload)` through a pluggable
  `Channel` (`BroadcastChannel`: lossless, next tick). Payloads: `FullKnowledge(edge,
  multiplier)` (exact), `PartialKnowledge(edge, multiplier)` (lower bound; receivers keep the
  max, since bounds from different agents may overlap) and `Heartbeat(sender)`.

## Goal
Every agent reaches its target. `Stellar.run()` advances a fixed-tick clock
(`delta_time = 0.1`) until all have arrived or `max_time`, and returns a `RunResult`
(per-agent arrival time, replans, turnarounds, observations; makespan, total, mean).

## Decisions
1. **Objective**: every agent wants to reach its own target and minimises its own travel time
   (self-interested). Evaluation reports several metrics: sum of travel costs, makespan
   (last arrival), mean arrival time.
2. **Communication**: global broadcast for now, but the message layer must stay pluggable
   (range-limited, delayed, lossy channels later). `Heartbeat` is only a liveness sanity
   check; it carries no information for the model.
3. **Observation**: damage is **uniform along an edge**. The agent does not know how far
   along an edge it is, only how long it has spent there; it does **remember the points it
   has visited**. An agent can enter the same edge from **either end at different times**
   (go partway from u, turn back, later enter from v). Per edge, keep the longest time spent
   from each end, `t_u` and `t_v` (repeat entries from the same end overlap, so take the max).
   - **Partial** (stretches don't meet): they are disjoint pieces of a uniform edge, so
     `m_e ≥ (t_u + t_v) · speed / c_e` — a lower bound.
   - **Exact**: on reaching the far end, or on reaching a point already visited from the other
     end (the stretches meet). Then the edge has been covered exactly once and
     `m_e = (t_u + t_v) · speed / c_e`, with `t_v` the time until the meeting point.
4. **Replanning**: only when new information arrives (own observation or received knowledge).
5. **Twin models**: usually differ from the true model; model mismatch is a research question.
6. **Routing**: expected costs by default; the cost source (expected multipliers vs the
   heuristic-map sample) is a parameter.
7. **Costs** are time-based: traversal time of edge e is `c_e · m_e / speed`.
