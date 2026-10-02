# Status & roadmap

_Last updated: 2026-10-02. Update this file at the end of each task._

## Done
- Shared types (`src/typedefs.py`).
- Cost function (`src/cost_function.py`): model-agnostic severity ↔ multiplier map over any
  scipy distribution, no built-in default (the caller picks it; `main.py` uses lognormal);
  state bands, likelihood and inverse live in `DiscreteStateModel`.
- `MRFModel`: centered autologistic MRF, vectorised colour-class Gibbs sampling, full + partial
  evidence, warm starts.
- Messages (`src/message.py`): `FullKnowledge`, `PartialKnowledge`, `Heartbeat`.
- Pluggable message channel (`src/channel.py`), `BroadcastChannel`.
- Cost-multiplier `DisasterModel` interface; discrete-state machinery in `DiscreteStateModel`;
  `Agent`/`Stellar` decoupled from MRF/state types (observations and messages are multipliers).
- Canonical edge orientation (`src/edges.py`), validated by the model; observed lower bounds
  enforced; prior expectation = model's own marginals; first update after agent init is cold.
- `Agent` + `Stellar` simulator (`src/scenario_generator.py`): fixed-tick movement with
  carry-over, time-based costs, exact / overdue-bound observations with per-end reach memory,
  replanning on new information with turnaround, `RoutingCost` parameter, heartbeats,
  `RunResult` metrics.
- Demo disaster generation + per-state plots, and a 40×40 three-agent simulation (`src/main.py`):
  common corner-to-corner trip, each twin with priors drawn from
  `Dirichlet(PRIOR_CONCENTRATION · true prior)` (`perturb_priors`) for route variety.

## TODO
- [ ] Experiment runner over seeds; model-mismatch experiments (twin priors/β/T ≠ truth).
- [ ] Baselines (e.g. single-agent CTP, no-communication, full-information optimum).
- [ ] More channels: range-limited, delayed, lossy.
- [ ] Tooling: dev deps (ruff, pyright/mypy, pytest), `tests/` (port the scratch checks:
      full-info lower bound, determinism, overdue/exact, meeting, turnaround, messages).

## Known gaps / limits
| Where | Issue |
|---|---|
| `src/cost_function.py` | Severity 1 maps to an infinite multiplier (lognormal tail); such an edge is never crossed and the run stops at `max_time`. |
| `src/scenario_generator.py` | With a thin-tailed prior, an overdue bound barely raises the expected rest of an edge, so agents almost never turn back on their own (they do on received knowledge). Expected behaviour, but worth knowing for experiments. |
| `src/typedefs.py` | `Observation` dataclass is unused. |
