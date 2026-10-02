# Status & roadmap

_Last updated: 2026-10-02. Update this file at the end of each task._

## Done
- Shared types (`src/typedefs.py`).
- Cost function (`src/cost_function.py`): model-agnostic severity ↔ multiplier map over any
  scipy distribution, no built-in default (the caller picks it; `main.py` uses lognormal);
  state bands, likelihood and inverse live in `DiscreteStateModel`.
- `MRFModel`: centered autologistic MRF, Gibbs sampling, full + partial evidence, warm starts.
- Messages (`src/message.py`).
- Cost-multiplier `DisasterModel` interface; discrete-state machinery in `DiscreteStateModel`;
  `Agent`/`Stelllar` decoupled from MRF/state types (observations and messages are multipliers).
- Canonical edge orientation (`src/edges.py`), validated by the model; observed lower bounds
  enforced; prior expectation = model's own marginals; first update after agent init is cold.
- Demo disaster generation + per-state plots (`src/main.py`).

## In progress
- `Agent` (`src/scenario_generator.py`): routing on expected cost, knowledge updates.
- `Stelllar` simulator loop.

## TODO
- [x] Decide the modelling questions (see Decisions in `docs/PROBLEM.md`).
- [ ] Decide how per-side partial progress combines into one lower bound (`docs/PROBLEM.md`).
- [ ] Agent movement per tick (`speed`, `offset`, `DELTA_TIME`), arrival detection.
- [ ] Exact observation after a full traversal; lower bound from partial traversal, tracked per
      edge per entry side → `KnowledgeBase` → `update()` → replan (only on new information).
- [ ] Time-based edge costs (`c_e · m_e / speed`); routing cost source (expected vs sample) as a parameter.
- [ ] Message handling: apply `Update` payloads; pluggable channel (broadcast now; range-limited,
      delayed, lossy later); `Heartbeat` as a liveness sanity check only.
- [ ] Turn-around logic in `_generate_route`.
- [ ] Metrics (total cost, makespan, #replans) and experiment runner with seeds.
- [ ] Baselines (e.g. single-agent CTP, no-communication, full-information optimum).
- [ ] Tooling: `.gitignore`, dev deps (ruff, pyright/mypy, pytest), `tests/`, package layout.

## Known gaps / suspicious spots (not yet fixed)
| Where | Issue |
|---|---|
| `src/scenario_generator.py:58-68` | TODO: knowledge-base update and message consumption should be merged; partial observations not handled. |
| `src/scenario_generator.py:110-114` | Backward (turn-around) branch unfinished (`tmp = self._prev_node`). |
| `src/scenario_generator.py:123-127` | `consume_messages` filters by target but never uses payloads. |
| `src/scenario_generator.py:130-132` | `has_completed_goal` flagged as possibly wrong (uses `_prev_node`). |
| `src/scenario_generator.py:136` | Class name typo `Stelllar`. |
| `src/scenario_generator.py:157-165` | `run_stellar` busy-loops forever: no setup, movement, or clock advance. |
| `src/scenario_generator.py:3-5` | Import order and `typing.Callable` would fail ruff (`I`, `UP035`). |
| repo root | No `.gitignore`; generated PNGs, `__pycache__/`, `.idea/` are untracked noise. |
