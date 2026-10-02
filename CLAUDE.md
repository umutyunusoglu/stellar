# stellar

Research code for a **multi-agent Canadian Traveller Problem (CTP)** on a road graph after a
disaster. A disaster model damages edges (discrete damage states → cost multipliers); several
agents, each holding its own "digital twin" belief model, travel source → target, observe
edges, share observations by message, and replan.

Status: disaster model and agent/simulator loop work; experiments and baselines are next. See `docs/STATUS.md`.

## Run / lint
- Python ≥ 3.14 (uses PEP 695 `type` aliases), managed with **uv**.
- Imports are flat (`from typedefs import ...`), so **run from `src/`**:
  `cd src && uv run python main.py` → writes `disaster_state_{k}.png` into `src/`, then runs and prints a small simulation.
- Lint: `uv run ruff check src` (rules in `pyproject.toml`: E,F,I,UP,B,SIM,C4,RET,PTH,RUF).
- No tests or type-checker config yet.

## Layout (`src/`)
| File | Responsibility |
|---|---|
| `typedefs.py` | All shared type aliases (`Node`, `Edge`, `State`, `Scenario`, `Beliefs`, `KnowledgeBase`, ...) |
| `constants.py` | `NEG_INF` |
| `cost_function.py` | `CostFunction`: wraps any scipy distribution of multipliers (conditioned on ≥ 1); monotone `multiplier(severity)` / `severity(multiplier)`, no states. |
| `disaster_model.py` | `DisasterModel` interface only (cost multipliers — all `Agent`/`Stellar` see) |
| `discrete_model.py` | `DiscreteStateModel` (priors over states, equal severity bands per damaged state, multiplier cache, states ↔ multipliers) and `MRFModel` (centered autologistic MRF, vectorised colour-class Gibbs) |
| `edges.py` | `canonical(u, v)` edge orientation, `canonical_line_graph(graph)` |
| `message.py` | Inter-agent `Message` with `FullKnowledge`/`PartialKnowledge`/`Heartbeat` payloads, `ALL` broadcast target |
| `channel.py` | Pluggable `Channel` protocol; `BroadcastChannel` (lossless, next tick) |
| `scenario_generator.py` | `Agent` (twin model, knowledge, observation, replanning) and `Stellar` fixed-tick simulator → `RunResult` |
| `main.py` | Demos: 40×40 disaster plot per state; 40×40 three-agent corner-to-corner simulation with perturbed twin priors |

## Conventions
- New shared types go in `typedefs.py` as PEP 695 `type X = ...` (or `NewType` for dicts that need a distinct name).
- Dataclasses everywhere; private fields prefixed `_`, derived state via `field(init=False)` + `__post_init__`.
- Docstrings use `Params:` / `Returns:` sections; inline explanatory comments use `##`.
- Randomness always through an injected, seeded `random.Random` (or `np.random.Generator`) for reproducibility.
- Edges are `tuple[Node, Node]` on an **undirected** graph, always **canonical** `(min, max)` — build with `edges.canonical`; the edge list/order comes from `canonical_line_graph(graph).nodes`. Models reject non-canonical/unknown edges.
- Agents/simulator talk to models only through the `DisasterModel` interface, in cost multipliers; base costs live on the topology (`graph[u][v]["cost"]`).
- Pass the **cumulative** `KnowledgeBase` to `DisasterModel.update`, never just the newest observation.

## Don't
- Don't run scripts from the repo root (imports break).
- Don't commit generated PNGs, `__pycache__/`, or `.idea/`.
- Don't change code when the user only asked for review/planning — confirm first.

## More context
- `docs/PROBLEM.md` — problem formulation and open modelling questions
- `docs/ARCHITECTURE.md` — data flow and invariants
- `docs/STATUS.md` — roadmap and known gaps (keep this updated after each task)
- `docs/REFERENCES.md` — literature
