import math
import random
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
import numpy.typing as npt

from constants import NEG_INF
from cost_function import CostFunction
from disaster_model import DisasterModel
from edges import canonical
from typedefs import (
    Beliefs,
    CostScenario,
    DistanceMatrix,
    Edge,
    ExpectedMultipliers,
    HopDistance,
    KnowledgeBase,
    LogStateWeights,
    Scenario,
    State,
    StateDistribution,
)


@dataclass(slots=True)
class DiscreteStateModel(DisasterModel):
    """
    A disaster model whose edges take discrete, ordinal damage states.

    Subclasses reason in states only. This layer translates between states
    and the cost multipliers of the DisasterModel interface. State 0 is
    undamaged, with multiplier exactly 1. The damaged states split the cost
    function's severity range [0, 1] into equal bands, worse states taking
    higher bands, so every state-level operation reduces to the cost
    function's two maps:

    - a state's multiplier is drawn once per edge, from a severity uniform
      in its band, and cached;
    - an observed multiplier is clamped to the state whose band holds its
      severity;
    - a lower bound cuts each band at the bound's severity, which gives the
      likelihood, and redraws any multiplier below the bound from the part
      of its band above it, so posterior multipliers respect the
      observation.

    attributes:
        prior_probs: Per-edge prior distribution over damage states. Every
            edge must cover states 0..K-1, and be canonical.
        cost_function: How damage multipliers are distributed.
        multiplier_rng: Source of randomness for drawing multipliers. Kept
            apart from rng so the state draws do not depend on it.
    """

    prior_probs: dict[Edge, StateDistribution]
    cost_function: CostFunction
    multiplier_rng: random.Random = field(default_factory=random.Random, kw_only=True)

    _edges: list[Edge] = field(init=False)
    _n_states: int = field(init=False)
    _cost_multipliers: dict[tuple[Edge, State], float] = field(init=False)
    _observed: dict[Edge, float] = field(init=False, default_factory=dict)
    _full_states: dict[Edge, State] = field(init=False, default_factory=dict)
    _lower_bounds: dict[Edge, float] = field(init=False, default_factory=dict)
    _conditioned: dict[tuple[Edge, State], float] = field(
        init=False, default_factory=dict
    )
    _expected: ExpectedMultipliers | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.prior_probs:
            raise ValueError("prior_probs is empty")

        self._edges = list(self.prior_probs.keys())
        flipped = [e for e in self._edges if e != canonical(*e)]
        if flipped:
            raise ValueError(
                f"{len(flipped)} edges are not canonical, e.g. {flipped[0]}"
            )

        expected = set(self.prior_probs[self._edges[0]])
        for edge, prior in self.prior_probs.items():
            if set(prior) != expected:
                raise ValueError(f"prior for {edge} covers a different state set")
        self._n_states = len(expected)
        if expected != set(range(self._n_states)):
            raise ValueError("states must be 0..K-1")
        if self._n_states < 2:
            raise ValueError("need at least one damaged state")

        ## One vectorised call: a custom scipy distribution can be slow per
        ## call, since it inverts its cdf numerically.
        keys = [
            (edge, state) for edge in self._edges for state in range(1, self._n_states)
        ]
        severities = [self.multiplier_rng.uniform(*self._band(s)) for _, s in keys]
        multipliers = self.cost_function.multiplier(severities).tolist()
        self._cost_multipliers = dict.fromkeys(
            ((edge, 0) for edge in self._edges), 1.0
        )
        self._cost_multipliers.update(zip(keys, multipliers, strict=True))

    @abstractmethod
    def generate_states(self) -> Scenario:
        """
        Returns:
            - Scenario: the damage state of every edge in one scenario drawn
              from the model, ignoring any observations
        """

    @abstractmethod
    def prior_run(self) -> tuple[Scenario, Beliefs]:
        """
        Returns:
            - Scenario: one joint draw with no observations
            - Beliefs: the model's own per-edge marginals with no
              observations. These need not equal prior_probs, since the
              model may couple and reweight edges.
        """

    @abstractmethod
    def update_states(
        self, full_states: dict[Edge, State], lower_bounds: dict[Edge, float]
    ) -> tuple[Scenario, Beliefs]:
        """
        Params:
            - full_states: the known state of every fully observed edge
            - lower_bounds: a lower bound on the multiplier of every
              partially observed edge

        Returns:
            - Scenario: a scenario consistent with the observations
            - Beliefs: per-edge posterior distribution over damage states
        """

    @abstractmethod
    def sample_states(
        self, full_states: dict[Edge, State], lower_bounds: dict[Edge, float]
    ) -> Scenario:
        """
        Params:
            - full_states: the known state of every fully observed edge
            - lower_bounds: a lower bound on the multiplier of every
              partially observed edge

        Returns:
            - Scenario: a fresh joint draw consistent with the observations
        """

    @override
    def generate_disaster(self) -> CostScenario:
        return {
            edge: self.get_cost_multiplier(edge, state)
            for edge, state in self.generate_states().items()
        }

    @override
    def update(
        self, observations: KnowledgeBase
    ) -> tuple[CostScenario, ExpectedMultipliers]:
        unknown = [
            edge
            for edge in (
                *observations.fully_observed_edges,
                *observations.partially_observed_edges,
            )
            if edge not in self.prior_probs
        ]
        if unknown:
            raise ValueError(
                f"{len(unknown)} observed edges are unknown, e.g. {unknown[0]}; "
                "edges must be canonical"
            )

        self._observed = dict(observations.fully_observed_edges)
        self._full_states = {
            edge: self._inverse(multiplier)
            for edge, multiplier in self._observed.items()
        }
        self._lower_bounds = dict(observations.partially_observed_edges)
        self._condition_on_lower_bounds()

        scenario, beliefs = self.update_states(self._full_states, self._lower_bounds)

        self._expected = {
            edge: self._mean_multiplier(edge, beliefs[edge]) for edge in self._edges
        }
        return self._posterior_multipliers(scenario), dict(self._expected)

    @override
    def expected_multiplier(self, edge: Edge) -> float:
        if self._expected is None:
            self._run_prior()
        return self._expected[edge]  # pyright: ignore[reportOptionalSubscript]

    @override
    def sample(self) -> CostScenario:
        ## The first query before any update pays for one cold run, and both
        ## the prior marginals and this draw come out of it.
        if self._expected is None:
            return self._posterior_multipliers(self._run_prior())
        return self._posterior_multipliers(
            self.sample_states(self._full_states, self._lower_bounds)
        )

    def _run_prior(self) -> Scenario:
        """Compute and cache the prior expected multipliers.

        Done once, lazily, since it costs a full cold run of the model.

        Returns:
            The joint draw made along the way.
        """
        scenario, beliefs = self.prior_run()
        self._expected = {
            edge: self._mean_multiplier(edge, beliefs[edge]) for edge in self._edges
        }
        return scenario

    def _condition_on_lower_bounds(self) -> None:
        """Redraw every possible multiplier that sits below an observed bound.

        A state that cannot reach the bound gets zero posterior weight, so it
        is left alone. A redrawn value is kept while it stays above the bound,
        and bounds only rise, so it is redrawn at most once per new bound.
        """
        for edge, lower in self._lower_bounds.items():
            for state in range(self._n_states):
                if self.state_log_likelihood(state, lower) == NEG_INF:
                    continue
                if self._state_multiplier(edge, state) < lower:
                    self._conditioned[(edge, state)] = self._sample_at_least(
                        state, lower
                    )

    def _band(self, state: State) -> tuple[float, float]:
        """Severity band of a damaged state.

        Params:
            state: A damaged state, 1..K-1.

        Returns:
            The (low, high) severities of its band.
        """
        n_damaged = self._n_states - 1
        return (state - 1) / n_damaged, state / n_damaged

    def state_log_likelihood(self, state: State, lower: float) -> float:
        """log P(multiplier >= lower | state).

        Params:
            state: The damage state.
            lower: An observed lower bound on the multiplier.

        Returns:
            The log-probability; NEG_INF when the state cannot reach it.
        """
        if state == 0:
            return NEG_INF if lower > 1.0 else 0.0

        lo, hi = self._band(state)
        u = self.cost_function.severity(lower)
        if u <= lo:
            return 0.0
        if u >= hi:
            return NEG_INF
        return math.log((hi - u) / (hi - lo))

    def _inverse(self, multiplier: float) -> State:
        """Recover the state whose band holds an observed multiplier.

        Params:
            multiplier: An exact observed multiplier.

        Returns:
            The state.
        """
        ## Only state 0 maps to 1; every damaged band lies strictly above it.
        if multiplier <= 1.0:
            return 0

        n_damaged = self._n_states - 1
        ## Clamping absorbs rounding at the outer edges.
        band = math.ceil(self.cost_function.severity(multiplier) * n_damaged)
        return min(max(band, 1), n_damaged)

    def _sample_at_least(self, state: State, lower: float) -> float:
        """Draw a multiplier for a state, conditioned on reaching a bound.

        Params:
            state: A state that can reach the bound.
            lower: The observed lower bound.

        Returns:
            A multiplier in the state's band that is at least lower.
        """
        if self.state_log_likelihood(state, lower) == NEG_INF:
            raise ValueError(f"state {state} cannot reach multiplier {lower}")
        if state == 0:
            return 1.0

        lo, hi = self._band(state)
        lo = max(lo, self.cost_function.severity(lower))
        severity = self.multiplier_rng.uniform(lo, hi)
        ## The cdf/ppf round trip can land a hair under the bound.
        return max(float(self.cost_function.multiplier(severity)), lower)

    def _state_multiplier(self, edge: Edge, state: State) -> float:
        """Multiplier of an edge in a state, after lower-bound conditioning."""
        conditioned = self._conditioned.get((edge, state))
        return (
            conditioned
            if conditioned is not None
            else self.get_cost_multiplier(edge, state)
        )

    def _multiplier(self, edge: Edge, state: State) -> float:
        """Posterior multiplier of an edge in a state.

        Observed edges keep the multiplier that was actually observed. The
        cache may hold a different value for the same state, since every
        model instance draws its own.
        """
        observed = self._observed.get(edge)
        return observed if observed is not None else self._state_multiplier(edge, state)

    def _mean_multiplier(self, edge: Edge, probs: StateDistribution) -> float:
        """Expected multiplier of an edge under a distribution over states.

        Params:
            edge: The edge whose multiplier is requested.
            probs: Distribution over the edge's damage states.

        Returns:
            The probability-weighted mean of its posterior multipliers.
        """
        return sum(p * self._multiplier(edge, s) for s, p in probs.items())

    def _posterior_multipliers(self, scenario: Scenario) -> CostScenario:
        """Turn a posterior scenario into multipliers.

        Params:
            scenario: The damage state of every edge.

        Returns:
            The cost multiplier of every edge.
        """
        return {edge: self._multiplier(edge, state) for edge, state in scenario.items()}

    def sample_from_log_weights(self, weights: LogStateWeights) -> State:
        """Draw one state from unnormalised log-weights.

        Params:
            weights: Mapping from state to its unnormalised log-weight.

        Returns:
            The sampled state.
        """
        top = max(weights.values())
        exps = {s: math.exp(w - top) for s, w in weights.items()}
        total = sum(exps.values())
        threshold = self.rng.random() * total

        cumulative = 0.0
        for state, val in exps.items():
            cumulative += val
            if cumulative >= threshold:
                return state

        return next(reversed(exps))

    def sample_from_probs(self, probs: StateDistribution) -> State:
        """Draw one state from a normalised probability distribution.

        Params:
            probs: Mapping from state to its probability. Sums to 1.

        Returns:
            The sampled state.
        """
        threshold = self.rng.random()
        cumulative = 0.0
        for state, p in probs.items():
            cumulative += p
            if cumulative >= threshold:
                return state
        return next(reversed(probs))

    def get_cost_multiplier(self, edge: Edge, state: State) -> float:
        return self._cost_multipliers[(edge, state)]


@dataclass(slots=True)
class MRFModel(DiscreteStateModel):
    """
    Markov Random Field disaster model with centered autologistic coupling.

    Damage states are ordinal: 0 is undamaged and larger indices mean more
    severe damage. An edge is pushed towards heavier damage when its
    neighbourhood carries more damage than the priors alone would predict,
    and towards lighter damage when it carries less.

    The coupling is centered on the prior expectation, following the
    centered autologistic model of Caragea and Kaiser (2009). Without that
    centering the coupling term only ever adds, so the field has no
    restoring force: weak beta does nothing at all and strong beta drives
    every edge to the same state. Centering keeps beta interpretable as the
    effect of a neighbourhood deviating from its baseline, independent of
    how many neighbours the cutoff admits.

    attributes:
        distances: Hop distances between every pair of edges, at least up
            to the cutoff.
        temperature: Sampling temperature. Lower values sharpen the state
            distribution.
        beta: Coupling strength per hop distance. Entry d - 1 applies to
            neighbours at distance d, so its length sets the cutoff.
        n_iter: Sweeps used for a cold run, starting from the priors.
        burn_in: Sweeps discarded before counting, on a cold run only.
        warm_iter: Sweeps used when resuming from the previous chain state.
    """

    distances: DistanceMatrix
    temperature: float
    beta: npt.NDArray[np.float64]
    n_iter: int = field(default=1000)
    burn_in: int = field(default=200)
    warm_iter: int = field(default=50)

    _neighbours: dict[Edge, dict[Edge, HopDistance]] = field(init=False)
    _log_prior: dict[Edge, LogStateWeights] = field(init=False)
    _mu: dict[Edge, float] = field(init=False)
    _beta: list[float] = field(init=False)
    _last_state: Scenario | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super(MRFModel, self).__post_init__()

        missing = [e for e in self._edges if e not in self.distances]
        if missing:
            raise ValueError(f"distances is missing {len(missing)} edges")

        cutoff = len(self.beta)
        self._neighbours = {
            u: {v: dist for v, dist in row.items() if u != v and 1 <= dist <= cutoff}
            for u, row in self.distances.items()
        }

        self._beta = [float(b) for b in self.beta]
        self._log_prior = {e: self._field(e) for e in self._edges}
        self._mu = {e: self._expected_state(e) for e in self._edges}

    def _field(self, edge: Edge) -> LogStateWeights:
        """Turn an edge's prior into a log-space external field.

        Params:
            edge: The edge whose prior is requested.

        Returns:
            Mapping from state to its log-prior.
        """
        return LogStateWeights(
            {
                state: math.log(max(prob, 1e-7))
                for state, prob in self.prior_probs[edge].items()
            }
        )

    def _expected_state(self, edge: Edge) -> float:
        """Mean damage state of an edge under its prior alone.

        This is the baseline the coupling is centered on, so a neighbour
        sitting exactly at its prior expectation exerts no pull.

        Params:
            edge: The edge whose prior expectation is requested.

        Returns:
            The prior mean of the edge's damage state.
        """
        return sum(s * p for s, p in self.prior_probs[edge].items())

    def _deviation(self, edge: Edge, scenario: Scenario) -> float:
        """Measure how far an edge's neighbourhood departs from baseline.

        Positive when neighbours carry more damage than their priors
        predict, negative when they carry less. Averaging over the
        neighbourhood keeps the scale independent of the cutoff, so beta
        does not need recalibrating when the cutoff changes.

        Params:
            edge: The edge whose neighbourhood is measured.
            scenario: The current configuration of the whole network.

        Returns:
            The distance-weighted mean deviation of the neighbours.
        """
        neighbours = self._neighbours[edge]
        if not neighbours:
            return 0.0

        return sum(
            self._beta[dist - 1] * (scenario[neighbour] - self._mu[neighbour])
            for neighbour, dist in neighbours.items()
        ) / len(neighbours)

    def _run_gibbs(
        self, full_states: dict[Edge, State], lower_bounds: dict[Edge, float]
    ) -> tuple[Scenario, Beliefs]:
        """Run the Gibbs chain over all edges.

        Observed edges are clamped and never resampled. The final
        configuration is a draw from the joint distribution, so it keeps the
        clustering that the coupling induces. The marginals describe each
        edge on its own.

        On a cold run the chain starts from the priors and discards burn_in
        sweeps. On a warm run it resumes from the previous configuration,
        which is already at equilibrium, so every sweep is counted.

        Params:
            full_states: The known state of every fully observed edge.
            lower_bounds: A lower bound on the multiplier of every partially
                observed edge. Both are cumulative: passing only the newest
                observation would let earlier ones drift.

        Returns:
            The final configuration, and the per-edge posterior
            distributions.
        """
        fully_observed_edges = full_states
        evidence = {
            edge: {
                s: self.state_log_likelihood(s, lower)
                for s in range(self._n_states)
            }
            for edge, lower in lower_bounds.items()
        }
        if self._last_state is None:

            iters = self.n_iter
            burn_in = self.burn_in
            scenario: Scenario = {}
            for edge in self._edges:
                if edge in fully_observed_edges:
                    scenario[edge] = fully_observed_edges[edge]
                else:
                    log_likelihood = evidence.get(edge)
                    if log_likelihood is None:
                        scenario[edge] = self.sample_from_probs(self.prior_probs[edge])
                    else:
                        scenario[edge] = self.sample_from_log_weights(
                            LogStateWeights(
                                {
                                    s: self._log_prior[edge][s] + log_likelihood[s]
                                    for s in range(self._n_states)
                                    if log_likelihood[s] != NEG_INF
                                }
                            )
                        )
        else:
            iters = self.warm_iter
            burn_in = 0
            scenario = dict(self._last_state)
            for edge, state in fully_observed_edges.items():
                scenario[edge] = state

        state_counts: dict[Edge, dict[State, int]] = {
            e: dict.fromkeys(range(self._n_states), 0) for e in self._edges
        }
        n_samples = 0

        for sweep in range(iters):
            for edge in self._edges:
                if edge in fully_observed_edges:
                    continue

                log_prior = self._log_prior[edge]
                log_likelihood = evidence.get(edge)
                deviation = self._deviation(edge, scenario)
                candidates = (
                    [c for c in range(self._n_states) if log_likelihood[c] != NEG_INF]
                    if log_likelihood
                    else range(self._n_states)
                )

                weights = LogStateWeights(
                    {
                        candidate: (log_prior[candidate] + candidate * deviation)
                        / self.temperature
                        + (log_likelihood[candidate] if log_likelihood else 0.0)
                        for candidate in candidates
                    }
                )

                scenario[edge] = self.sample_from_log_weights(weights)

            if sweep >= burn_in:
                for edge in self._edges:
                    state_counts[edge][scenario[edge]] += 1
                n_samples += 1

        beliefs: Beliefs = {}

        for edge in self._edges:
            if edge in fully_observed_edges:
                beliefs[edge] = StateDistribution(
                    {
                        s: 1.0 if s == fully_observed_edges[edge] else 0.0
                        for s in range(self._n_states)
                    }
                )
            elif n_samples > 0:
                beliefs[edge] = StateDistribution(
                    {
                        s: state_counts[edge][s] / n_samples
                        for s in range(self._n_states)
                    }
                )
            else:
                beliefs[edge] = StateDistribution(dict(self.prior_probs[edge]))

        self._last_state = dict(scenario)
        return scenario, beliefs

    def _cold_run(
        self, full_states: dict[Edge, State], lower_bounds: dict[Edge, float]
    ) -> tuple[Scenario, Beliefs]:
        """Run a cold chain that leaves the update chain untouched.

        Only update_states may create or keep a chain. Otherwise a run made
        before any observation would leave the chain warm, and the first
        real update would get only warm_iter sweeps.

        Params:
            full_states: The known state of every fully observed edge.
            lower_bounds: A lower bound on the multiplier of every partially
                observed edge.

        Returns:
            The final configuration, and the per-edge marginals.
        """
        saved = self._last_state
        self._last_state = None
        try:
            return self._run_gibbs(full_states, lower_bounds)
        finally:
            self._last_state = saved

    @override
    def generate_states(self) -> Scenario:
        """Draw a fresh scenario from a cold chain with no evidence."""
        scenario, _ = self._cold_run({}, {})
        return scenario

    @override
    def prior_run(self) -> tuple[Scenario, Beliefs]:
        return self._cold_run({}, {})

    @override
    def update_states(
        self, full_states: dict[Edge, State], lower_bounds: dict[Edge, float]
    ) -> tuple[Scenario, Beliefs]:
        return self._run_gibbs(full_states, lower_bounds)

    @override
    def sample_states(
        self, full_states: dict[Edge, State], lower_bounds: dict[Edge, float]
    ) -> Scenario:
        """Continue the update chain, or run cold if there is none yet."""
        if self._last_state is None:
            scenario, _ = self._cold_run(full_states, lower_bounds)
        else:
            scenario, _ = self._run_gibbs(full_states, lower_bounds)
        return scenario
