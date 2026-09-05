import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

import numpy as np
import numpy.typing as npt

from typedefs import (
    Beliefs,
    DistanceMatrix,
    Edge,
    HopDistance,
    LogStateWeights,
    Observations,
    Scenario,
    State,
    StateDistribution,
)


@dataclass(slots=True)
class DisasterModel(ABC):
    """
    DisasterModel specifies how the disaster is modelled.

    attributes:
        distances: Hop distances between every pair of edges.
        prior_probs: Per-edge prior distribution over damage states. Every
            edge must cover the same set of states.
        rng: Source of randomness. Pass a seeded Random for reproducible
            runs.
    """

    distances: DistanceMatrix
    prior_probs: dict[Edge, StateDistribution]
    rng: random.Random = field(default_factory=random.Random, kw_only=True)

    _edges: list[Edge] = field(init=False)
    _n_states: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.prior_probs:
            raise ValueError("prior_probs is empty")

        self._edges = list(self.prior_probs.keys())

        expected = set(self.prior_probs[self._edges[0]])
        for edge, prior in self.prior_probs.items():
            if set(prior) != expected:
                raise ValueError(f"prior for {edge} covers a different state set")

        self._n_states = len(expected)

    @abstractmethod
    def generate_disaster(self) -> Scenario:
        """
        Returns:
            - Scenario: the damage state of every edge in one scenario drawn
              from the model
        """

    @abstractmethod
    def update(self, observations: Observations) -> tuple[Scenario, Beliefs]:
        """
        Params:
            - observations: every edge observed so far, cumulatively

        Returns:
            - Scenario: a scenario consistent with the observations, keeping
              the spatial clustering the model implies
            - Beliefs: per-edge posterior distribution over damage states,
              for reasoning about one edge at a time
        """

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


@dataclass(slots=True)
class MRFModel(DisasterModel):
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
        temperature: Sampling temperature. Lower values sharpen the state
            distribution.
        beta: Coupling strength per hop distance. Entry d - 1 applies to
            neighbours at distance d, so its length sets the cutoff.
        n_iter: Sweeps used for a cold run, starting from the priors.
        burn_in: Sweeps discarded before counting, on a cold run only.
        warm_iter: Sweeps used when resuming from the previous chain state.
    """

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

    def _run_gibbs(self, observations: Observations) -> tuple[Scenario, Beliefs]:
        """Run the Gibbs chain over all edges.

        Observed edges are clamped and never resampled. The final
        configuration is a draw from the joint distribution, so it keeps the
        clustering that the coupling induces. The marginals describe each
        edge on its own.

        On a cold run the chain starts from the priors and discards burn_in
        sweeps. On a warm run it resumes from the previous configuration,
        which is already at equilibrium, so every sweep is counted.

        Params:
            observations: Every edge observed so far, cumulatively. Passing
                only the newest observation would let earlier ones drift.

        Returns:
            The final configuration, and the per-edge posterior
            distributions.
        """
        if self._last_state is None:
            iters = self.n_iter
            burn_in = self.burn_in
            scenario: Scenario = {}
            for edge in self._edges:
                if edge in observations:
                    scenario[edge] = observations[edge]
                else:
                    scenario[edge] = self.sample_from_probs(self.prior_probs[edge])
        else:
            iters = self.warm_iter
            burn_in = 0
            scenario = dict(self._last_state)
            for edge, state in observations.items():
                scenario[edge] = state

        state_counts: dict[Edge, dict[State, int]] = {
            e: {s: 0 for s in range(self._n_states)} for e in self._edges
        }
        n_samples = 0

        for sweep in range(iters):
            for edge in self._edges:
                if edge in observations:
                    continue

                log_prior = self._log_prior[edge]
                deviation = self._deviation(edge, scenario)

                weights = LogStateWeights(
                    {
                        candidate: (log_prior[candidate] + candidate * deviation)
                        / self.temperature
                        for candidate in range(self._n_states)
                    }
                )

                scenario[edge] = self.sample_from_log_weights(weights)

            if sweep >= burn_in:
                for edge in self._edges:
                    state_counts[edge][scenario[edge]] += 1
                n_samples += 1

        beliefs: Beliefs = {}

        for edge in self._edges:
            if edge in observations:
                beliefs[edge] = StateDistribution(
                    {
                        s: 1.0 if s == observations[edge] else 0.0
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

    @override
    def generate_disaster(self) -> Scenario:
        """Draw a fresh scenario, discarding any chain from earlier calls."""
        self._last_state = None
        scenario, _ = self._run_gibbs({})
        self._last_state = None
        return scenario

    @override
    def update(self, observations: Observations) -> tuple[Scenario, Beliefs]:
        return self._run_gibbs(observations)
