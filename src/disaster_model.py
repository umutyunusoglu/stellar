import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typedefs import CostScenario, Edge, ExpectedMultipliers, KnowledgeBase


@dataclass(slots=True)
class DisasterModel(ABC):
    """
    DisasterModel specifies how the disaster is modelled.

    This is the only interface the simulator and the agents depend on. It
    speaks in cost multipliers alone, so models without discrete damage
    states plug in the same way.

    attributes:
        rng: Source of randomness. Pass a seeded Random for reproducible
            runs.
    """

    rng: random.Random = field(default_factory=random.Random, kw_only=True)

    @abstractmethod
    def generate_disaster(self) -> CostScenario:
        """
        Returns:
            - CostScenario: the cost multiplier of every edge in one scenario
              drawn from the model, ignoring any observations
        """

    @abstractmethod
    def update(
        self, observations: KnowledgeBase
    ) -> tuple[CostScenario, ExpectedMultipliers]:
        """
        Params:
            - observations: every edge observed so far, cumulatively

        Returns:
            - CostScenario: a scenario consistent with the observations,
              keeping the spatial clustering the model implies
            - ExpectedMultipliers: per-edge posterior mean multiplier, for
              reasoning about one edge at a time
        """

    @abstractmethod
    def expected_multiplier(self, edge: Edge) -> float:
        """
        Params:
            - edge: the edge whose multiplier is requested

        Returns:
            - float: its posterior mean multiplier after the latest update,
              or its prior mean before any update
        """

    @abstractmethod
    def sample(self) -> CostScenario:
        """
        Returns:
            - CostScenario: a fresh joint draw consistent with the
              observations of the latest update
        """
