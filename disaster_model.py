from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field
from typing import override

import numpy as np

from typedefs import Edge


@dataclass()
class DisasterModel(ABC):
    """
    DisasterModel specifies how the disaster is modelled.
    """

    distances: dict[Edge, dict[Edge, int]]
    base_prob: dict[Edge, float]

    @abstractmethod
    def generate_disaster(
        self, raw_edge_probabilites: dict[Edge, float]
    ) -> dict[Edge, float]:
        """
        Params:
            - topology: A networkx graph that the disaster is applied to.

        Returns:
            - dict[Edge,float]: probabilites of edges being affected after the disaster

        """

    @abstractmethod
    def get_updated_beliefs(self, observations: dict[Edge, int]) -> dict[Edge, float]:
        """
        Returns:
        - dict[Edge,float]: updated probabilites of edges being affected
        """


@dataclass()
class MRFModel(DisasterModel):
    """
    Markov Random Field Model For Disaster

    attributes:
    """

    temperature: float
    beta: np.ndarray
    n_iter: int = field(default=1000)

    def __post_init__(self):
        cutoff = len(self.beta)
        distances = {}

        for u, d in self.distances.items():

            dist_of_u = {}
            for v, dist in d.items():
                if u == v:
                    continue
                if dist <= cutoff:
                    dist_of_u[v] = dist

            distances[u] = dist_of_u

        self.distances: dict[Edge, dict[Edge, float]] = distances

    def _field(self, edge: Edge) -> float:
        p = self.base_prob[edge]
        p = max(1e-5, min(p, 1 - 1e-5))

        return 0.5 * math.log(p / (1 - p))

    @override
    def generate_disaster(
        self, raw_edge_probabilites: dict[Edge, float]
    ) -> dict[Edge, float]:

        return super().generate_disaster(raw_edge_probabilites)

    @override
    def get_updated_beliefs(self, observations: dict[Edge, int]) -> dict[Edge, float]:
        return super().get_updated_beliefs(observations)
