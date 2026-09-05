from __future__ import annotations

from dataclasses import dataclass, field
from mimetypes import init

import networkx as nx

from disaster_model import DisasterModel
from typedefs import Edge, Node, State, StateDistribution


@dataclass
class Agent:
    """
    Agents have their own digital twin model of the disaster.
    Each agent sample the disaster according to their model.
    Agents update their knowledge base according as they explore the graph.

    Attributes:
    - twin_model: DisasterModel so called 'digital twin' of the disaster
    - knowledge_base: a dictionary that holds the knowledge of expored edges
    - heuristic_map: a sampled heuristic using disaster model with the same topology of the original graph

    """

    _twin_model: DisasterModel
    _topology: nx.Graph[Node]
    _source: Node
    _target: Node
    _knowledge_base: dict[Edge, State] = field(init=False, default={})
    _heuristic_map: dict[Edge, State] = field(init=False)
    _beliefs: dict[Edge, StateDistribution] = field(init=False)
    _path_history: list[Edge] = field(init=False, default=[])
    _prev_node: Node = field(
        init=False,
    )
    _next_node: Node = field(init=False)
    _offset: float = field(init=False, default=0.0)

    def __post_init__(self):
        self._heuristic_map = self._twin_model.generate_disaster()
        self._beliefs = self._twin_model.prior_probs
        self._prev_node = self._source

    def update_knowledge_base(self, edge: Edge, state: State):
        self._knowledge_base[edge] = state
        self._update_heuristic_map()

    def _update_heuristic_map(self):
        self._heuristic_map, self._beliefs = self._twin_model.update(
            self._knowledge_base
        )

    def _generate_route():
        pass

    @property
    def has_completed_goal(self) -> bool:
        # TODO: This function might be problematic
        return self._prev_node == self._target


@dataclass
class Stelllar:
    """
    Scenario specifies a graph perturbance problem.

    Attributes:
        - disaster_model: DisasterModel
        - agents: list[Agent]
        - graph: nx.Graph
    """

    _disaster_model: DisasterModel
    _agents: list[Agent]
    _topology: nx.Graph[Node]
    _realised_disaster: dict[Edge, State] = field(init=False)

    def __post_init__(self) -> None:
        self._realised_disaster = self._disaster_model.generate_disaster()

    def _has_all_agents_completed_goal(self):
        for a in self._agents:
            if not a.has_completed_goal:
                return False
        return True

    def run_stellar(self):

        while not self._has_all_agents_completed_goal():
            continue
