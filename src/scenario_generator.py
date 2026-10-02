from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Callable

import networkx as nx

from disaster_model import DisasterModel
from edges import canonical
from message import ALL, Message
from typedefs import (
    AgentId,
    CostScenario,
    Edge,
    KnowledgeBase,
    Node,
)


@dataclass
class Agent:
    """
    Agents have their own digital twin model of the disaster.
    Each agent sample the disaster according to their model.
    Agents update their knowledge base according as they explore the graph.

    Attributes:
    - twin_model: DisasterModel so called 'digital twin' of the disaster
    - knowledge_base: a dictionary that holds the knowledge of expored edges
    - heuristic_map: one joint cost-multiplier sample from the twin model,
      consistent with the knowledge base

    """

    _id: AgentId
    _twin_model: DisasterModel
    _topology: nx.Graph[Node]
    _source: Node
    _target: Node
    _speed: float
    _knowledge_base: KnowledgeBase
    _heuristic_map: CostScenario = field(init=False)
    _path_history: list[Edge] = field(init=False, default_factory=list)
    _current_route: list[Node] = field(init=False)
    _prev_node: Node = field(
        init=False,
    )
    _next_node: Node = field(init=False)
    _offset: float = field(init=False, default=0.0)

    def __post_init__(self):
        ## Not update(): a run before any observation must not warm the
        ## twin's chain, or the first real update would be a short one.
        self._heuristic_map = self._twin_model.sample()
        self._prev_node = self._source

    # TODO: These two functions are fucked up
    # find a way to merge them with message consuming
    # add knowledge on partially observable edges
    def update_knowledge_base(self, edge: Edge, multiplier: float):
        self._knowledge_base.fully_observed_edges[canonical(*edge)] = multiplier
        self._update_heuristic_map()

    def _update_heuristic_map(self):
        self._heuristic_map, _ = self._twin_model.update(self._knowledge_base)

    def _generate_route(self, initial: bool = False):

        def _cost(u: Node, v: Node, attrb: dict):
            multiplier = self._twin_model.expected_multiplier(canonical(u, v))
            return self._topology[u][v]["cost"] * multiplier

        def _path_cost(path, cost_fn: Callable):
            return sum(
                cost_fn(u, v, self._topology[u][v]) for u, v in itertools.pairwise(path)
            )

        if initial:
            self._current_route = nx.dijkstra_path(
                G=self._topology, source=self._source, target=self._target, weight=_cost
            )
            return
        """
        At any time point agent has two options:
            - create a route from self._prev_node
            - create a route from self._next_node

            in any option agent must pay the price to reach to designated starting point
        """

        route_forward = nx.dijkstra_path(
            self._topology, source=self._next_node, target=self._target, weight=_cost
        )

        forward_cost_belief = (
            _cost(self._prev_node, self._next_node, {})
            - self._offset
            + _path_cost(route_forward, _cost)
        )

        route_backward = nx.dijkstra_path(
            self._topology, source=self._prev_node, target=self._target, weight=_cost
        )
        backward_cost_belief = self._offset + _path_cost(route_backward, _cost)
        ##TODO: Add a logic the for swapping the head of the agent.
        if forward_cost_belief <= backward_cost_belief:
            self._current_route = route_forward
        else:
            tmp = self._prev_node

    def setup(self):
        """
        Sets up the inital route.
        Ensures the agent starts from the source.
        """

        self._generate_route(initial=True)
        self._prev_node = self._source
        self._next_node = self._current_route[1]
        self._offset = 0

    def consume_messages(self, messages: list[Message]):

        for message in messages:
            if message.message_target != ALL and self._id not in message.message_target:
                continue

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
    _realised_disaster: CostScenario = field(init=False)

    def __post_init__(self) -> None:
        self._realised_disaster = self._disaster_model.generate_disaster()

    def _has_all_agents_completed_goal(self):
        return all(a.has_completed_goal for a in self._agents)

    def run_stellar(self):

        ## A simulation clock with delta_time seconds for each loop iteration
        ## 1 Let all agents create its initial route

        DELTA_TIME = 0.1

        while not self._has_all_agents_completed_goal():
            continue
