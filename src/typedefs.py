from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

import networkx as nx

type Node = int
type AgentId = int
type State = int
type HopDistance = int
type Edge = tuple[Node, Node]

StateDistribution = NewType("StateDistribution", dict[State, float])
LogStateWeights = NewType("LogStateWeights", dict[State, float])

type Scenario = dict[Edge, State]
type Beliefs = dict[Edge, StateDistribution]

type CostScenario = dict[Edge, float]
type ExpectedMultipliers = dict[Edge, float]


@dataclass
class Observation:
    observed_cost: float
    is_fully_observed: bool


@dataclass
class KnowledgeBase:
    """
    Everything an agent has observed so far, as cost multipliers.

    attributes:
        fully_observed_edges: The exact cost multiplier of each edge.
        partially_observed_edges: A lower bound on the cost multiplier of
            each edge.
    """

    fully_observed_edges: dict[Edge, float]
    partially_observed_edges: dict[Edge, float]


type DistanceMatrix = Mapping[Edge, Mapping[Edge, HopDistance]]
type EdgeGraph = nx.Graph[Edge]
type NodeGraph = nx.Graph[Node]
