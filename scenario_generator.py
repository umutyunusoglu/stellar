from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import networkx as nx

from disaster_model import DisasterModel
from graph_generation_policy import GraphGenerationPolicy
from problem_type import ProblemType
from typedefs import Edge, Node


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

    twin_model: DisasterModel
    topology: nx.Graph[Node]
    knowledge_base: dict[Edge, float]
    heuristic_graph: nx.Graph[Node] = field(init=False)

    def __post_init__(self):
        self.heuristic_graph = self.twin_model.generate_disaster(self.topology)

    def update_knowledge_base(self, key: Edge, value: float):
        self.knowledge_base[key] = value


@dataclass
class Scenario:
    """
    Scenario specifies a graph perturbance problem.

    Attributes:
        - problem_type: ProblemType
        - disaster_model: DisasterModel
        - agents: list[Agent]
        - graph: nx.Graph
    """

    problem_type: ProblemType
    disaster_model: DisasterModel
    agents: list[Agent]
    topology: nx.Graph[Node]
    disaster_graph: nx.Graph[Node]

    def __post_init__(self) -> None:
        self.disaster_graph = self.disaster_model.generate_disaster(self.topology)
