from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import override

import networkx as nx
import numpy as np

from typedefs import Edge, Node


class GraphGenerationPolicy(ABC):
    def initialize_graph(self) -> nx.Graph[Node]:
        graph = self._build_graph()
        if graph.is_directed():
            raise ValueError(f"{type(self).__name__} produced a directed graph")
        if not nx.is_weighted(graph):
            raise ValueError(f"{type(self).__name__} produced an unweighted graph")
        return graph

    @abstractmethod
    def _build_graph(self) -> nx.Graph[Node]:
        pass


class FromAdjacencyMatrix(GraphGenerationPolicy):
    def __init__(self, adj_matrix: np.ndarray) -> None:
        super().__init__()
        self.adj_matrix: np.ndarray = adj_matrix

    @override
    def _build_graph(
        self,
    ) -> nx.Graph[Node]:
        return nx.from_numpy_array(self.adj_matrix)


class FromEdgeList(GraphGenerationPolicy):
    def __init__(self, edge_list: Iterable[Edge]):
        super().__init__()
        self.edge_list: Iterable[Edge] = edge_list

    @override
    def _build_graph(self) -> nx.Graph[Node]:
        return nx.from_edgelist(self.edge_list)


class FromNetworkxGraph(GraphGenerationPolicy):

    def __init__(self, graph: nx.Graph[Node]) -> None:
        self.graph: nx.Graph[Node] = graph

    @override
    def _build_graph(self) -> nx.Graph[Node]:
        return self.graph
