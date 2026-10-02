from typing import cast

import networkx as nx

from typedefs import Edge, EdgeGraph, Node, NodeGraph


def canonical(u: Node, v: Node) -> Edge:
    """Orient an undirected edge the one way every model and agent keys on.

    Params:
        u: One endpoint.
        v: The other endpoint.

    Returns:
        The edge as (smaller node, larger node).
    """
    return (u, v) if u <= v else (v, u)


def canonical_line_graph(graph: NodeGraph) -> EdgeGraph:
    """Build the line graph with every node in canonical orientation.

    The node order of nx.line_graph is kept, so the edge order derived from
    it is unchanged.

    Params:
        graph: The road network.

    Returns:
        The line graph, whose nodes are canonical edges.
    """
    return cast(
        EdgeGraph,
        nx.relabel_nodes(nx.line_graph(graph), lambda edge: canonical(*edge)),
    )
