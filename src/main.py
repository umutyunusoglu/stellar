import pprint
import random
from collections import Counter
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

from disaster_model import MRFModel
from typedefs import (
    DistanceMatrix,
    Edge,
    Node,
    NodeGraph,
    Scenario,
    State,
    StateDistribution,
)

STATE_COLORS = ("#008000", "#f4a300", "#c1272d")
N_STATES = len(STATE_COLORS)
MUTED_COLOR = "#d9d9d9"


GRID_SIDE = 40
TEMPERATURE = 2.0
BETA = np.array([0.8, 0.2])
SEED = 0


def random_priors(
    edges: list[Edge], rng: np.random.Generator
) -> dict[Edge, StateDistribution]:
    """Draw a uniform random prior distribution for every edge.

    Params:
        edges: The edges to draw priors for.
        rng: Source of randomness.

    Returns:
        Per-edge prior distribution over damage states.
    """
    concentration: npt.NDArray[np.float64] = np.ones(N_STATES)
    return {
        edge: StateDistribution(
            {s: float(p) for s, p in zip(range(N_STATES), rng.dirichlet(concentration))}
        )
        for edge in edges
    }


def save_plot(
    graph: NodeGraph,
    pos: dict[Node, tuple[int, int]],
    scenario: Scenario,
    state: State,
    path: str,
) -> None:
    """Draw the network with one damage state highlighted.

    Edges in the given state carry that state's colour, every other edge is
    muted, so the spatial extent of a single state is readable on its own.

    Params:
        graph: The network being drawn.
        pos: The position of every node.
        scenario: The damage state of every edge.
        state: The damage state to highlight.
        path: Where to write the image.
    """
    highlighted = [e for e, s in scenario.items() if s == state]
    muted = [e for e, s in scenario.items() if s != state]

    fig, ax = plt.subplots(figsize=(10, 10))
    ## Muted first so the highlighted edges sit on top at crossings.
    nx.draw_networkx_edges(
        graph, pos, edgelist=muted, edge_color=MUTED_COLOR, width=1, ax=ax
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=highlighted,
        edge_color=STATE_COLORS[state],
        width=2,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ## grid_2d_graph labels nodes by coordinate, which does not match the
    ## integer Node type. The coordinate is kept for drawing.
    graph = cast(
        NodeGraph,
        nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(GRID_SIDE, GRID_SIDE), label_attribute="pos"
        ),
    )
    pos = cast(dict[Node, tuple[int, int]], nx.get_node_attributes(graph, "pos"))

    ## Nodes of the line graph are the edges of the network. Take the edge
    ## list from here rather than graph.edges: the two orderings need not
    ## agree, and the model keys on this one.
    line_graph = nx.line_graph(graph)
    edges = cast(list[Edge], list(line_graph.nodes))

    ## Only distances the coupling can use. All-pairs would be quadratic in
    ## the edge count and discarded almost entirely.
    distances: DistanceMatrix = {
        edge: nx.single_source_shortest_path_length(line_graph, edge, cutoff=len(BETA))
        for edge in edges
    }

    priors = random_priors(edges, np.random.default_rng(SEED))

    model = MRFModel(
        distances,
        priors,
        temperature=TEMPERATURE,
        beta=BETA,
        rng=random.Random(SEED),
    )

    disaster = model.generate_disaster()
    pprint.pprint(Counter(disaster.values()))

    for state in range(N_STATES):
        save_plot(graph, pos, disaster, state, f"disaster_state_{state}.png")


if __name__ == "__main__":
    main()
