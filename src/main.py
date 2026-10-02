import pprint
import random
from collections import Counter
from typing import cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import stats

from cost_function import CostFunction
from discrete_model import MRFModel
from edges import canonical_line_graph
from scenario_generator import Agent, Stellar
from typedefs import (
    DistanceMatrix,
    Edge,
    Node,
    NodeGraph,
    Scenario,
    State,
    StateDistribution,
)

STATE_COLORS = ("#008000","#FA5CF7", "#f4a300", "#c1272d",)
N_STATES = len(STATE_COLORS)
MUTED_COLOR = "#d9d9d9"


GRID_SIDE = 40
TEMPERATURE = 2.0
BETA = np.array([0.8, 0.2])
SEED = 0
SIM_GRID_SIDE = GRID_SIDE
N_AGENTS = 3
## How closely a twin's priors follow the true ones: each twin draws its
## per-edge prior from Dirichlet(PRIOR_CONCENTRATION * true prior). Larger
## values mean less variety between agents and less mismatch with the truth.
PRIOR_CONCENTRATION = 5.0


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
            {
                s: float(p)
                for s, p in zip(
                    range(N_STATES), rng.dirichlet(concentration), strict=True
                )
            }
        )
        for edge in edges
    }


def perturb_priors(
    priors: dict[Edge, StateDistribution],
    concentration: float,
    rng: np.random.Generator,
) -> dict[Edge, StateDistribution]:
    """Draw a noisy copy of the priors, centred on them.

    Gives each agent's twin its own beliefs, so agents route differently.

    Params:
        priors: The priors to perturb.
        concentration: Dirichlet concentration; larger stays closer.
        rng: Source of randomness.

    Returns:
        Per-edge prior distribution over damage states.
    """
    perturbed: dict[Edge, StateDistribution] = {}
    for edge, prior in priors.items():
        states = sorted(prior)
        ## Floored, since Dirichlet parameters must be positive.
        alpha = np.maximum([concentration * prior[s] for s in states], 1e-3)
        draw = rng.dirichlet(alpha)
        perturbed[edge] = StateDistribution(
            {s: float(p) for s, p in zip(states, draw, strict=True)}
        )
    return perturbed


def corners(graph: NodeGraph) -> tuple[Node, Node]:
    """Top-left and bottom-right nodes of a grid.

    Params:
        graph: A grid whose nodes carry their coordinate in "pos".

    Returns:
        The node with the smallest coordinate and the one with the largest.
    """
    pos = cast(dict[Node, tuple[int, int]], nx.get_node_attributes(graph, "pos"))
    return min(pos, key=pos.__getitem__), max(pos, key=pos.__getitem__)


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


def build_network(side: int) -> tuple[NodeGraph, list[Edge], DistanceMatrix]:
    """Build a grid road network with base costs.

    Params:
        side: Number of nodes along each side of the grid.

    Returns:
        The network, its canonical edge list in model order, and the
        edge-edge hop distances the coupling can use.
    """
    ## grid_2d_graph labels nodes by coordinate, which does not match the
    ## integer Node type. The coordinate is kept for drawing.
    graph = cast(
        NodeGraph,
        nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(side, side), label_attribute="pos"
        ),
    )

    ## Nodes of the line graph are the edges of the network. Take the edge
    ## list from here rather than graph.edges: the two orderings need not
    ## agree, and the model keys on this one.
    line_graph = canonical_line_graph(graph)
    edges = cast(list[Edge], list(line_graph.nodes))

    ## Only distances the coupling can use. All-pairs would be quadratic in
    ## the edge count and discarded almost entirely.
    distances: DistanceMatrix = {
        edge: nx.single_source_shortest_path_length(line_graph, edge, cutoff=len(BETA))
        for edge in edges
    }

    ## Base costs live on the topology; the model only reasons about
    ## multipliers. A separate seeded Random keeps the model's draws unchanged.
    cost_rng = random.Random(SEED)
    for u, v in edges:
        graph[u][v]["cost"] = cost_rng.uniform(10, 50)
    return graph, edges, distances


def build_model(
    distances: DistanceMatrix,
    priors: dict[Edge, StateDistribution],
    seed: int,
) -> MRFModel:
    """Build an MRF disaster model.

    Twin models are built the same way; giving them other priors or
    parameters than the true model is where model mismatch plugs in.

    Params:
        distances: Edge-edge hop distances.
        priors: Per-edge prior over damage states.
        seed: Seeds both the state sampler and the multiplier draws.

    Returns:
        The model.
    """
    return MRFModel(
        distances=distances,
        prior_probs=priors,
        ## Upper half of a lognormal(0, 1): median 1, so no damage is multiplier 1.
        cost_function=CostFunction(stats.lognorm(s=1)),
        temperature=TEMPERATURE,
        beta=BETA,
        rng=random.Random(seed),
        multiplier_rng=random.Random(seed + 1),
    )


def plot_disaster() -> None:
    """Sample one disaster and plot each damage state."""
    graph, edges, distances = build_network(GRID_SIDE)
    pos = cast(dict[Node, tuple[int, int]], nx.get_node_attributes(graph, "pos"))
    priors = random_priors(edges, np.random.default_rng(SEED))
    model = build_model(distances, priors, SEED)

    ## Plotting is per damage state, so ask the discrete layer for states.
    disaster = model.generate_states()
    pprint.pprint(Counter(disaster.values()))

    for state in range(N_STATES):
        save_plot(graph, pos, disaster, state, f"disaster_state_{state}.png")


def simulate() -> None:
    """Run the agents on one sampled disaster and print the outcome."""
    graph, edges, distances = build_network(SIM_GRID_SIDE)
    priors = random_priors(edges, np.random.default_rng(SEED))
    true_model = build_model(distances, priors, SEED)

    ## Every agent shares the same trip, corner to corner, so what one
    ## discovers is relevant to all of them.
    source, target = corners(graph)
    prior_rng = np.random.default_rng(SEED + 3)
    agents = []
    for agent_id in range(N_AGENTS):
        twin_priors = perturb_priors(priors, PRIOR_CONCENTRATION, prior_rng)
        twin_seed = SEED + 10 * (agent_id + 1)
        agents.append(
            Agent(
                _id=agent_id,
                ## Own priors and seed: agents believe different things about
                ## the disaster, so they explore different routes.
                _twin_model=build_model(distances, twin_priors, twin_seed),
                _topology=graph,
                _source=source,
                _target=target,
                _speed=1.0,
            )
        )

    result = Stellar(_disaster_model=true_model, _agents=agents, _topology=graph).run()
    print(f"{source} -> {target}")
    for outcome in result.agents:
        print(outcome)
    print(
        f"makespan={result.makespan:.2f} total={result.total_time:.2f} "
        f"mean={result.mean_arrival:.2f} end={result.end_time:.2f}"
    )


def main() -> None:
    plot_disaster()
    simulate()


if __name__ == "__main__":
    main()
