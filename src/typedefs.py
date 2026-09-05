from collections.abc import Mapping
from typing import NewType

import networkx as nx

type Node = int
type AgentId = int
type State = int
type HopDistance = int

## Directed: (u, v) and (v, u) are distinct edges.
type Edge = tuple[Node, Node]

## The one distinction worth enforcing: the two samplers take different
## spaces and neither raises if given the other.
StateDistribution = NewType("StateDistribution", dict[State, float])
LogStateWeights = NewType("LogStateWeights", dict[State, float])

## Total: holds an entry for every edge.
type Scenario = dict[Edge, State]
## Partial and read-only: holds only the edges observed so far.
type Observations = Mapping[Edge, State]
type Beliefs = dict[Edge, StateDistribution]
type DistanceMatrix = Mapping[Edge, Mapping[Edge, HopDistance]]

type EdgeGraph = nx.Graph[Edge]
type NodeGraph = nx.Graph[Node]
