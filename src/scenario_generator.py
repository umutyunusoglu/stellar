import enum
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import networkx as nx

from channel import BroadcastChannel, Channel
from disaster_model import DisasterModel
from edges import canonical
from message import ALL, FullKnowledge, Heartbeat, Message, PartialKnowledge
from typedefs import (
    AgentId,
    CostScenario,
    Edge,
    KnowledgeBase,
    Node,
)

## Slack past the expected traversal time before an edge counts as overdue,
## so a bound that only matches the expectation does not retrigger at once.
OVERDUE_SLACK = 1e-6
## Absorbs rounding when comparing positions on an edge.
POS_TOL = 1e-9

type TraversalTime = Callable[[Edge], float]


class RoutingCost(enum.Enum):
    """Which multiplier estimate routing uses for an unobserved edge."""

    ## Posterior mean multiplier of the twin model.
    EXPECTED = enum.auto()
    ## The joint sample (heuristic map) of the latest update.
    SAMPLE = enum.auto()


def _empty_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(fully_observed_edges={}, partially_observed_edges={})


@dataclass
class Agent:
    """
    Agents have their own digital twin model of the disaster.
    Each agent samples the disaster according to its model, routes on its
    estimates, observes the edges it travels and shares what it learns.

    Time is the unit of cost: crossing edge e takes c_e * m_e / speed. Damage
    is uniform along an edge, and the agent knows how long it has spent on
    an edge from each end, not how far it got. It recognises points it has
    visited, so the edge becomes fully explored once the stretches explored
    from its two ends meet.

    Attributes:
    - id: unique agent id
    - twin_model: DisasterModel so called 'digital twin' of the disaster
    - topology: the road network, base costs in graph[u][v]["cost"]
    - source, target: start and goal nodes
    - speed: travel speed
    - routing_cost: which multiplier estimate routing uses
    - heartbeat_interval: time between heartbeats; None disables them
    - knowledge_base: every exact multiplier and lower bound known so far
    - heuristic_map: one joint cost-multiplier sample from the twin model,
      consistent with the knowledge base
    """

    _id: AgentId
    _twin_model: DisasterModel
    _topology: nx.Graph[Node]
    _source: Node
    _target: Node
    _speed: float
    _routing_cost: RoutingCost = RoutingCost.EXPECTED
    _heartbeat_interval: float | None = 1.0
    _knowledge_base: KnowledgeBase = field(default_factory=_empty_knowledge_base)
    _heuristic_map: CostScenario = field(init=False)

    ## Position: at _node, or on the edge (_entry, _far) heading to _heading,
    ## _pos time units from _entry.
    _node: Node | None = field(init=False, default=None)
    _entry: Node = field(init=False)
    _far: Node = field(init=False)
    _heading: Node = field(init=False)
    _pos: float = field(init=False, default=0.0)
    ## Nodes still to visit after the node the agent heads to.
    _route: list[Node] = field(init=False, default_factory=list)

    ## Furthest time-distance reached into each edge from each of its ends.
    _reach: dict[Edge, dict[Node, float]] = field(init=False, default_factory=dict)
    ## Edges this agent has fully explored itself.
    _explored: set[Edge] = field(init=False, default_factory=set)
    _new_info: bool = field(init=False, default=False)
    _outbox: list[Message] = field(init=False, default_factory=list)
    _next_heartbeat: float = field(init=False, default=0.0)
    _last_seen: dict[AgentId, float] = field(init=False, default_factory=dict)

    arrival_time: float | None = field(init=False, default=None)
    n_replans: int = field(init=False, default=0)
    n_turnarounds: int = field(init=False, default=0)
    n_observations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        ## Not update(): a run before any observation must not warm the
        ## twin's chain, or the first real update would be a short one.
        self._heuristic_map = self._twin_model.sample()

    @property
    def id(self) -> AgentId:
        return self._id

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._knowledge_base

    @property
    def has_completed_goal(self) -> bool:
        return self.arrival_time is not None

    def setup(self) -> None:
        """Place the agent at its source and plan the initial route.

        The route uses the twin model's prior estimates; no update is run.
        """
        self._node = self._source
        if self._source == self._target:
            self.arrival_time = 0.0
            return
        _, path = self._shortest_path(self._source)
        self._route = path[1:]

    def ingest(self, messages: Iterable[Message], time: float) -> None:
        """Merge received messages into the knowledge base.

        Replans once if any of them carried new information.

        Params:
            messages: The messages delivered to this agent.
            time: The simulation time of delivery.
        """
        kb = self._knowledge_base
        for message in messages:
            match message.payload:
                case FullKnowledge(edge, multiplier):
                    known = kb.fully_observed_edges.get(edge)
                    ## Agents measuring the same edge differ only by rounding.
                    if known is None or not math.isclose(known, multiplier):
                        kb.fully_observed_edges[edge] = multiplier
                        kb.partially_observed_edges.pop(edge, None)
                        self._new_info = True
                case PartialKnowledge(edge, multiplier):
                    ## Bounds from different agents may cover the same
                    ## stretch, so only the larger one is safe to keep.
                    if (
                        edge not in kb.fully_observed_edges
                        and multiplier > kb.partially_observed_edges.get(edge, 1.0)
                    ):
                        kb.partially_observed_edges[edge] = multiplier
                        self._new_info = True
                case Heartbeat(sender):
                    self._last_seen[sender] = time
        if self._new_info and not self.has_completed_goal:
            self._replan()

    def step(self, dt: float, time: float, traversal_time: TraversalTime) -> None:
        """Move for one tick, observing and replanning on the way.

        Leftover time after reaching a node carries onto the next edge.

        Params:
            dt: Length of the tick.
            time: Simulation time at the start of the tick.
            traversal_time: True time this agent needs to cross an edge. Used
                only to tell when the agent reaches a node or a point it has
                visited before.
        """
        remaining = dt
        while remaining > 0 and not self.has_completed_goal:
            if self._node is not None:
                self._depart()
            remaining = self._advance(remaining, traversal_time)
            clock = time + dt - remaining
            if self._node is not None:
                self._arrive(self._node, clock)
            elif self._new_info:
                self._replan()

    def outbox(self, time: float) -> list[Message]:
        """Collect the messages to send this tick.

        Params:
            time: The simulation time at the end of the tick.

        Returns:
            New observations, plus a heartbeat when one is due.
        """
        messages, self._outbox = self._outbox, []
        if (
            self._heartbeat_interval is not None
            and not self.has_completed_goal
            and time >= self._next_heartbeat
        ):
            messages.append(Message(self._id, ALL, Heartbeat(self._id)))
            self._next_heartbeat = time + self._heartbeat_interval
        return messages

    def _advance(self, remaining: float, traversal_time: TraversalTime) -> float:
        """Move along the current edge until the tick ends or something happens.

        Events are reaching an end, the explored stretches meeting, and the
        edge becoming overdue. Sets _node when an end is reached.

        Params:
            remaining: Time left in the tick.
            traversal_time: True crossing time of an edge for this agent.

        Returns:
            Time left in the tick afterwards.
        """
        edge = canonical(self._entry, self._far)
        reach = self._reach.setdefault(edge, {})

        if self._heading == self._entry:
            moved = min(remaining, self._pos)
            self._pos -= moved
            if self._pos <= POS_TOL:
                self._node = self._entry
            return remaining - moved

        kb = self._knowledge_base
        true_time = traversal_time(edge)
        other = reach.get(self._far, 0.0)
        ## Reaching the far end is the stretches meeting with nothing explored
        ## from that side, so one event covers both.
        to_explored = (
            true_time - other - self._pos if edge not in self._explored else math.inf
        )
        to_overdue = (
            self._overdue_at(edge, other) - self._pos
            if edge not in kb.fully_observed_edges
            else math.inf
        )
        to_end = true_time - self._pos
        moved = max(min(remaining, to_explored, to_overdue, to_end), 0.0)

        self._pos += moved
        if to_explored <= moved + POS_TOL:
            self._pos = true_time - other
        reach[self._entry] = max(reach.get(self._entry, 0.0), self._pos)

        ## An earlier entry from this end may have gone further than now.
        explored = reach[self._entry] + other
        if to_explored <= moved + POS_TOL:
            self._observe_exact(edge, explored)
        elif to_overdue <= moved + POS_TOL:
            self._observe_bound(edge, explored)
        if self._pos >= true_time - POS_TOL:
            self._node = self._far
        return remaining - moved

    def _overdue_at(self, edge: Edge, other: float) -> float:
        """Position on the current edge at which it becomes overdue.

        Past the expected crossing time, and past the last bound recorded, so
        a bound that brings no news never fires again.

        Params:
            edge: The current edge.
            other: Time-distance explored from the far end.

        Returns:
            The position, measured from the entry end.
        """
        known = self._knowledge_base.partially_observed_edges.get(edge, 0.0)
        bound_pos = known * self._base(edge) / self._speed - other
        return max(self._estimate(edge), bound_pos) + OVERDUE_SLACK

    def _observe_exact(self, edge: Edge, explored: float) -> None:
        """Record an edge explored end to end.

        Params:
            edge: The edge.
            explored: Total time-distance explored, its full crossing time.
        """
        self._explored.add(edge)
        kb = self._knowledge_base
        if edge in kb.fully_observed_edges:
            return
        multiplier = explored * self._speed / self._base(edge)
        kb.fully_observed_edges[edge] = multiplier
        kb.partially_observed_edges.pop(edge, None)
        self._outbox.append(Message(self._id, ALL, FullKnowledge(edge, multiplier)))
        self._new_info = True
        self.n_observations += 1

    def _observe_bound(self, edge: Edge, explored: float) -> None:
        """Record a lower bound from an edge that is taking longer than expected.

        The explored stretches from the two ends are disjoint pieces of a
        uniformly damaged edge, so their times add up.

        Params:
            edge: The edge.
            explored: Total time-distance explored from both ends.
        """
        kb = self._knowledge_base
        bound = explored * self._speed / self._base(edge)
        if bound <= kb.partially_observed_edges.get(edge, 1.0):
            return
        kb.partially_observed_edges[edge] = bound
        self._outbox.append(Message(self._id, ALL, PartialKnowledge(edge, bound)))
        self._new_info = True
        self.n_observations += 1

    def _arrive(self, node: Node, clock: float) -> None:
        """Handle reaching a node.

        Params:
            node: The node reached.
            clock: The simulation time of arrival.
        """
        self._node = node
        self._pos = 0.0
        if node == self._target:
            self.arrival_time = clock
            return
        if self._new_info:
            self._replan()

    def _depart(self) -> None:
        """Leave the current node along the route."""
        assert self._node is not None
        self._entry = self._node
        self._far = self._heading = self._route.pop(0)
        self._pos = 0.0
        self._node = None

    def _replan(self) -> None:
        """Update the twin model with all knowledge and choose a new route.

        On an edge, going back to the entry end costs exactly the time spent
        getting here, while going on costs the estimated rest of the edge.
        Ties keep the current heading.
        """
        self._new_info = False
        self.n_replans += 1
        self._heuristic_map, _ = self._twin_model.update(self._knowledge_base)

        if self._node is not None:
            _, path = self._shortest_path(self._node)
            self._route = path[1:]
            return

        edge = canonical(self._entry, self._far)
        to_entry, entry_path = self._shortest_path(self._entry)
        to_far, far_path = self._shortest_path(self._far)
        back = self._pos + to_entry
        forward = max(self._estimate(edge) - self._pos, 0.0) + to_far

        heading = self._heading
        if back < forward:
            heading = self._entry
        elif forward < back:
            heading = self._far
        if heading != self._heading:
            self.n_turnarounds += 1
            self._heading = heading
        self._route = (entry_path if heading == self._entry else far_path)[1:]

    def _shortest_path(self, source: Node) -> tuple[float, list[Node]]:
        """Cheapest route to the target under the current estimates.

        Params:
            source: Where the route starts.

        Returns:
            Its estimated time and its nodes, source first.
        """
        return nx.single_source_dijkstra(
            self._topology,
            source,
            self._target,
            weight=lambda u, v, _: self._estimate(canonical(u, v)),
        )

    def _estimate(self, edge: Edge) -> float:
        """Estimated time to cross an edge.

        Params:
            edge: A canonical edge.

        Returns:
            Base cost times the multiplier estimate, over speed.
        """
        return self._base(edge) * self._multiplier(edge) / self._speed

    def _multiplier(self, edge: Edge) -> float:
        """Multiplier estimate of an edge; exact once known."""
        known = self._knowledge_base.fully_observed_edges.get(edge)
        if known is not None:
            return known
        if self._routing_cost is RoutingCost.SAMPLE:
            return self._heuristic_map[edge]
        return self._twin_model.expected_multiplier(edge)

    def _base(self, edge: Edge) -> float:
        u, v = edge
        return self._topology[u][v]["cost"]


@dataclass(frozen=True)
class AgentResult:
    agent_id: AgentId
    arrival_time: float | None
    n_replans: int
    n_turnarounds: int
    n_observations: int


@dataclass(frozen=True)
class RunResult:
    """
    Outcome of one simulation.

    attributes:
        agents: Per-agent outcome; arrival_time is None if it never arrived.
        end_time: Simulation time when the run stopped.
    """

    agents: list[AgentResult]
    end_time: float

    @property
    def all_arrived(self) -> bool:
        return all(a.arrival_time is not None for a in self.agents)

    @property
    def _arrivals(self) -> list[float]:
        return [a.arrival_time for a in self.agents if a.arrival_time is not None]

    @property
    def makespan(self) -> float:
        """Last arrival time; infinite if an agent never arrived."""
        return max(self._arrivals, default=0.0) if self.all_arrived else math.inf

    @property
    def total_time(self) -> float:
        """Sum of arrival times; infinite if an agent never arrived."""
        return sum(self._arrivals) if self.all_arrived else math.inf

    @property
    def mean_arrival(self) -> float:
        """Mean arrival time; infinite if an agent never arrived."""
        return self.total_time / len(self.agents) if self.agents else 0.0


@dataclass
class Stellar:
    """
    Simulates the agents on one realised disaster with a fixed-tick clock.

    Each tick: deliver last tick's messages, let agents merge them (and
    replan once if needed), move every unfinished agent, then send what they
    emitted.

    Attributes:
        - disaster_model: DisasterModel the true disaster is drawn from
        - agents: list[Agent]
        - topology: nx.Graph with base costs in graph[u][v]["cost"]
        - channel: how messages travel between agents
        - delta_time: length of one tick
        - max_time: the run stops here even if agents are still travelling
    """

    _disaster_model: DisasterModel
    _agents: list[Agent]
    _topology: nx.Graph[Node]
    _channel: Channel = field(default_factory=BroadcastChannel)
    _delta_time: float = 0.1
    _max_time: float = 10_000.0
    _realised_disaster: CostScenario = field(init=False)

    def __post_init__(self) -> None:
        self._realised_disaster = self._disaster_model.generate_disaster()

    @property
    def realised_disaster(self) -> CostScenario:
        return self._realised_disaster

    def traversal_time(self, agent: Agent) -> TraversalTime:
        """True edge crossing times for one agent.

        Params:
            agent: The agent travelling.

        Returns:
            Edge -> base cost times realised multiplier, over its speed.
        """

        def time(edge: Edge) -> float:
            u, v = edge
            return (
                self._topology[u][v]["cost"]
                * self._realised_disaster[edge]
                / agent.speed
            )

        return time

    def run(self) -> RunResult:
        """Run until every agent arrives or max_time is reached.

        Returns:
            The per-agent outcome and the stopping time.
        """
        ids = [a.id for a in self._agents]
        times = {a.id: self.traversal_time(a) for a in self._agents}
        for agent in self._agents:
            agent.setup()

        tick = 0
        time = 0.0
        while not self._has_all_agents_completed_goal() and time < self._max_time:
            inbox = self._channel.deliver(time, ids)
            for agent in self._agents:
                if not agent.has_completed_goal:
                    agent.ingest(inbox[agent.id], time)
            for agent in self._agents:
                if not agent.has_completed_goal:
                    agent.step(self._delta_time, time, times[agent.id])
            ## Multiplying rather than accumulating keeps the clock exact.
            tick += 1
            time = tick * self._delta_time
            for agent in self._agents:
                for message in agent.outbox(time):
                    self._channel.send(message, time)

        return RunResult(
            agents=[
                AgentResult(
                    agent_id=a.id,
                    arrival_time=a.arrival_time,
                    n_replans=a.n_replans,
                    n_turnarounds=a.n_turnarounds,
                    n_observations=a.n_observations,
                )
                for a in self._agents
            ],
            end_time=time,
        )

    def _has_all_agents_completed_goal(self) -> bool:
        return all(a.has_completed_goal for a in self._agents)
