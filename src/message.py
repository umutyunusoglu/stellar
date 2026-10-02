import enum
from dataclasses import dataclass
from typing import Final, Literal

from edges import canonical
from typedefs import AgentId, Edge

__all__ = [
    "ALL",
    "FullKnowledge",
    "Heartbeat",
    "Message",
    "MessageTarget",
    "PartialKnowledge",
    "Payload",
]


class _ALL(enum.StrEnum):
    ALL = enum.auto()


ALL: Final = _ALL.ALL

type MessageTarget = Literal[_ALL.ALL] | frozenset[AgentId]


@dataclass(frozen=True, slots=True)
class FullKnowledge:
    """The exact cost multiplier of an edge."""

    edge: Edge
    multiplier: float

    def __post_init__(self) -> None:
        ## Frozen, so bypass __setattr__ to store the canonical orientation.
        object.__setattr__(self, "edge", canonical(*self.edge))


@dataclass(frozen=True, slots=True)
class PartialKnowledge:
    """A lower bound on the cost multiplier of an edge."""

    edge: Edge
    multiplier: float

    def __post_init__(self) -> None:
        ## Frozen, so bypass __setattr__ to store the canonical orientation.
        object.__setattr__(self, "edge", canonical(*self.edge))


@dataclass(frozen=True, slots=True)
class Heartbeat:
    """A liveness sanity check; carries no information for the model."""

    sender: AgentId


type Payload = FullKnowledge | PartialKnowledge | Heartbeat


@dataclass(frozen=True, slots=True)
class Message:

    message_source: AgentId
    message_target: MessageTarget
    payload: Payload
