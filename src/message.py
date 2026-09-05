import enum
from dataclasses import dataclass
from typing import Final, Literal

from typedefs import AgentId, Edge, State

__all__ = ["ALL", "Heartbeat", "Message", "Payload", "Update"]


class _ALL(enum.StrEnum):
    ALL = enum.auto()


ALL: Final = _ALL.ALL

type MessageTarget = Literal[_ALL.ALL] | frozenset[AgentId]


@dataclass(frozen=True, slots=True)
class Update:
    edge: Edge
    state: State


@dataclass(frozen=True, slots=True)
class Heartbeat:
    sender: AgentId


type Payload = Update | Heartbeat


@dataclass(frozen=True, slots=True)
class Message:

    message_source: AgentId
    message_target: MessageTarget
    payload: Payload
