from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from message import ALL, Message
from typedefs import AgentId


class Channel(Protocol):
    """How messages travel between agents.

    The simulator sends every message an agent emits during a tick and asks
    for deliveries at the start of the next one, so delayed, lossy or
    range-limited channels plug in by deciding what deliver returns.
    """

    def send(self, message: Message, time: float) -> None: ...

    def deliver(
        self, time: float, recipients: Iterable[AgentId]
    ) -> dict[AgentId, list[Message]]: ...


@dataclass
class BroadcastChannel:
    """Lossless channel that delivers everything at the next tick.

    A message reaches every recipient its target names, never its sender.
    """

    _queue: list[Message] = field(init=False, default_factory=list)

    def send(self, message: Message, time: float) -> None:
        """Queue a message for the next delivery.

        Params:
            message: The message to send.
            time: The simulation time it is sent at.
        """
        self._queue.append(message)

    def deliver(
        self, time: float, recipients: Iterable[AgentId]
    ) -> dict[AgentId, list[Message]]:
        """Hand every queued message to its recipients and empty the queue.

        Params:
            time: The simulation time of delivery.
            recipients: The agents that can receive.

        Returns:
            The messages each recipient receives, in sending order.
        """
        inbox: dict[AgentId, list[Message]] = {r: [] for r in recipients}
        for message in self._queue:
            for recipient, messages in inbox.items():
                if recipient == message.message_source:
                    continue
                if message.message_target == ALL or recipient in message.message_target:
                    messages.append(message)
        self._queue.clear()
        return inbox
