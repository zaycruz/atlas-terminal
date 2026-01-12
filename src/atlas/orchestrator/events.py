"""Event bus for pub/sub communication between components."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List
from uuid import uuid4
import time


class EventType(str, Enum):
    """Types of events that can be published."""
    AGENT_UPDATE = "agent_update"
    DATA_UPDATE = "data_update"
    ALERT = "alert"
    JOB_PROGRESS = "job_progress"
    LLM_STREAM = "llm_stream"
    USER_MESSAGE = "user_message"
    SYSTEM_MESSAGE = "system_message"


@dataclass
class Event:
    """An event that can be published to subscribers."""
    type: EventType
    payload: Any
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid4().hex[:8])


EventCallback = Callable[[Event], None]
AsyncEventCallback = Callable[[Event], Any]


class EventBus:
    """Simple pub/sub event bus for component communication."""

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, Dict[str, AsyncEventCallback]] = {
            event_type: {} for event_type in EventType
        }
        self._global_subscribers: Dict[str, AsyncEventCallback] = {}

    def subscribe(
        self,
        event_type: EventType,
        callback: AsyncEventCallback,
    ) -> str:
        """Subscribe to a specific event type.

        Returns a subscription ID that can be used to unsubscribe.
        """
        sub_id = uuid4().hex[:12]
        self._subscribers[event_type][sub_id] = callback
        return sub_id

    def subscribe_all(self, callback: AsyncEventCallback) -> str:
        """Subscribe to all event types.

        Returns a subscription ID that can be used to unsubscribe.
        """
        sub_id = uuid4().hex[:12]
        self._global_subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by ID.

        Returns True if the subscription was found and removed.
        """
        # Check global subscribers
        if subscription_id in self._global_subscribers:
            del self._global_subscribers[subscription_id]
            return True

        # Check type-specific subscribers
        for subscribers in self._subscribers.values():
            if subscription_id in subscribers:
                del subscribers[subscription_id]
                return True

        return False

    async def publish(self, event: Event) -> None:
        """Publish an event to all relevant subscribers."""
        callbacks: List[AsyncEventCallback] = []

        # Get type-specific subscribers
        callbacks.extend(self._subscribers[event.type].values())

        # Get global subscribers
        callbacks.extend(self._global_subscribers.values())

        # Call all callbacks concurrently
        if callbacks:
            await asyncio.gather(
                *[self._invoke_callback(cb, event) for cb in callbacks],
                return_exceptions=True,
            )

    async def _invoke_callback(
        self,
        callback: AsyncEventCallback,
        event: Event,
    ) -> None:
        """Invoke a callback, handling both sync and async functions."""
        result = callback(event)
        if asyncio.iscoroutine(result):
            await result

    def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously (for use from sync code).

        Creates a new event loop if needed.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
