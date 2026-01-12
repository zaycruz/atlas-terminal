"""Base protocol and types for data feeds."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncIterable, Protocol
from uuid import uuid4


class FeedType(str, Enum):
    """Types of data feeds."""
    NEWS = "news"
    MARKET = "market"
    ECONOMIC = "economic"
    SOCIAL = "social"
    EVENTS = "events"


@dataclass
class FeedItem:
    """Single item from a data feed."""
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    feed_type: FeedType = FeedType.NEWS
    source: str = ""
    title: str = ""
    content: str | None = None
    url: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict | None = None
    priority: int = 0  # Higher = more important

    def to_alert_payload(self) -> dict:
        """Convert to alert event payload."""
        return {
            "id": self.id,
            "type": self.feed_type.value,
            "title": self.title,
            "priority": self.priority,
            "url": self.url,
            "source": self.source,
        }


class DataFeed(Protocol):
    """Protocol for data feed sources."""

    @property
    def name(self) -> str:
        """Feed identifier."""
        ...

    @property
    def feed_type(self) -> FeedType:
        """Type classification."""
        ...

    async def fetch(self, since: datetime | None = None) -> list[FeedItem]:
        """Fetch items, optionally since a timestamp."""
        ...

    async def stream(self) -> AsyncIterable[FeedItem]:
        """Real-time streaming (if supported)."""
        ...
