"""Federal Reserve event monitor."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterable

import httpx

from ..base import FeedItem, FeedType


# Key Fed events to monitor
FED_EVENTS = [
    "FOMC Meeting",
    "Fed Chair Speech",
    "Fed Minutes",
    "Interest Rate Decision",
    "Fed Balance Sheet",
]


class FedEventMonitor:
    """Monitor Federal Reserve events and announcements."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._seen_ids: set[str] = set()

    @property
    def name(self) -> str:
        return "fed"

    @property
    def feed_type(self) -> FeedType:
        return FeedType.EVENTS

    async def fetch(self, since: datetime | None = None) -> list[FeedItem]:
        """Fetch Fed events from RSS feed."""
        items: list[FeedItem] = []

        try:
            # Federal Reserve RSS feed
            url = "https://www.federalreserve.gov/feeds/press_all.xml"
            response = await self._client.get(url)

            if response.status_code == 200:
                # Simple XML parsing for RSS
                import feedparser
                parsed = feedparser.parse(response.text)

                for entry in parsed.entries[:10]:
                    timestamp = datetime.now()
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            timestamp = datetime(*entry.published_parsed[:6])
                        except Exception:
                            pass

                    if since and timestamp < since:
                        continue

                    item_id = entry.get("id", entry.get("link", ""))[:20]
                    if item_id in self._seen_ids:
                        continue
                    self._seen_ids.add(item_id)

                    # Check if high priority
                    title = entry.get("title", "")
                    priority = 2 if any(evt in title for evt in FED_EVENTS) else 0

                    items.append(FeedItem(
                        id=item_id,
                        feed_type=FeedType.EVENTS,
                        source="Federal Reserve",
                        title=title,
                        content=entry.get("summary"),
                        url=entry.get("link"),
                        timestamp=timestamp,
                        priority=priority,
                        metadata={"category": "fed"},
                    ))
        except Exception:
            pass

        return items

    async def stream(self) -> AsyncIterable[FeedItem]:
        """Stream Fed events."""
        items = await self.fetch()
        for item in items:
            yield item

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
