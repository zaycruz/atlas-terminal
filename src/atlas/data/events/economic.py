"""Economic calendar and macro event monitor."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterable

import httpx

from ..base import FeedItem, FeedType


# High-impact economic events
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls",
    "CPI",
    "GDP",
    "FOMC",
    "Interest Rate",
    "Unemployment",
    "Retail Sales",
    "PMI",
    "PCE",
    "Consumer Confidence",
]


class EconomicCalendar:
    """Economic calendar and macro event monitor."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        self._seen_ids: set[str] = set()

    @property
    def name(self) -> str:
        return "economic"

    @property
    def feed_type(self) -> FeedType:
        return FeedType.ECONOMIC

    async def fetch(self, since: datetime | None = None) -> list[FeedItem]:
        """Fetch economic calendar events."""
        items: list[FeedItem] = []

        # Try to fetch from investing.com economic calendar RSS
        try:
            url = "https://www.investing.com/rss/economic_calendar.rss"
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0"
            }
            response = await self._client.get(url, headers=headers)

            if response.status_code == 200:
                import feedparser
                parsed = feedparser.parse(response.text)

                for entry in parsed.entries[:15]:
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

                    title = entry.get("title", "")
                    priority = 2 if any(evt.lower() in title.lower() for evt in HIGH_IMPACT_EVENTS) else 0

                    items.append(FeedItem(
                        id=item_id,
                        feed_type=FeedType.ECONOMIC,
                        source="Economic Calendar",
                        title=title,
                        content=entry.get("summary"),
                        url=entry.get("link"),
                        timestamp=timestamp,
                        priority=priority,
                        metadata={"category": "macro"},
                    ))
        except Exception:
            pass

        return items

    async def stream(self) -> AsyncIterable[FeedItem]:
        """Stream economic events."""
        items = await self.fetch()
        for item in items:
            yield item

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()
