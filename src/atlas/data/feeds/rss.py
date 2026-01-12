"""RSS/Atom feed client."""

from __future__ import annotations

from datetime import datetime
from typing import AsyncIterable

import feedparser

from ..base import DataFeed, FeedItem, FeedType


# Default financial news RSS feeds
DEFAULT_RSS_FEEDS = [
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("Bloomberg Markets", "https://feeds.bloomberg.com/markets/news.rss"),
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
]


class RSSFeed:
    """RSS/Atom feed client for financial news."""

    def __init__(
        self,
        feeds: list[tuple[str, str]] | None = None,
    ) -> None:
        self._feeds = feeds or DEFAULT_RSS_FEEDS
        self._seen_ids: set[str] = set()

    @property
    def name(self) -> str:
        return "rss"

    @property
    def feed_type(self) -> FeedType:
        return FeedType.NEWS

    async def fetch(self, since: datetime | None = None) -> list[FeedItem]:
        """Fetch items from all configured feeds."""
        items: list[FeedItem] = []

        for source_name, url in self._feeds:
            try:
                feed_items = await self._fetch_feed(source_name, url, since)
                items.extend(feed_items)
            except Exception:
                continue  # Skip failed feeds

        # Sort by timestamp, newest first
        items.sort(key=lambda x: x.timestamp, reverse=True)

        return items

    async def _fetch_feed(
        self,
        source_name: str,
        url: str,
        since: datetime | None,
    ) -> list[FeedItem]:
        """Fetch a single RSS feed."""
        # feedparser is synchronous, but fast enough
        parsed = feedparser.parse(url)
        items: list[FeedItem] = []

        for entry in parsed.entries[:20]:  # Limit per feed
            # Parse timestamp
            timestamp = datetime.now()
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    timestamp = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass

            # Filter by since
            if since and timestamp < since:
                continue

            # Create unique ID
            item_id = entry.get("id") or entry.get("link") or entry.get("title", "")[:50]

            # Skip duplicates
            if item_id in self._seen_ids:
                continue
            self._seen_ids.add(item_id)

            items.append(FeedItem(
                id=item_id[:20],
                feed_type=FeedType.NEWS,
                source=source_name,
                title=entry.get("title", "No title"),
                content=entry.get("summary"),
                url=entry.get("link"),
                timestamp=timestamp,
                metadata={"feed_url": url},
            ))

        return items

    async def stream(self) -> AsyncIterable[FeedItem]:
        """Stream is not supported for RSS; use periodic fetch instead."""
        items = await self.fetch()
        for item in items:
            yield item

    def add_feed(self, name: str, url: str) -> None:
        """Add a new RSS feed."""
        self._feeds.append((name, url))

    def remove_feed(self, name: str) -> bool:
        """Remove a feed by name."""
        for i, (n, _) in enumerate(self._feeds):
            if n == name:
                self._feeds.pop(i)
                return True
        return False
