"""Market data feed using free APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterable

import httpx

from ..base import DataFeed, FeedItem, FeedType
from ...environment import get_polygon_api_key


class MarketDataFeed:
    """Market data feed for quotes and news."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_polygon_api_key()
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def name(self) -> str:
        return "market"

    @property
    def feed_type(self) -> FeedType:
        return FeedType.MARKET

    async def fetch(self, since: datetime | None = None) -> list[FeedItem]:
        """Fetch market news."""
        items: list[FeedItem] = []

        if not self._api_key:
            return items

        try:
            # Polygon.io news endpoint
            url = "https://api.polygon.io/v2/reference/news"
            params = {
                "apiKey": self._api_key,
                "limit": 20,
                "order": "desc",
            }

            response = await self._client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for article in data.get("results", []):
                    timestamp = datetime.now()
                    if article.get("published_utc"):
                        try:
                            timestamp = datetime.fromisoformat(
                                article["published_utc"].replace("Z", "+00:00")
                            )
                        except Exception:
                            pass

                    if since and timestamp < since:
                        continue

                    items.append(FeedItem(
                        id=article.get("id", "")[:20],
                        feed_type=FeedType.MARKET,
                        source=article.get("publisher", {}).get("name", "Unknown"),
                        title=article.get("title", ""),
                        content=article.get("description"),
                        url=article.get("article_url"),
                        timestamp=timestamp,
                        metadata={
                            "tickers": article.get("tickers", []),
                            "keywords": article.get("keywords", []),
                        },
                        priority=1 if article.get("tickers") else 0,
                    ))
        except Exception:
            pass

        return items

    async def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Get latest quote for a symbol."""
        if not self._api_key:
            return None

        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
            params = {"apiKey": self._api_key}

            response = await self._client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    r = results[0]
                    return {
                        "symbol": symbol,
                        "open": r.get("o"),
                        "high": r.get("h"),
                        "low": r.get("l"),
                        "close": r.get("c"),
                        "volume": r.get("v"),
                        "vwap": r.get("vw"),
                    }
        except Exception:
            pass

        return None

    async def stream(self) -> AsyncIterable[FeedItem]:
        """Stream market news."""
        items = await self.fetch()
        for item in items:
            yield item

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
