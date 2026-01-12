"""Feed manager for aggregating and scheduling data feeds."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Callable

from .base import FeedItem, FeedType
from .feeds.rss import RSSFeed
from .feeds.market import MarketDataFeed
from .events.fed import FedEventMonitor
from .events.economic import EconomicCalendar
from ..orchestrator.events import EventBus, Event, EventType, get_event_bus


class FeedManager:
    """Manages and aggregates all data feeds."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        refresh_interval: int = 300,  # 5 minutes default
    ) -> None:
        self._bus = event_bus or get_event_bus()
        self._refresh_interval = refresh_interval
        self._running = False
        self._task: asyncio.Task | None = None

        # Initialize feeds
        self._rss = RSSFeed()
        self._market = MarketDataFeed()
        self._fed = FedEventMonitor()
        self._economic = EconomicCalendar()

        self._last_fetch: datetime | None = None
        self._callbacks: list[Callable[[FeedItem], None]] = []

    async def start(self) -> None:
        """Start the feed manager background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the feed manager."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main refresh loop."""
        while self._running:
            try:
                await self.refresh()
            except Exception:
                pass  # Continue on errors

            await asyncio.sleep(self._refresh_interval)

    async def refresh(self) -> list[FeedItem]:
        """Fetch all feeds and publish new items."""
        since = self._last_fetch
        self._last_fetch = datetime.now()

        all_items: list[FeedItem] = []

        # Fetch all feeds concurrently
        results = await asyncio.gather(
            self._rss.fetch(since),
            self._market.fetch(since),
            self._fed.fetch(since),
            self._economic.fetch(since),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, list):
                all_items.extend(result)

        # Sort by priority then timestamp
        all_items.sort(
            key=lambda x: (x.priority, x.timestamp.timestamp()),
            reverse=True,
        )

        # Publish as alerts
        for item in all_items[:20]:  # Limit to 20 new items
            event = Event(
                type=EventType.ALERT,
                payload=item.to_alert_payload(),
            )
            await self._bus.publish(event)

            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    callback(item)
                except Exception:
                    pass

        return all_items

    async def fetch_all(self) -> list[FeedItem]:
        """Fetch all feeds without time filter."""
        all_items: list[FeedItem] = []

        results = await asyncio.gather(
            self._rss.fetch(),
            self._market.fetch(),
            self._fed.fetch(),
            self._economic.fetch(),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, list):
                all_items.extend(result)

        all_items.sort(
            key=lambda x: (x.priority, x.timestamp.timestamp()),
            reverse=True,
        )

        return all_items

    def on_item(self, callback: Callable[[FeedItem], None]) -> None:
        """Register a callback for new feed items."""
        self._callbacks.append(callback)

    async def get_quote(self, symbol: str) -> dict | None:
        """Get a market quote."""
        return await self._market.get_quote(symbol)

    async def close(self) -> None:
        """Close all feed clients."""
        await self.stop()
        await self._market.close()
        await self._fed.close()
        await self._economic.close()


# Global instance
_manager: FeedManager | None = None


def get_feed_manager() -> FeedManager:
    """Get the global feed manager."""
    global _manager
    if _manager is None:
        _manager = FeedManager()
    return _manager
