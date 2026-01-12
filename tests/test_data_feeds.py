"""Tests for data feed layer."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from atlas.data.base import FeedItem, FeedType
from atlas.data.manager import FeedManager


class TestFeedType:
    """Tests for FeedType enum."""

    def test_feed_type_values(self):
        assert FeedType.NEWS.value == "news"
        assert FeedType.MARKET.value == "market"
        assert FeedType.ECONOMIC.value == "economic"
        assert FeedType.SOCIAL.value == "social"
        assert FeedType.EVENTS.value == "events"


class TestFeedItem:
    """Tests for FeedItem dataclass."""

    def test_creation_with_defaults(self):
        item = FeedItem()
        assert len(item.id) == 8
        assert item.feed_type == FeedType.NEWS
        assert item.source == ""
        assert item.title == ""
        assert item.content is None
        assert item.url is None
        assert isinstance(item.timestamp, datetime)
        assert item.metadata is None
        assert item.priority == 0

    def test_creation_with_values(self):
        ts = datetime(2024, 1, 15, 12, 0, 0)
        item = FeedItem(
            id="item123",
            feed_type=FeedType.MARKET,
            source="Reuters",
            title="Breaking News",
            content="Full article content",
            url="https://example.com/news",
            timestamp=ts,
            metadata={"category": "finance"},
            priority=5,
        )
        assert item.id == "item123"
        assert item.feed_type == FeedType.MARKET
        assert item.source == "Reuters"
        assert item.title == "Breaking News"
        assert item.content == "Full article content"
        assert item.url == "https://example.com/news"
        assert item.timestamp == ts
        assert item.metadata == {"category": "finance"}
        assert item.priority == 5

    def test_to_alert_payload(self):
        item = FeedItem(
            id="alert1",
            feed_type=FeedType.ECONOMIC,
            source="Fed",
            title="Rate Decision",
            priority=10,
            url="https://fed.gov",
        )
        payload = item.to_alert_payload()
        assert payload == {
            "id": "alert1",
            "type": "economic",
            "title": "Rate Decision",
            "priority": 10,
            "url": "https://fed.gov",
            "source": "Fed",
        }


class TestFeedManager:
    """Tests for FeedManager."""

    @pytest.fixture
    def manager(self):
        """Create a FeedManager with mocked feeds."""
        with patch("atlas.data.manager.RSSFeed") as MockRSS, \
             patch("atlas.data.manager.MarketDataFeed") as MockMarket, \
             patch("atlas.data.manager.FedEventMonitor") as MockFed, \
             patch("atlas.data.manager.EconomicCalendar") as MockEconomic:

            # Set up mock fetch methods
            MockRSS.return_value.fetch = AsyncMock(return_value=[
                FeedItem(id="rss1", title="RSS News", priority=1)
            ])
            MockMarket.return_value.fetch = AsyncMock(return_value=[
                FeedItem(id="market1", title="Market Update", priority=2)
            ])
            MockFed.return_value.fetch = AsyncMock(return_value=[])
            MockEconomic.return_value.fetch = AsyncMock(return_value=[])

            # Set up close methods
            MockMarket.return_value.close = AsyncMock()
            MockFed.return_value.close = AsyncMock()
            MockEconomic.return_value.close = AsyncMock()

            return FeedManager(refresh_interval=60)

    @pytest.mark.asyncio
    async def test_fetch_all(self, manager):
        """Test fetching all feeds."""
        items = await manager.fetch_all()
        assert len(items) == 2
        # Should be sorted by priority (descending)
        assert items[0].priority >= items[1].priority

    @pytest.mark.asyncio
    async def test_refresh(self, manager):
        """Test refresh functionality."""
        items = await manager.refresh()
        assert len(items) == 2
        assert manager._last_fetch is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, manager):
        """Test starting and stopping the feed manager."""
        await manager.start()
        assert manager._running is True
        assert manager._task is not None

        await manager.stop()
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_close(self, manager):
        """Test closing all feed clients."""
        await manager.start()
        await manager.close()

        # Verify close methods were called
        manager._market.close.assert_called_once()
        manager._fed.close.assert_called_once()
        manager._economic.close.assert_called_once()

    def test_on_item_callback(self, manager):
        """Test registering item callbacks."""
        callback = MagicMock()
        manager.on_item(callback)
        assert callback in manager._callbacks


class TestFeedManagerEventPublishing:
    """Tests for FeedManager event publishing."""

    @pytest.fixture
    def manager_with_bus(self):
        """Create a FeedManager with mocked feeds and event bus."""
        from atlas.orchestrator.events import EventBus

        bus = EventBus()

        with patch("atlas.data.manager.RSSFeed") as MockRSS, \
             patch("atlas.data.manager.MarketDataFeed") as MockMarket, \
             patch("atlas.data.manager.FedEventMonitor") as MockFed, \
             patch("atlas.data.manager.EconomicCalendar") as MockEconomic:

            MockRSS.return_value.fetch = AsyncMock(return_value=[
                FeedItem(id="item1", title="News 1", priority=5)
            ])
            MockMarket.return_value.fetch = AsyncMock(return_value=[])
            MockFed.return_value.fetch = AsyncMock(return_value=[])
            MockEconomic.return_value.fetch = AsyncMock(return_value=[])

            return FeedManager(event_bus=bus, refresh_interval=60), bus

    @pytest.mark.asyncio
    async def test_refresh_publishes_alerts(self, manager_with_bus):
        """Test that refresh publishes alert events."""
        manager, bus = manager_with_bus

        received_events = []

        async def handler(event):
            received_events.append(event)

        from atlas.orchestrator.events import EventType
        bus.subscribe(EventType.ALERT, handler)

        await manager.refresh()

        # Should have published alert event for the feed item
        assert len(received_events) == 1
        assert received_events[0].payload["title"] == "News 1"


class TestFeedManagerErrorHandling:
    """Tests for FeedManager error handling."""

    @pytest.fixture
    def manager_with_errors(self):
        """Create a FeedManager with feeds that raise errors."""
        with patch("atlas.data.manager.RSSFeed") as MockRSS, \
             patch("atlas.data.manager.MarketDataFeed") as MockMarket, \
             patch("atlas.data.manager.FedEventMonitor") as MockFed, \
             patch("atlas.data.manager.EconomicCalendar") as MockEconomic:

            # One feed raises an error
            MockRSS.return_value.fetch = AsyncMock(
                side_effect=Exception("RSS fetch failed")
            )
            # Others return data
            MockMarket.return_value.fetch = AsyncMock(return_value=[
                FeedItem(id="m1", title="Market")
            ])
            MockFed.return_value.fetch = AsyncMock(return_value=[])
            MockEconomic.return_value.fetch = AsyncMock(return_value=[])

            return FeedManager(refresh_interval=60)

    @pytest.mark.asyncio
    async def test_fetch_all_handles_feed_errors(self, manager_with_errors):
        """Test that errors in one feed don't break others."""
        # Should not raise, and should still return items from working feeds
        items = await manager_with_errors.fetch_all()
        assert len(items) == 1  # Only market feed items
        assert items[0].title == "Market"
