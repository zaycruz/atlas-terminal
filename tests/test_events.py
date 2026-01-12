"""Tests for the event bus system."""

import asyncio
import pytest
from atlas.orchestrator.events import EventBus, Event, EventType, get_event_bus


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation_with_defaults(self):
        event = Event(type=EventType.ALERT, payload={"key": "value"})
        assert event.type == EventType.ALERT
        assert event.payload == {"key": "value"}
        assert event.timestamp > 0
        assert len(event.id) == 8

    def test_event_creation_with_custom_values(self):
        event = Event(
            type=EventType.AGENT_UPDATE,
            payload={"status": "running"},
            timestamp=12345.0,
            id="custom12",
        )
        assert event.type == EventType.AGENT_UPDATE
        assert event.payload == {"status": "running"}
        assert event.timestamp == 12345.0
        assert event.id == "custom12"


class TestEventBus:
    """Tests for EventBus functionality."""

    @pytest.fixture
    def bus(self):
        """Create a fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, bus):
        """Test basic subscribe and publish."""
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.ALERT, handler)

        event = Event(type=EventType.ALERT, payload={"message": "test"})
        await bus.publish(event)

        assert len(received) == 1
        assert received[0].payload == {"message": "test"}

    @pytest.mark.asyncio
    async def test_subscribe_returns_id(self, bus):
        """Test that subscribe returns a subscription ID."""
        async def handler(event: Event):
            pass

        sub_id = bus.subscribe(EventType.ALERT, handler)
        assert isinstance(sub_id, str)
        assert len(sub_id) == 12

    @pytest.mark.asyncio
    async def test_subscribe_different_types(self, bus):
        """Test subscribers only receive events of subscribed type."""
        alert_received = []
        update_received = []

        async def alert_handler(event: Event):
            alert_received.append(event)

        async def update_handler(event: Event):
            update_received.append(event)

        bus.subscribe(EventType.ALERT, alert_handler)
        bus.subscribe(EventType.AGENT_UPDATE, update_handler)

        await bus.publish(Event(type=EventType.ALERT, payload={"type": "alert"}))
        await bus.publish(Event(type=EventType.AGENT_UPDATE, payload={"type": "update"}))

        assert len(alert_received) == 1
        assert alert_received[0].payload["type"] == "alert"
        assert len(update_received) == 1
        assert update_received[0].payload["type"] == "update"

    @pytest.mark.asyncio
    async def test_subscribe_all(self, bus):
        """Test global subscription to all event types."""
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe_all(handler)

        await bus.publish(Event(type=EventType.ALERT, payload={"n": 1}))
        await bus.publish(Event(type=EventType.AGENT_UPDATE, payload={"n": 2}))
        await bus.publish(Event(type=EventType.LLM_STREAM, payload={"n": 3}))

        assert len(received) == 3
        assert [r.payload["n"] for r in received] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_unsubscribe_typed(self, bus):
        """Test unsubscribing from a typed subscription."""
        received = []

        async def handler(event: Event):
            received.append(event)

        sub_id = bus.subscribe(EventType.ALERT, handler)

        await bus.publish(Event(type=EventType.ALERT, payload={"n": 1}))
        assert len(received) == 1

        result = bus.unsubscribe(sub_id)
        assert result is True

        await bus.publish(Event(type=EventType.ALERT, payload={"n": 2}))
        assert len(received) == 1  # No new events

    @pytest.mark.asyncio
    async def test_unsubscribe_global(self, bus):
        """Test unsubscribing from a global subscription."""
        received = []

        async def handler(event: Event):
            received.append(event)

        sub_id = bus.subscribe_all(handler)

        await bus.publish(Event(type=EventType.ALERT, payload={"n": 1}))
        assert len(received) == 1

        result = bus.unsubscribe(sub_id)
        assert result is True

        await bus.publish(Event(type=EventType.ALERT, payload={"n": 2}))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_unknown_id(self, bus):
        """Test unsubscribing with unknown ID returns False."""
        result = bus.unsubscribe("unknown12345")
        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_type(self, bus):
        """Test multiple subscribers for the same event type."""
        received1 = []
        received2 = []

        async def handler1(event: Event):
            received1.append(event)

        async def handler2(event: Event):
            received2.append(event)

        bus.subscribe(EventType.ALERT, handler1)
        bus.subscribe(EventType.ALERT, handler2)

        await bus.publish(Event(type=EventType.ALERT, payload={"message": "test"}))

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_sync_callback(self, bus):
        """Test that sync callbacks also work."""
        received = []

        def sync_handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.ALERT, sync_handler)

        await bus.publish(Event(type=EventType.ALERT, payload={"sync": True}))

        assert len(received) == 1
        assert received[0].payload["sync"] is True

    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_break_others(self, bus):
        """Test that one callback failing doesn't prevent others from running."""
        received = []

        async def failing_handler(event: Event):
            raise ValueError("Handler error")

        async def working_handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.ALERT, failing_handler)
        bus.subscribe(EventType.ALERT, working_handler)

        # Should not raise, and working_handler should still receive event
        await bus.publish(Event(type=EventType.ALERT, payload={"test": True}))

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self, bus):
        """Test publishing when no subscribers exist."""
        # Should not raise
        await bus.publish(Event(type=EventType.ALERT, payload={"lonely": True}))


class TestGetEventBus:
    """Tests for the global event bus singleton."""

    def test_get_event_bus_returns_same_instance(self):
        """Test that get_event_bus returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_get_event_bus_returns_eventbus(self):
        """Test that get_event_bus returns an EventBus instance."""
        bus = get_event_bus()
        assert isinstance(bus, EventBus)


class TestEventTypes:
    """Tests for EventType enum."""

    def test_event_type_values(self):
        """Test that EventType has expected values."""
        assert EventType.AGENT_UPDATE.value == "agent_update"
        assert EventType.DATA_UPDATE.value == "data_update"
        assert EventType.ALERT.value == "alert"
        assert EventType.JOB_PROGRESS.value == "job_progress"
        assert EventType.LLM_STREAM.value == "llm_stream"
        assert EventType.USER_MESSAGE.value == "user_message"
        assert EventType.SYSTEM_MESSAGE.value == "system_message"

    def test_event_type_is_string_enum(self):
        """Test that EventType can be used as string."""
        assert str(EventType.ALERT) == "EventType.ALERT"
        assert EventType.ALERT == "alert"
