"""Tests for the orchestration router."""

import pytest
from typing import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, patch

from atlas.orchestrator.router import OrchestrationRouter, ROUTER_PROMPT
from atlas.orchestrator.events import EventBus, Event, EventType
from atlas.agents.base import Agent, AgentCapability, AgentRequest, AgentUpdate
from atlas.agents.registry import AgentRegistry
from atlas.llm.base import Message, StreamChunk


class MockAgent:
    """A mock agent for testing."""

    def __init__(self, name: str = "mock", desc: str = "Mock agent"):
        self._name = name
        self._desc = desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[AgentCapability]:
        return {AgentCapability.DATA_ANALYSIS}

    @property
    def description(self) -> str:
        return self._desc

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        yield AgentUpdate(
            agent_name=self.name,
            status="running",
            message="Processing...",
            job_id="job_123",
        )
        yield AgentUpdate(
            agent_name=self.name,
            status="completed",
            message="Done!",
            job_id="job_123",
            artifacts=[{"type": "metrics", "data": {"total_return": 0.15}}],
        )


class MockLLMClient:
    """A mock LLM client for testing."""

    def __init__(self, classification: dict | None = None, chat_response: str = "Hello!"):
        self._classification = classification or {
            "agent": "chat",
            "intent": "test",
            "parameters": {},
        }
        self._chat_response = chat_response

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model(self) -> str:
        return "mock-model"

    async def complete(self, messages, **kwargs) -> Message:
        import json
        return Message(
            role="assistant",
            content=json.dumps(self._classification),
        )

    async def stream(self, messages, **kwargs):
        # Simulate streaming by yielding tokens
        for token in self._chat_response.split():
            yield StreamChunk(delta=token + " ")
        yield StreamChunk(done=True, finish_reason="stop")


class TestRouterPrompt:
    """Tests for router prompt generation."""

    def test_router_prompt_has_placeholders(self):
        assert "{agent_descriptions}" in ROUTER_PROMPT

    def test_router_prompt_has_json_format(self):
        assert '"agent"' in ROUTER_PROMPT
        assert '"intent"' in ROUTER_PROMPT
        assert '"parameters"' in ROUTER_PROMPT


class TestOrchestrationRouter:
    """Tests for OrchestrationRouter."""

    @pytest.fixture
    def registry(self):
        """Create a registry with mock agents."""
        reg = AgentRegistry()
        reg.register(MockAgent(name="backtest", desc="Run backtests"))
        reg.register(MockAgent(name="analysis", desc="Run analysis"))
        return reg

    @pytest.fixture
    def event_bus(self):
        """Create an event bus."""
        return EventBus()

    @pytest.fixture
    def router(self, registry, event_bus):
        """Create a router with mocked LLM."""
        llm = MockLLMClient()
        return OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

    def test_get_router_prompt_includes_agents(self, router):
        """Test that router prompt includes agent descriptions."""
        prompt = router._get_router_prompt()
        assert "backtest" in prompt
        assert "analysis" in prompt
        assert "Run backtests" in prompt
        assert "Run analysis" in prompt

    @pytest.mark.asyncio
    async def test_classify_intent_chat(self, registry, event_bus):
        """Test classifying a chat intent."""
        llm = MockLLMClient(classification={
            "agent": "chat",
            "intent": "greeting",
            "parameters": {},
        })
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        result = await router._classify_intent("Hello!")
        assert result["agent"] == "chat"

    @pytest.mark.asyncio
    async def test_classify_intent_backtest(self, registry, event_bus):
        """Test classifying a backtest intent."""
        llm = MockLLMClient(classification={
            "agent": "backtest",
            "intent": "Run backtest on AAPL",
            "parameters": {"symbol": "AAPL"},
        })
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        result = await router._classify_intent("Backtest AAPL with EMA strategy")
        assert result["agent"] == "backtest"
        assert result["parameters"]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_classify_intent_invalid_json_falls_back_to_chat(self, registry, event_bus):
        """Test that invalid JSON classification falls back to chat."""
        llm = MockLLMClient()
        # Override complete to return invalid JSON
        async def bad_complete(messages, **kwargs):
            return Message(role="assistant", content="not valid json")
        llm.complete = bad_complete

        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        result = await router._classify_intent("test")
        assert result["agent"] == "chat"

    @pytest.mark.asyncio
    async def test_route_to_chat(self, registry, event_bus):
        """Test routing to chat handler."""
        llm = MockLLMClient(
            classification={"agent": "chat", "intent": "greeting", "parameters": {}},
            chat_response="Hello! How can I help?",
        )
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        events = []
        async for event in router.route("Hi there"):
            events.append(event)

        # Should have LLM_STREAM events
        assert any(e.type == EventType.LLM_STREAM for e in events)
        # Should have start, content, and end
        assert events[0].payload.get("start") is True
        assert events[-1].payload.get("end") is True

    @pytest.mark.asyncio
    async def test_route_to_agent(self, registry, event_bus):
        """Test routing to a specific agent."""
        llm = MockLLMClient(classification={
            "agent": "backtest",
            "intent": "Run backtest",
            "parameters": {},
        })
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        events = []
        async for event in router.route("Backtest AAPL"):
            events.append(event)

        # Should have AGENT_UPDATE events
        agent_updates = [e for e in events if e.type == EventType.AGENT_UPDATE]
        assert len(agent_updates) >= 2

        # First update should be running
        assert agent_updates[0].payload["status"] == "running"
        # Last agent update should be completed
        completed = [u for u in agent_updates if u.payload["status"] == "completed"]
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_route_unknown_agent_falls_back_to_chat(self, registry, event_bus):
        """Test that unknown agent falls back to chat."""
        llm = MockLLMClient(
            classification={"agent": "unknown_agent", "intent": "test", "parameters": {}},
            chat_response="I'll help you with that.",
        )
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        events = []
        async for event in router.route("Do something"):
            events.append(event)

        # Should have LLM_STREAM events (from chat fallback)
        assert any(e.type == EventType.LLM_STREAM for e in events)

    @pytest.mark.asyncio
    async def test_conversation_history_tracking(self, registry, event_bus):
        """Test that conversation history is tracked."""
        llm = MockLLMClient(
            classification={"agent": "chat", "intent": "test", "parameters": {}},
            chat_response="Response",
        )
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        # Make some calls
        async for _ in router.route("First message"):
            pass
        async for _ in router.route("Second message"):
            pass

        # Should have conversation history
        assert len(router._conversation_history) >= 2

    @pytest.mark.asyncio
    async def test_clear_history(self, registry, event_bus):
        """Test clearing conversation history."""
        llm = MockLLMClient(
            classification={"agent": "chat", "intent": "test", "parameters": {}},
            chat_response="Response",
        )
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        async for _ in router.route("Message"):
            pass

        assert len(router._conversation_history) > 0

        router.clear_history()
        assert len(router._conversation_history) == 0

    @pytest.mark.asyncio
    async def test_events_published_to_bus(self, registry, event_bus):
        """Test that events are published to the event bus."""
        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        event_bus.subscribe(EventType.AGENT_UPDATE, handler)

        llm = MockLLMClient(classification={
            "agent": "backtest",
            "intent": "Run backtest",
            "parameters": {},
        })
        router = OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

        async for _ in router.route("Backtest AAPL"):
            pass

        # Events should have been published to bus
        assert len(received_events) >= 1


class TestGenerateSummary:
    """Tests for summary generation."""

    @pytest.fixture
    def router(self):
        """Create a minimal router for testing."""
        llm = MockLLMClient()
        registry = AgentRegistry()
        event_bus = EventBus()
        return OrchestrationRouter(llm=llm, registry=registry, event_bus=event_bus)

    @pytest.mark.asyncio
    async def test_generate_summary_with_metrics(self, router):
        """Test summary generation with metrics artifact."""
        update = AgentUpdate(
            agent_name="backtest",
            status="completed",
            message="Backtest complete",
            artifacts=[{
                "type": "metrics",
                "data": {
                    "total_return": 0.25,
                    "cagr": 0.15,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.10,
                    "win_rate": 0.55,
                    "total_trades": 100,
                },
            }],
        )

        summary = await router._generate_summary(update)
        assert "Backtest Agent Complete" in summary
        assert "25.0%" in summary  # total_return
        assert "15.0%" in summary  # cagr
        assert "1.50" in summary  # sharpe
        assert "-10.0%" in summary  # max_drawdown

    @pytest.mark.asyncio
    async def test_generate_summary_with_analysis(self, router):
        """Test summary generation with analysis artifact."""
        update = AgentUpdate(
            agent_name="analysis",
            status="completed",
            message="Analysis complete",
            artifacts=[{
                "type": "analysis",
                "data": "This is the analysis result text " * 20,
            }],
        )

        summary = await router._generate_summary(update)
        assert "Analysis Agent Complete" in summary
        assert "Analysis:" in summary

    @pytest.mark.asyncio
    async def test_generate_summary_with_forecast(self, router):
        """Test summary generation with forecast artifact."""
        update = AgentUpdate(
            agent_name="forecasting",
            status="completed",
            message="Forecast complete",
            artifacts=[{
                "type": "forecast",
                "data": "Forecast data here",
            }],
        )

        summary = await router._generate_summary(update)
        assert "Forecasting Agent Complete" in summary
        assert "Forecast:" in summary
