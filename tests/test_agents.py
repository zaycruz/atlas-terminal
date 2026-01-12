"""Tests for the agent base classes and registry."""

import pytest
from typing import AsyncIterable

from atlas.agents.base import (
    Agent,
    AgentCapability,
    AgentRequest,
    AgentUpdate,
)
from atlas.agents.registry import AgentRegistry, get_agent_registry


class TestAgentRequest:
    """Tests for AgentRequest dataclass."""

    def test_creation_with_defaults(self):
        req = AgentRequest(intent="test intent")
        assert req.intent == "test intent"
        assert req.parameters == {}
        assert req.context == {}
        assert len(req.request_id) == 8

    def test_creation_with_all_fields(self):
        req = AgentRequest(
            intent="backtest",
            parameters={"symbol": "AAPL"},
            context={"user_id": "123"},
            request_id="custom12",
        )
        assert req.intent == "backtest"
        assert req.parameters == {"symbol": "AAPL"}
        assert req.context == {"user_id": "123"}
        assert req.request_id == "custom12"


class TestAgentUpdate:
    """Tests for AgentUpdate dataclass."""

    def test_creation_with_required_fields(self):
        update = AgentUpdate(
            agent_name="test_agent",
            status="running",
            message="Processing...",
        )
        assert update.agent_name == "test_agent"
        assert update.status == "running"
        assert update.message == "Processing..."
        assert update.progress is None
        assert update.artifacts is None
        assert update.error is None
        assert update.job_id == ""

    def test_creation_with_all_fields(self):
        update = AgentUpdate(
            agent_name="backtest",
            status="completed",
            message="Backtest complete",
            progress=1.0,
            artifacts=[{"type": "metrics", "data": {}}],
            error=None,
            job_id="job123",
        )
        assert update.agent_name == "backtest"
        assert update.status == "completed"
        assert update.progress == 1.0
        assert len(update.artifacts) == 1
        assert update.job_id == "job123"

    def test_to_event_payload(self):
        update = AgentUpdate(
            agent_name="test",
            status="running",
            message="Working",
            progress=0.5,
            artifacts=[{"type": "data"}],
            error=None,
            job_id="j1",
        )
        payload = update.to_event_payload()
        assert payload == {
            "job_id": "j1",
            "agent_name": "test",
            "status": "running",
            "message": "Working",
            "progress": 0.5,
            "artifacts": [{"type": "data"}],
            "error": None,
        }

    def test_to_event_payload_with_error(self):
        update = AgentUpdate(
            agent_name="test",
            status="failed",
            message="Failed",
            error="Connection timeout",
            job_id="j2",
        )
        payload = update.to_event_payload()
        assert payload["status"] == "failed"
        assert payload["error"] == "Connection timeout"


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_capability_values(self):
        assert AgentCapability.CODE_EXECUTION.value == "code_execution"
        assert AgentCapability.DATA_ANALYSIS.value == "data_analysis"
        assert AgentCapability.RESEARCH.value == "research"
        assert AgentCapability.MONITORING.value == "monitoring"
        assert AgentCapability.FORECASTING.value == "forecasting"
        assert AgentCapability.WEB_SEARCH.value == "web_search"


class MockAgent:
    """A mock agent for testing."""

    def __init__(self, name: str = "mock", caps: set | None = None, desc: str = "A mock agent"):
        self._name = name
        self._caps = caps or {AgentCapability.DATA_ANALYSIS}
        self._desc = desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[AgentCapability]:
        return self._caps

    @property
    def description(self) -> str:
        return self._desc

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        yield AgentUpdate(
            agent_name=self.name,
            status="completed",
            message="Mock execution complete",
        )


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return AgentRegistry()

    def test_register_agent(self, registry):
        agent = MockAgent(name="test_agent")
        registry.register(agent)
        assert "test_agent" in registry.list_agents()

    def test_register_multiple_agents(self, registry):
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")
        registry.register(agent1)
        registry.register(agent2)
        assert len(registry) == 2
        assert set(registry.list_agents()) == {"agent1", "agent2"}

    def test_get_registered_agent(self, registry):
        agent = MockAgent(name="test")
        registry.register(agent)
        retrieved = registry.get("test")
        assert retrieved is agent

    def test_get_unregistered_agent_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_unregister_agent(self, registry):
        agent = MockAgent(name="test")
        registry.register(agent)
        assert registry.unregister("test") is True
        assert registry.get("test") is None
        assert len(registry) == 0

    def test_unregister_unknown_agent_returns_false(self, registry):
        assert registry.unregister("unknown") is False

    def test_find_by_capability(self, registry):
        agent1 = MockAgent(name="data", caps={AgentCapability.DATA_ANALYSIS})
        agent2 = MockAgent(name="code", caps={AgentCapability.CODE_EXECUTION})
        agent3 = MockAgent(
            name="multi",
            caps={AgentCapability.DATA_ANALYSIS, AgentCapability.CODE_EXECUTION},
        )
        registry.register(agent1)
        registry.register(agent2)
        registry.register(agent3)

        data_agents = registry.find_by_capability(AgentCapability.DATA_ANALYSIS)
        assert len(data_agents) == 2
        names = {a.name for a in data_agents}
        assert names == {"data", "multi"}

        code_agents = registry.find_by_capability(AgentCapability.CODE_EXECUTION)
        assert len(code_agents) == 2
        names = {a.name for a in code_agents}
        assert names == {"code", "multi"}

    def test_find_by_capability_no_match(self, registry):
        agent = MockAgent(name="data", caps={AgentCapability.DATA_ANALYSIS})
        registry.register(agent)
        agents = registry.find_by_capability(AgentCapability.WEB_SEARCH)
        assert len(agents) == 0

    def test_get_agent_descriptions(self, registry):
        agent1 = MockAgent(
            name="test1",
            caps={AgentCapability.DATA_ANALYSIS},
            desc="Test agent 1",
        )
        agent2 = MockAgent(
            name="test2",
            caps={AgentCapability.CODE_EXECUTION, AgentCapability.RESEARCH},
            desc="Test agent 2",
        )
        registry.register(agent1)
        registry.register(agent2)

        descriptions = registry.get_agent_descriptions()
        assert len(descriptions) == 2

        desc1 = next(d for d in descriptions if d["name"] == "test1")
        assert desc1["description"] == "Test agent 1"
        assert desc1["capabilities"] == ["data_analysis"]

        desc2 = next(d for d in descriptions if d["name"] == "test2")
        assert set(desc2["capabilities"]) == {"code_execution", "research"}

    def test_registry_iteration(self, registry):
        agents = [MockAgent(name=f"agent{i}") for i in range(3)]
        for agent in agents:
            registry.register(agent)

        iterated = list(registry)
        assert len(iterated) == 3

    def test_registry_len(self, registry):
        assert len(registry) == 0
        registry.register(MockAgent(name="a"))
        assert len(registry) == 1
        registry.register(MockAgent(name="b"))
        assert len(registry) == 2


class TestGetAgentRegistry:
    """Tests for the global registry singleton."""

    def test_get_agent_registry_returns_same_instance(self):
        reg1 = get_agent_registry()
        reg2 = get_agent_registry()
        assert reg1 is reg2

    def test_get_agent_registry_returns_agentregistry(self):
        reg = get_agent_registry()
        assert isinstance(reg, AgentRegistry)


class TestMockAgentExecution:
    """Tests for agent execution flow."""

    @pytest.mark.asyncio
    async def test_mock_agent_execute(self):
        agent = MockAgent(name="test")
        request = AgentRequest(intent="test intent")

        updates = []
        async for update in agent.execute(request):
            updates.append(update)

        assert len(updates) == 1
        assert updates[0].agent_name == "test"
        assert updates[0].status == "completed"
