"""Base protocol and types for agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterable, Protocol
from uuid import uuid4


class AgentCapability(str, Enum):
    """Capabilities that agents can have."""
    CODE_EXECUTION = "code_execution"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"
    MONITORING = "monitoring"
    FORECASTING = "forecasting"
    WEB_SEARCH = "web_search"


@dataclass
class AgentRequest:
    """Request dispatched to a specialized agent."""
    intent: str
    parameters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex[:8])


@dataclass
class AgentUpdate:
    """Progress/status update from an agent."""
    agent_name: str
    status: str  # "queued", "running", "completed", "failed"
    message: str
    progress: float | None = None  # 0.0 - 1.0
    artifacts: list[dict[str, Any]] | None = None
    error: str | None = None
    job_id: str = ""

    def to_event_payload(self) -> dict[str, Any]:
        """Convert to event bus payload."""
        return {
            "job_id": self.job_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "artifacts": self.artifacts,
            "error": self.error,
        }


class Agent(Protocol):
    """Protocol for specialized agents."""

    @property
    def name(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    def capabilities(self) -> set[AgentCapability]:
        """What this agent can do."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for routing."""
        ...

    async def execute(self, request: AgentRequest) -> AsyncIterable[AgentUpdate]:
        """Execute the request, yielding progress updates."""
        ...
