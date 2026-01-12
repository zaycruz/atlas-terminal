"""Agent registry for managing available agents."""

from __future__ import annotations

from typing import Dict, Iterable

from .base import Agent, AgentCapability


class AgentRegistry:
    """Registry of available agents."""

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent

    def unregister(self, name: str) -> bool:
        """Unregister an agent by name."""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def get(self, name: str) -> Agent | None:
        """Get an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def find_by_capability(self, capability: AgentCapability) -> list[Agent]:
        """Find agents with a specific capability."""
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    def get_agent_descriptions(self) -> list[dict[str, str]]:
        """Get descriptions for all agents (for LLM routing)."""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "capabilities": [c.value for c in agent.capabilities],
            }
            for agent in self._agents.values()
        ]

    def __iter__(self) -> Iterable[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)


# Global registry instance
_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
