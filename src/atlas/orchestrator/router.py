"""Orchestrator router for natural language to agent routing."""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterable

from .events import EventBus, Event, EventType, get_event_bus
from ..llm import ZAIClient, Message, get_llm_client
from ..agents import AgentRegistry, AgentRequest, AgentUpdate, get_agent_registry
from ..environment import get_system_prompt


ROUTER_PROMPT = """You are an intent router for a financial research platform.

Given a user request, determine which agent(s) should handle it:

Available agents:
{agent_descriptions}

Respond with a JSON object:
{{
  "agent": "agent_name",
  "intent": "refined description of what the user wants",
  "parameters": {{}}  // any extracted parameters
}}

If no specific agent is needed (general conversation), respond:
{{
  "agent": "chat",
  "intent": "user message",
  "parameters": {{}}
}}

Only output valid JSON."""


class OrchestrationRouter:
    """Routes natural language to appropriate agents."""

    def __init__(
        self,
        llm: ZAIClient | None = None,
        registry: AgentRegistry | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._llm = llm or get_llm_client()
        self._registry = registry or get_agent_registry()
        self._bus = event_bus or get_event_bus()
        self._conversation_history: list[Message] = []

    def _get_router_prompt(self) -> str:
        """Build router prompt with available agents."""
        descriptions = self._registry.get_agent_descriptions()
        desc_text = "\n".join(
            f"- {d['name']}: {d['description']}"
            for d in descriptions
        )
        return ROUTER_PROMPT.format(agent_descriptions=desc_text)

    async def route(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterable[Event]:
        """Parse intent and dispatch to agents."""
        context = context or {}

        # Add user message to history
        self._conversation_history.append(
            Message(role="user", content=user_input)
        )

        # Classify intent
        routing = await self._classify_intent(user_input)
        agent_name = routing.get("agent", "chat")
        intent = routing.get("intent", user_input)
        parameters = routing.get("parameters", {})

        # Handle chat (no specific agent)
        if agent_name == "chat":
            async for event in self._handle_chat(user_input):
                yield event
            return

        # Get agent
        agent = self._registry.get(agent_name)
        if not agent:
            # Fall back to chat
            async for event in self._handle_chat(user_input):
                yield event
            return

        # Execute agent
        request = AgentRequest(
            intent=intent,
            parameters=parameters,
            context=context,
        )

        async for update in agent.execute(request):
            event = Event(
                type=EventType.AGENT_UPDATE,
                payload=update.to_event_payload(),
            )
            await self._bus.publish(event)
            yield event

            # On completion, generate summary
            if update.status == "completed" and update.artifacts:
                summary = await self._generate_summary(update)
                yield Event(
                    type=EventType.LLM_STREAM,
                    payload={"start": True},
                )
                yield Event(
                    type=EventType.LLM_STREAM,
                    payload={"token": summary},
                )
                yield Event(
                    type=EventType.LLM_STREAM,
                    payload={"end": True},
                )

    async def _classify_intent(self, user_input: str) -> dict[str, Any]:
        """Use LLM to classify user intent."""
        messages = [
            Message(role="system", content=self._get_router_prompt()),
            Message(role="user", content=user_input),
        ]

        response = await self._llm.complete(messages, temperature=0.1)
        content = response.content.strip()

        # Extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"agent": "chat", "intent": user_input, "parameters": {}}

    async def _handle_chat(self, user_input: str) -> AsyncIterable[Event]:
        """Handle general chat without specific agent."""
        messages = [
            Message(role="system", content=get_system_prompt()),
            *self._conversation_history[-10:],  # Last 10 messages for context
        ]

        # Stream response
        yield Event(type=EventType.LLM_STREAM, payload={"start": True})

        full_response = ""
        async for chunk in self._llm.stream(messages):
            if chunk.delta:
                full_response += chunk.delta
                yield Event(
                    type=EventType.LLM_STREAM,
                    payload={"token": chunk.delta},
                )

        yield Event(type=EventType.LLM_STREAM, payload={"end": True})

        # Add to history
        self._conversation_history.append(
            Message(role="assistant", content=full_response)
        )

    async def _generate_summary(self, update: AgentUpdate) -> str:
        """Generate a summary of agent results."""
        artifacts_summary = ""
        for artifact in update.artifacts or []:
            if artifact["type"] == "metrics":
                data = artifact["data"]
                artifacts_summary += f"\n**Performance Metrics:**\n"
                artifacts_summary += f"- Total Return: {data.get('total_return', 0):.1%}\n"
                artifacts_summary += f"- CAGR: {data.get('cagr', 0):.1%}\n"
                artifacts_summary += f"- Sharpe Ratio: {data.get('sharpe_ratio', 0):.2f}\n"
                artifacts_summary += f"- Max Drawdown: {data.get('max_drawdown', 0):.1%}\n"
                artifacts_summary += f"- Win Rate: {data.get('win_rate', 0):.1%}\n"
                artifacts_summary += f"- Total Trades: {data.get('total_trades', 0)}\n"
            elif artifact["type"] == "analysis":
                artifacts_summary += f"\n**Analysis:**\n{artifact['data'][:500]}...\n"
            elif artifact["type"] == "forecast":
                artifacts_summary += f"\n**Forecast:**\n{artifact['data'][:500]}...\n"

        return f"**{update.agent_name.title()} Agent Complete**\n{update.message}\n{artifacts_summary}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()


# Global router instance
_router: OrchestrationRouter | None = None


def get_router() -> OrchestrationRouter:
    """Get the global orchestration router."""
    global _router
    if _router is None:
        _router = OrchestrationRouter()
    return _router
