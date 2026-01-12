"""Base protocol and types for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterable, Protocol, Sequence


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to API format."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": str(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class StreamChunk:
    """A streaming response chunk."""
    delta: str = ""
    tool_calls: list[ToolCall] | None = None
    done: bool = False
    finish_reason: str | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def name(self) -> str:
        """Provider identifier."""
        ...

    @property
    def model(self) -> str:
        """Current model name."""
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Message:
        """Non-streaming completion."""
        ...

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterable[StreamChunk]:
        """Streaming completion."""
        ...
