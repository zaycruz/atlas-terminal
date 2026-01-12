"""OpenAI-compatible LLM client for zai.org/GLM-4."""

from __future__ import annotations

import json
from typing import Any, AsyncIterable, Sequence

from openai import AsyncOpenAI

from .base import LLMProvider, Message, StreamChunk, ToolCall
from ..environment import get_llm_api_key, get_llm_base_url, get_llm_model


class ZAIClient:
    """OpenAI-compatible client for zai.org/GLM-4."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or get_llm_api_key()
        self._base_url = base_url or get_llm_base_url()
        self._model = model or get_llm_model()

        if not self._api_key:
            raise ValueError(
                "LLM API key required. Set ZAI_API_KEY or ATLAS_LLM_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    @property
    def name(self) -> str:
        return "zai"

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Message:
        """Non-streaming completion."""
        api_messages = [self._to_api_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = [self._to_api_tool(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=self._parse_arguments(tc.function.arguments),
                )
                for tc in choice.message.tool_calls
            ]

        return Message(
            role="assistant",
            content=choice.message.content or "",
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterable[StreamChunk]:
        """Streaming completion."""
        api_messages = [self._to_api_message(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = [self._to_api_tool(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(**kwargs)

        tool_call_chunks: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_call_chunks[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_chunks[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_call_chunks[idx]["arguments"] += tc.function.arguments

            # Yield content chunks
            if delta.content:
                yield StreamChunk(delta=delta.content)

            # Check for completion
            if finish_reason:
                tool_calls = None
                if tool_call_chunks:
                    tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=self._parse_arguments(tc["arguments"]),
                        )
                        for tc in tool_call_chunks.values()
                    ]

                yield StreamChunk(
                    done=True,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                )
                break

    def _to_api_message(self, message: Message) -> dict[str, Any]:
        """Convert Message to API format."""
        d: dict[str, Any] = {"role": message.role, "content": message.content}

        if message.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in message.tool_calls
            ]

        if message.tool_call_id:
            d["tool_call_id"] = message.tool_call_id

        if message.name:
            d["name"] = message.name

        return d

    def _to_api_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert tool spec to API format."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        }

    def _parse_arguments(self, args: str) -> dict[str, Any]:
        """Parse tool call arguments."""
        if not args:
            return {}
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}


# Global client instance
_llm_client: ZAIClient | None = None


def get_llm_client() -> ZAIClient:
    """Get the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = ZAIClient()
    return _llm_client
