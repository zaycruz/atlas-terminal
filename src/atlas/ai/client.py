"""Minimal Ollama client used by Atlas AI mode."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import requests


class OllamaError(RuntimeError):
    """Raised when communication with the Ollama server fails."""


@dataclass
class OllamaMessage:
    role: str
    content: str
    name: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class OllamaResponse:
    message: OllamaMessage
    raw: Mapping[str, Any]


class OllamaClient:
    """Thin wrapper over the Ollama HTTP API."""

    def __init__(self, host: str, model: str, timeout: int = 30) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout

    def chat_stream(
        self,
        messages: Sequence[OllamaMessage],
        tools: Iterable[Mapping[str, Any]] | None = None,
    ) -> Iterable[Mapping[str, Any]]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = list(tools)

        try:
            response = requests.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=self._timeout,
                stream=True,
            )
        except requests.RequestException as exc:  # pragma: no cover - network error
            raise OllamaError(f"Failed to reach Ollama at {self._host}: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama responded with status {response.status_code}: {response.text.strip()}"
            )

        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                yield data
        finally:
            response.close()


    def chat(
        self,
        messages: Sequence[OllamaMessage],
        tools: Iterable[Mapping[str, Any]] | None = None,
    ) -> OllamaResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)

        try:
            response = requests.post(
                f"{self._host}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network error
            raise OllamaError(f"Failed to reach Ollama at {self._host}: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(
                f"Ollama responded with status {response.status_code}: {response.text.strip()}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - invalid response
            raise OllamaError("Failed to decode Ollama response as JSON") from exc

        msg = data.get("message") or {}
        content = msg.get("content", "")
        role = msg.get("role", "assistant")
        return OllamaResponse(OllamaMessage(role=role, content=content), data)

    @property
    def model(self) -> str:
        return self._model
