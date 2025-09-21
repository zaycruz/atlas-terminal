"""Lightweight client scaffolding for the Docker MCP server."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import asyncio
import threading
from fastmcp import Client as FastMCPClient
from fastmcp.client.client import ToolError
from mcp import types as mcp_types


class MCPToolError(RuntimeError):
    """Raised when the MCP server reports an error while executing a tool."""


@dataclass
class DockerMCPConfig:
    """Runtime configuration for the Docker MCP client."""

    endpoint: str
    token: str | None = None
    default_image: str = "python:3.11-slim"
    request_timeout: float = 120.0


class DockerMCPClient:
    """Scaffolding for interacting with the docker_mcp FastMCP server.

    The concrete transport (HTTP/WebSocket) is not wired up yet; this class
    provides the interface the backtesting agent will rely on. Implementations
    should override :meth:`_call_tool` with the actual MCP invocation logic.
    """

    def __init__(self, config: DockerMCPConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public tool wrappers
    def list_containers(self, show_all: bool = True) -> Any:
        return self._call_tool(
            "list_containers", {"show_all": show_all}
        )

    def create_container(
        self,
        image: str | None = None,
        *,
        container_name: str,
        dependencies: str | None = None,
    ) -> Any:
        payload = {
            "image": image or self._config.default_image,
            "container_name": container_name,
        }
        if dependencies:
            payload["dependencies"] = dependencies
        return self._call_tool("create_container", payload)

    def add_dependencies(self, container_name: str, dependencies: str) -> Any:
        return self._call_tool(
            "add_dependencies",
            {"container_name": container_name, "dependencies": dependencies},
        )

    def execute_code(self, container_name: str, command: str) -> Any:
        return self._call_tool(
            "execute_code",
            {"container_name": container_name, "command": command},
        )

    def execute_python_script(
        self,
        container_name: str,
        script_content: str,
        *,
        script_args: Sequence[str] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "container_name": container_name,
            "script_content": script_content,
        }
        if script_args:
            payload["script_args"] = list(script_args)
        return self._call_tool("execute_python_script", payload)

    def cleanup_container(self, container_name: str) -> Any:
        return self._call_tool("cleanup_container", {"container_name": container_name})

    # ------------------------------------------------------------------
    # Transport hook
    def _call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        """Dispatch a tool call to the MCP server.

        This base implementation only raises ``NotImplementedError``. Concrete
        transports (HTTP/WebSocket) should subclass :class:`DockerMCPClient` or
        monkeypatch :meth:`_call_tool` to perform the actual request/response
        cycle with the MCP server.
        """

        raise NotImplementedError(
            "DockerMCPClient._call_tool must be implemented by the transport layer"
        )



class FastMCPDockerClient(DockerMCPClient):
    """Concrete Docker MCP client using fastmcp.Client under the hood."""

    def __init__(self, config: DockerMCPConfig) -> None:
        super().__init__(config)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        coroutine = self._call_tool_async(name, dict(arguments))
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self._create_timeout())
        except ToolError as exc:  # pragma: no cover - bubbled from async context
            raise MCPToolError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network/runtime issues
            raise exc

    async def _call_tool_async(self, name: str, arguments: Mapping[str, Any]) -> Any:
        client_kwargs: dict[str, Any] = {
            "transport": self._config.endpoint,
            "timeout": self._config.request_timeout,
        }
        if self._config.token:
            client_kwargs["auth"] = self._config.token

        async with FastMCPClient(**client_kwargs) as client:
            try:
                result = await client.call_tool(name, dict(arguments))
            except ToolError as exc:
                raise MCPToolError(str(exc)) from exc

        if result.data is not None:
            return result.data
        if result.structured_content:
            return result.structured_content
        return [self._content_block_to_python(block) for block in result.content]

    def _content_block_to_python(self, block: mcp_types.ContentBlock) -> Any:
        if isinstance(block, mcp_types.TextContent):
            return block.text
        if isinstance(block, mcp_types.ImageContent):
            return {"type": "image", "data": block.data, "mimeType": block.mimeType}
        if isinstance(block, mcp_types.AudioContent):
            return {"type": "audio", "data": block.data, "mimeType": block.mimeType}
        return block

    def _create_timeout(self) -> float:
        return float(self._config.request_timeout or 120.0)


__all__ = ["DockerMCPConfig", "DockerMCPClient", "MCPToolError", "FastMCPDockerClient"]
