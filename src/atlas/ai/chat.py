"""Conversation loop for Atlas AI mode."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Mapping

from rich.console import Console
from termcolor import colored

from ..brokers import AlpacaBroker, Order
from ..brokers.base import BrokerError
from ..terminal import render_account, render_orders, render_positions, render_quote
from .client import OllamaClient, OllamaMessage, OllamaResponse, OllamaError
from .tools import AVAILABLE_TOOLS, ToolResult, iter_tool_specs, run_tool

TOOL_BLOCK_PATTERN = re.compile(r"```atlas_tool\s*(\{.*?\})\s*```", re.DOTALL)
MAX_TOOL_ITERATIONS = 3


@dataclass
class AIChatConfig:
    host: str
    model: str
    system_prompt: str
    environment: str


def _extract_tool_calls(content: str) -> List[Mapping[str, object]]:
    calls: List[Mapping[str, object]] = []
    for raw in TOOL_BLOCK_PATTERN.findall(content):
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "tool" in data:
                calls.append(data)
        except json.JSONDecodeError:
            continue
    return calls


def _strip_tool_blocks(content: str) -> str:
    return TOOL_BLOCK_PATTERN.sub("", content).strip()


def _render_tool_result(console: Console, env: str, result: ToolResult) -> None:
    if result.message and result.name not in {"buy", "sell", "cancel"}:
        console.print(colored(result.message, "cyan"))
    name = result.name
    data = result.data
    if name == "account":
        render_account(console, env, data)  # type: ignore[arg-type]
    elif name == "positions":
        render_positions(console, data)  # type: ignore[arg-type]
    elif name == "orders":
        render_orders(console, data, None)  # type: ignore[arg-type]
    elif name in {"buy", "sell"}:
        order: Order = data  # type: ignore[assignment]
        console.print(colored(result.message, "green"))
        console.print(json.dumps(asdict(order), default=str, indent=2))
    elif name == "cancel":
        console.print(colored(result.message, "green"))
    elif name == "quote":
        symbol = data.get("symbol") if isinstance(data, dict) else None
        quote = data.get("quote") if isinstance(data, dict) else None
        if isinstance(quote, dict):
            render_quote(console, symbol or "", quote)
        else:
            console.print(data)
    else:
        console.print(json.dumps(result.to_model_dict(), indent=2))


def _tool_result_payload(result: ToolResult) -> str:
    return json.dumps(result.to_model_dict())


def _tool_error_payload(name: str, error: str) -> str:
    return json.dumps({"tool": name, "success": False, "error": error})


def run_chat(broker: AlpacaBroker, config: AIChatConfig, console: Console | None = None) -> None:
    console = console or Console()
    client = OllamaClient(config.host, config.model)
    tool_specs = list(iter_tool_specs())

    tool_summary = "\n".join(
        f"- {tool.name}: {tool.description}" for tool in AVAILABLE_TOOLS.values()
    )
    guidance = (
        "Use tools only when broker data or actions are explicitly needed.\n"
        "If the user is greeting or chatting, reply naturally without calling a tool.\n"
        "When you invoke a tool, respond with JSON inside a fenced block tagged `atlas_tool`.\n"
        "Example:\n"
        "```atlas_tool\n"
        "{\"tool\": \"quote\", \"args\": {\"symbol\": \"AAPL\"}}\n"
        "```\n"
        "After the tool response is provided, explain the result in plain language.\n"
        "Confirm with the user before executing trade-altering tools like buy, sell, or cancel."
    )
    system_message = "\n".join([
        config.system_prompt,
        "",
        guidance,
        "Tools:",
        tool_summary,
    ])

    history: List[OllamaMessage] = [
        OllamaMessage(role="system", content=system_message),
    ]

    console.print(colored(f"AI chat mode using model '{client.model}'. Type 'exit' to leave.", "cyan"))

    while True:
        try:
            user_input = input(colored("you: ", "magenta"))
        except (KeyboardInterrupt, EOFError):
            console.print(colored("Exiting AI mode.", "cyan"))
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            console.print(colored("Bye.", "cyan"))
            break

        history.append(OllamaMessage(role="user", content=user_input))

        try:
            _generate_reply(history, client, tool_specs, broker, console, config)
        except OllamaError as exc:
            console.print(colored(str(exc), "red"))
            history.pop()  # Remove the last user input to retry later if desired
            continue
        except BrokerError as exc:
            console.print(colored(str(exc), "red"))
            continue


def _stream_model_response(
    history: List[OllamaMessage],
    client: OllamaClient,
    tool_specs,
    console: Console,
) -> tuple[str, bool]:
    """Stream assistant tokens while capturing the complete response."""
    assistant_text = ""
    should_print = True
    prefix_printed = False
    printed_any = False
    stream = console.file

    for chunk in client.chat_stream(history, tool_specs):
        if isinstance(chunk, dict) and chunk.get("error"):
            raise OllamaError(chunk["error"])

        message = chunk.get("message") if isinstance(chunk, dict) else None
        delta = ""
        if isinstance(message, dict):
            delta = message.get("content") or ""
        if delta:
            assistant_text += delta
            if should_print and "```atlas_tool" in assistant_text:
                should_print = False
                continue
            if should_print:
                if not prefix_printed:
                    console.print("[green]atlas-ai[/green]: ", end="")
                    prefix_printed = True
                stream.write(delta)
                stream.flush()
                printed_any = True

        if isinstance(chunk, dict) and chunk.get("done"):
            break

    displayed = False
    if should_print:
        if prefix_printed:
            stream.write("\n")
            stream.flush()
            displayed = True
        elif assistant_text.strip():
            console.print(f"[green]atlas-ai[/green]: {assistant_text}")
            displayed = True
    else:
        if prefix_printed and printed_any:
            stream.write("\n")
            stream.flush()

    return assistant_text, displayed

def _generate_reply(
    history: List[OllamaMessage],
    client: OllamaClient,
    tool_specs,
    broker: AlpacaBroker,
    console: Console,
    config: AIChatConfig,
) -> str:
    for _ in range(MAX_TOOL_ITERATIONS):
        assistant_text, displayed = _stream_model_response(history, client, tool_specs, console)
        history.append(OllamaMessage(role="assistant", content=assistant_text))

        tool_calls = _extract_tool_calls(assistant_text)
        if not tool_calls:
            return _strip_tool_blocks(assistant_text)

        for call in tool_calls:
            tool_name = str(call.get("tool"))
            args = call.get("args") or {}
            if not isinstance(args, dict):
                args = {}

            console.print(f"[green]atlas-ai[/green]: invoking `{tool_name}` tool")
            try:
                result = run_tool(tool_name, broker, args)
            except BrokerError as exc:
                console.print(colored(f"Tool '{tool_name}' error: {exc}", "red"))
                history.append(
                    OllamaMessage(
                        role="tool",
                        content=_tool_error_payload(tool_name, str(exc)),
                        name=tool_name,
                    )
                )
                continue

            _render_tool_result(console, config.environment, result)
            history.append(
                OllamaMessage(
                    role="tool",
                    content=_tool_result_payload(result),
                    name=tool_name,
                )
            )

    raise BrokerError("Maximum tool iterations exceeded. Conversation aborted.")
