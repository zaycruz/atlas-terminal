"""Tool registry exposed to the Atlas AI assistant."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, Mapping

import requests
from bs4 import BeautifulSoup

from ..brokers import Account, AlpacaBroker, Order, Position
from ..brokers.base import BrokerError
from ..environment import get_searxng_categories, get_searxng_endpoint

ToolHandler = Callable[[AlpacaBroker, Mapping[str, Any]], "ToolResult"]


@dataclass
class ToolResult:
    success: bool
    name: str
    data: Any
    message: str = ""

    def to_model_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "success": self.success,
            "message": self.message,
            "data": _serialize(self.data),
        }


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler

    def to_llm_spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def run(self, broker: AlpacaBroker, raw_args: Mapping[str, Any]) -> ToolResult:
        return self.handler(broker, raw_args or {})


def _serialize(value: Any) -> Any:
    """Convert nested results into JSON-serialisable structures."""
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Individual tool implementations


def _account_tool(broker: AlpacaBroker, _: Mapping[str, Any]) -> ToolResult:
    account = broker.get_account()
    return ToolResult(True, "account", account, "Fetched account details")


def _positions_tool(broker: AlpacaBroker, _: Mapping[str, Any]) -> ToolResult:
    positions = list(broker.get_positions())
    return ToolResult(True, "positions", positions, "Fetched open positions")


def _orders_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    status = args.get("status")
    orders = list(broker.get_orders(status))
    return ToolResult(True, "orders", orders, "Fetched recent orders")


def _buy_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    symbol = str(args.get("symbol", "")).strip()
    qty = float(args.get("qty", 0))
    if not symbol:
        raise BrokerError("buy tool requires 'symbol'")
    if qty <= 0:
        raise BrokerError("buy tool requires a positive 'qty'")
    order = broker.submit_market_order(symbol, qty, "buy")
    return ToolResult(True, "buy", order, f"Submitted BUY order for {symbol}")


def _sell_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    symbol = str(args.get("symbol", "")).strip()
    qty = float(args.get("qty", 0))
    if not symbol:
        raise BrokerError("sell tool requires 'symbol'")
    if qty <= 0:
        raise BrokerError("sell tool requires a positive 'qty'")
    order = broker.submit_market_order(symbol, qty, "sell")
    return ToolResult(True, "sell", order, f"Submitted SELL order for {symbol}")


def _cancel_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    order_id = str(args.get("order_id", "")).strip()
    if not order_id:
        raise BrokerError("cancel tool requires 'order_id'")
    broker.cancel_order(order_id)
    return ToolResult(True, "cancel", {"order_id": order_id}, f"Canceled order {order_id}")



def _search_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    query = str(args.get("query", "")).strip()
    if not query:
        raise BrokerError("search tool requires 'query'")

    endpoint = get_searxng_endpoint()
    params = {
        "q": query,
        "format": "json",
    }

    categories = args.get("categories")
    if categories:
        if isinstance(categories, str):
            params["categories"] = categories
        else:
            params["categories"] = ",".join(str(item) for item in categories)
    else:
        default_cats = get_searxng_categories()
        if default_cats:
            params["categories"] = ",".join(default_cats)

    if args.get("engines"):
        params["engines"] = args["engines"]
    if args.get("language"):
        params["language"] = args["language"]
    if args.get("safesearch") is not None:
        params["safesearch"] = args["safesearch"]

    max_results = int(args.get("max_results", 5))
    max_results = max(1, min(max_results, 10))
    params["max_results"] = max_results

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/117.0",
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1",
        "Accept": "application/json",
        "Referer": "http://localhost",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise BrokerError(f"SearxNG request failed: {exc}") from exc

    if response.status_code != 200:
        raise BrokerError(f"SearxNG returned status {response.status_code}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise BrokerError("Failed to decode SearxNG response") from exc

    results = [
        {
            "title": item.get("title"),
            "url": item.get("url") or item.get("href"),
            "snippet": item.get("content") or item.get("snippet") or item.get("body"),
            "source": item.get("source"),
        }
        for item in payload.get("results", [])
    ]

    message = f"Top {len(results)} results for '{query}'" if results else "No results found"
    return ToolResult(True, "search", results[:max_results], message)


def _fetch_url_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    url = str(args.get("url", "")).strip()
    if not url:
        raise BrokerError("fetch_url tool requires 'url'")

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:117.0) Gecko/20100101 Firefox/117.0",
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "127.0.0.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "http://localhost",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as exc:
        raise BrokerError(f"Failed to fetch URL: {exc}") from exc

    if response.status_code != 200:
        raise BrokerError(f"Fetching URL returned status {response.status_code}")

    content_type = response.headers.get("Content-Type", "")
    if "text" not in content_type and "html" not in content_type:
        raise BrokerError("URL did not return HTML content")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    raw_text = soup.get_text(separator="
")
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    text_content = "
".join(lines)
    max_length = int(args.get("max_chars", 8000))
    truncated = False
    if len(text_content) > max_length:
        text_content = text_content[:max_length].rstrip() + "…"
        truncated = True

    data = {
        "url": url,
        "title": title,
        "text": text_content,
        "truncated": truncated,
    }
    message = f"Fetched content from {url}" + (" (truncated)" if truncated else "")
    return ToolResult(True, "fetch_url", data, message)


def _quote_tool(broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    symbol = str(args.get("symbol", "")).strip().upper()
    if not symbol:
        raise BrokerError("quote tool requires 'symbol'")
    quote = broker.get_latest_quote(symbol)
    return ToolResult(True, "quote", {"symbol": symbol, "quote": quote}, f"Fetched quote for {symbol}")


# ---------------------------------------------------------------------------
# Registry

AVAILABLE_TOOLS: Dict[str, Tool] = {
    "account": Tool(
        name="account",
        description="Retrieve the current account status and balances",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_account_tool,
    ),
    "positions": Tool(
        name="positions",
        description="List currently open positions",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=_positions_tool,
    ),
    "orders": Tool(
        name="orders",
        description="List recent orders; optional status filter (open/closed/all)",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter orders by status",
                }
            },
        },
        handler=_orders_tool,
    ),
    "buy": Tool(
        name="buy",
        description="Submit a market BUY order",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "qty": {"type": "number", "minimum": 0.0001},
            },
            "required": ["symbol", "qty"],
        },
        handler=_buy_tool,
    ),
    "sell": Tool(
        name="sell",
        description="Submit a market SELL order",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "qty": {"type": "number", "minimum": 0.0001},
            },
            "required": ["symbol", "qty"],
        },
        handler=_sell_tool,
    ),
    "cancel": Tool(
        name="cancel",
        description="Cancel an existing order by id",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        handler=_cancel_tool,
    ),
    "search": Tool(
        name="search",
        description="Search the web via SearxNG",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "categories": {"type": "string", "description": "Comma-separated categories"},
                "engines": {"type": "string", "description": "Comma-separated engines"},
                "language": {"type": "string"},
                "safesearch": {"type": "string", "description": "0/1/2 or off/moderate/strict"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        handler=_search_tool,
    ),
    "quote": Tool(
        name="quote",
        description="Fetch the latest market quote for a symbol",
        parameters={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        handler=_quote_tool,
    ),
}


def iter_tool_specs() -> Iterable[Dict[str, Any]]:
    for tool in AVAILABLE_TOOLS.values():
        yield tool.to_llm_spec()


def run_tool(name: str, broker: AlpacaBroker, args: Mapping[str, Any]) -> ToolResult:
    tool = AVAILABLE_TOOLS.get(name)
    if not tool:
        raise BrokerError(f"Unknown tool '{name}'")
    return tool.run(broker, args)
