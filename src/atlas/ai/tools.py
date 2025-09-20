"""Tool registry exposed to the Atlas AI assistant."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, Mapping

from ..brokers import Account, AlpacaBroker, Order, Position
from ..brokers.base import BrokerError

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
