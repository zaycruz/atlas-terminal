"""Alpaca broker implementation used by the Atlas terminal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from .base import Broker, BrokerError
from .models import Account, Order, Position


def _to_float(value: Optional[object]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise BrokerError(f"Could not convert value '{value}' to float") from None


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    paper: bool = True


class AlpacaBroker(Broker):
    """Thin wrapper around alpaca-py providing Atlas-friendly models."""

    def __init__(self, config: AlpacaConfig) -> None:
        self._trading = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )

    # ------------------------------------------------------------------
    # Helpers
    def _map_account(self) -> Account:
        try:
            raw = self._trading.get_account()
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to fetch account: {exc}") from exc

        return Account(
            id=str(raw.id),
            status=str(raw.status),
            equity=_to_float(raw.equity),
            buying_power=_to_float(raw.buying_power),
            cash=_to_float(raw.cash),
            pattern_day_trader=bool(getattr(raw, "pattern_day_trader", False)),
            created_at=getattr(raw, "created_at", None),
        )

    def _map_position(self, raw) -> Position:
        return Position(
            symbol=str(raw.symbol),
            qty=_to_float(raw.qty),
            avg_entry_price=_to_float(raw.avg_entry_price),
            current_price=_to_float(raw.current_price),
            market_value=_to_float(raw.market_value),
            unrealized_pl=_to_float(raw.unrealized_pl),
            unrealized_plpc=_to_float(raw.unrealized_plpc),
        )

    def _map_order(self, raw: AlpacaOrder) -> Order:
        return Order(
            id=str(raw.id),
            symbol=str(raw.symbol),
            qty=_to_float(raw.qty),
            side=str(raw.side),
            type=str(raw.type),
            status=str(raw.status),
            submitted_at=getattr(raw, "submitted_at", None),
            filled_qty=_to_float(getattr(raw, "filled_qty", 0)) if getattr(raw, "filled_qty", None) is not None else None,
            filled_avg_price=_to_float(getattr(raw, "filled_avg_price", 0)) if getattr(raw, "filled_avg_price", None) is not None else None,
        )

    # ------------------------------------------------------------------
    # Broker interface
    def get_account(self) -> Account:
        return self._map_account()

    def get_positions(self) -> Iterable[Position]:
        try:
            raw_positions = self._trading.get_all_positions()
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc
        return [self._map_position(raw) for raw in raw_positions]

    def get_orders(self, status: str | None = None) -> Iterable[Order]:
        request = GetOrdersRequest()
        if status:
            lookup = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            try:
                request.status = lookup[status.lower()]
            except KeyError:
                raise BrokerError("Order status must be one of: open, closed, all")

        try:
            raw_orders = self._trading.get_orders(request)
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to fetch orders: {exc}") from exc
        return [self._map_order(raw) for raw in raw_orders]

    def submit_market_order(self, symbol: str, qty: float, side: str) -> Order:
        side_lower = side.lower()
        if side_lower not in {"buy", "sell"}:
            raise BrokerError("Order side must be 'buy' or 'sell'")

        request = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=str(qty),
            side=OrderSide.BUY if side_lower == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        try:
            raw_order = self._trading.submit_order(request)
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to submit order: {exc}") from exc
        return self._map_order(raw_order)

    def cancel_order(self, order_id: str) -> None:
        try:
            self._trading.cancel_order_by_id(order_id)
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to cancel order {order_id}: {exc}") from exc

    def get_latest_quote(self, symbol: str) -> dict[str, float | None]:
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
        try:
            quote = self._data.get_stock_latest_quote(request)
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to fetch quote for {symbol}: {exc}") from exc

        try:
            data = quote[symbol.upper()]
        except Exception:
            data = quote

        return {
            "bid": _to_float(getattr(data, "bid_price", None)) if getattr(data, "bid_price", None) is not None else None,
            "bid_size": _to_float(getattr(data, "bid_size", None)) if getattr(data, "bid_size", None) is not None else None,
            "ask": _to_float(getattr(data, "ask_price", None)) if getattr(data, "ask_price", None) is not None else None,
            "ask_size": _to_float(getattr(data, "ask_size", None)) if getattr(data, "ask_size", None) is not None else None,
            "timestamp": getattr(data, "timestamp", None),
        }
