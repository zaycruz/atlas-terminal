"""Alpaca broker implementation used by the Atlas terminal."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Optional

from alpaca.data import OptionsFeed, OptionChainRequest
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, PositionIntent, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from .base import Broker, BrokerError
from .models import Account, Order, Position


def _to_float(value: Optional[object]) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise BrokerError(f"Could not convert value '{value}' to float") from None


def _parse_option_symbol(symbol: str) -> tuple[str, str, float]:
    """Return (expiration YYYY-MM-DD, type ('call'|'put'), strike)."""
    if len(symbol) < 15:
        raise BrokerError(f"Unexpected option symbol format: {symbol}")
    tail = symbol[-15:]
    date_code = tail[:6]
    option_type = tail[6]
    strike_code = tail[7:]
    try:
        expiration = f"20{date_code[:2]}-{date_code[2:4]}-{date_code[4:]}"
        strike = int(strike_code) / 1000.0
    except ValueError as exc:
        raise BrokerError(f"Failed to parse option symbol {symbol}") from exc
    leg_type = "call" if option_type.upper() == "C" else "put"
    return expiration, leg_type, strike


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    paper: bool = True


class AlpacaBroker(Broker):
    """Thin wrapper around alpaca-py providing Atlas-friendly models."""

    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config
        self._trading = TradingClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
            paper=config.paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.secret_key,
        )
        self._options = OptionHistoricalDataClient(
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


    def submit_option_order(
        self,
        option_symbol: str,
        qty: float,
        side: str,
        *,
        intent: str | None = None,
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> Order:
        if qty <= 0:
            raise BrokerError("Quantity must be positive")

        side_lower = side.lower()
        if side_lower == "buy":
            order_side = OrderSide.BUY
        elif side_lower == "sell":
            order_side = OrderSide.SELL
        else:
            raise BrokerError("Side must be 'buy' or 'sell'")

        tif_lookup = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "opg": TimeInForce.OPG,
            "cls": TimeInForce.CLS,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        tif_value = tif_lookup.get(time_in_force.lower())
        if tif_value is None:
            raise BrokerError("Unsupported time in force")

        intent_lookup = {
            "buy_to_open": PositionIntent.BUY_TO_OPEN,
            "buy_to_close": PositionIntent.BUY_TO_CLOSE,
            "sell_to_open": PositionIntent.SELL_TO_OPEN,
            "sell_to_close": PositionIntent.SELL_TO_CLOSE,
        }
        if intent:
            intent_value = intent_lookup.get(intent.lower())
            if not intent_value:
                raise BrokerError("Invalid intent. Use buy_to_open/buy_to_close/sell_to_open/sell_to_close")
        else:
            intent_value = PositionIntent.BUY_TO_OPEN if order_side == OrderSide.BUY else PositionIntent.SELL_TO_OPEN

        request_type = order_type.lower()
        if request_type == "market":
            req = MarketOrderRequest(
                symbol=option_symbol.upper(),
                qty=str(qty),
                side=order_side,
                type=OrderType.MARKET,
                time_in_force=tif_value,
                position_intent=intent_value,
            )
        elif request_type == "limit":
            if limit_price is None:
                raise BrokerError("limit_price is required for limit orders")
            req = LimitOrderRequest(
                symbol=option_symbol.upper(),
                qty=str(qty),
                side=order_side,
                type=OrderType.LIMIT,
                time_in_force=tif_value,
                limit_price=limit_price,
                position_intent=intent_value,
            )
        else:
            raise BrokerError("Unsupported order type. Use market or limit.")

        try:
            raw_order = self._trading.submit_order(req)
        except Exception as exc:  # pragma: no cover - API errors
            raise BrokerError(f"Failed to submit option order: {exc}") from exc

        return self._map_order(raw_order)
    
    def cancel_order(self, order_id: str) -> None:
        try:
            self._trading.cancel_order_by_id(order_id)
        except Exception as exc:  # pragma: no cover - thin wrapper
            raise BrokerError(f"Failed to cancel order {order_id}: {exc}") from exc

    def get_option_chain(
        self,
        symbol: str,
        *,
        expiration: str | None = None,
        strikes: int = 5,
        option_type: str | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        quote = self.get_latest_quote(symbol)
        bid = quote.get("bid")
        ask = quote.get("ask")
        mid = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
        elif ask is not None:
            mid = ask
        elif bid is not None:
            mid = bid
        underlying_price = mid

        snapshots = {}
        last_error: Exception | None = None
        window = timedelta(days=3)
        today = datetime.utcnow().date()
        default_lte = today + timedelta(days=90)

        for feed in (OptionsFeed.INDICATIVE, OptionsFeed.OPRA):
            attempts: list[tuple[Optional[date], Optional[date]]] = []
            if expiration:
                try:
                    target_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                except ValueError as exc:
                    raise BrokerError("Expiration must be YYYY-MM-DD") from exc
                attempts.append((target_date - window, target_date + window))
                attempts.append((None, None))
            else:
                attempts.append((today, default_lte))
                attempts.append((None, None))

            for gte, lte in attempts:
                request = OptionChainRequest(underlying_symbol=symbol, feed=feed)
                if gte and lte:
                    request.expiration_date_gte = gte
                    request.expiration_date_lte = lte
                elif lte and not gte:
                    request.expiration_date_lte = lte
                try:
                    raw_chain = self._options.get_option_chain(request)
                    snapshots = dict(raw_chain)
                    if snapshots:
                        break
                except Exception as exc:  # pragma: no cover - network/API
                    last_error = exc
            if snapshots:
                break

        if not snapshots:
            if last_error:
                raise BrokerError(f"Failed to fetch option chain: {last_error}") from last_error
            raise BrokerError("No option data returned from server")

        entries: list[dict[str, Any]] = []
        for contract_symbol, snapshot in snapshots.items():
            exp, leg_type, strike = _parse_option_symbol(contract_symbol)

            quote_data = snapshot.latest_quote
            trade_data = snapshot.latest_trade
            greeks = snapshot.greeks

            def _safe(value: Optional[object]) -> Optional[float]:
                return _to_float(value) if value is not None else None

            entries.append(
                {
                    "symbol": contract_symbol,
                    "expiration": exp,
                    "type": leg_type,
                    "strike": strike,
                    "bid": _safe(getattr(quote_data, "bid_price", None)),
                    "ask": _safe(getattr(quote_data, "ask_price", None)),
                    "last": _safe(getattr(trade_data, "price", None)),
                    "iv": _safe(getattr(snapshot, "implied_volatility", None)),
                    "delta": _safe(getattr(greeks, "delta", None)) if greeks else None,
                    "gamma": _safe(getattr(greeks, "gamma", None)) if greeks else None,
                    "theta": _safe(getattr(greeks, "theta", None)) if greeks else None,
                    "vega": _safe(getattr(greeks, "vega", None)) if greeks else None,
                }
            )

        if not entries:
            raise BrokerError("No option contracts returned for symbol")

        expirations = sorted(
            {e["expiration"] for e in entries},
            key=lambda d: datetime.strptime(d, "%Y-%m-%d"),
        )
        if expiration and expiration not in expirations:
            selected_exp = min(
                expirations,
                key=lambda d: abs(
                    datetime.strptime(d, "%Y-%m-%d")
                    - datetime.strptime(expiration, "%Y-%m-%d")
                ),
            )
        else:
            selected_exp = expiration or expirations[0]

        filtered = [e for e in entries if e["expiration"] == selected_exp]
        if not filtered:
            raise BrokerError("Unable to find options for requested expiration")

        strikes = max(1, strikes)
        strikes_set = sorted({e["strike"] for e in filtered})
        if not strikes_set:
            raise BrokerError("No strikes available for selected expiration")

        if underlying_price is None:
            underlying_price = strikes_set[len(strikes_set) // 2]

        nearest_idx = min(range(len(strikes_set)), key=lambda i: abs(strikes_set[i] - underlying_price))
        start = max(0, nearest_idx - strikes)
        end = min(len(strikes_set) - 1, nearest_idx + strikes)
        selected_strikes = strikes_set[start : end + 1]

        rows = []
        for strike in selected_strikes:
            call = next((e for e in filtered if e["strike"] == strike and e["type"] == "call"), None)
            put = next((e for e in filtered if e["strike"] == strike and e["type"] == "put"), None)

            if option_type == "call":
                put = None
            elif option_type == "put":
                call = None

            rows.append({"strike": strike, "call": call, "put": put})

        return {
            "symbol": symbol,
            "expiration": selected_exp,
            "requested_expiration": expiration,
            "underlying_price": underlying_price,
            "available_expirations": expirations,
            "rows": rows,
        }

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
