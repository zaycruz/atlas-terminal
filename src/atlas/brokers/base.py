"""Abstract broker interface for the Atlas terminal."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .models import Account, Order, Position


class BrokerError(RuntimeError):
    """Raised when broker operations fail."""


class Broker(ABC):
    """Minimal interface the terminal expects from a broker implementation."""

    @abstractmethod
    def get_account(self) -> Account:
        """Return the trading account snapshot."""

    @abstractmethod
    def get_positions(self) -> Iterable[Position]:
        """Return currently open positions."""

    @abstractmethod
    def get_orders(self, status: str | None = None) -> Iterable[Order]:
        """Return recent orders filtered by status if provided."""

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: float, side: str) -> Order:
        """Submit a market order and return the resulting order object."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing order."""

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> dict[str, float | None]:
        """Return a simple dict with bid/ask information for a symbol."""
