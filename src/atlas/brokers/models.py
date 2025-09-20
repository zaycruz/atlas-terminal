"""Shared broker data models used by the Atlas terminal."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Account:
    id: str
    status: str
    equity: float
    buying_power: float
    cash: float
    pattern_day_trader: bool
    created_at: Optional[datetime] = None


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass
class Order:
    id: str
    symbol: str
    qty: float
    side: str
    type: str
    status: str
    submitted_at: Optional[datetime]
    filled_qty: Optional[float]
    filled_avg_price: Optional[float]
