"""Interactive terminal experience for the Atlas CLI."""
from __future__ import annotations

import cmd
import shlex
from dataclasses import asdict
from typing import Iterable, Optional

from rich.console import Console
from rich.table import Table

from .brokers import Account, AlpacaBroker, Order, Position
from .brokers.base import BrokerError


def render_account(console: Console, environment: str, account: Account) -> None:
    table = Table(title=f"Account ({environment})")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("id", account.id)
    table.add_row("status", account.status)
    table.add_row("equity", f"{account.equity:,.2f}")
    table.add_row("buying_power", f"{account.buying_power:,.2f}")
    table.add_row("cash", f"{account.cash:,.2f}")
    table.add_row("pattern_day_trader", str(account.pattern_day_trader))
    console.print(table)


def render_positions(console: Console, positions: Iterable[Position]) -> None:
    table = Table(title="Open Positions")
    table.add_column("Symbol")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Entry", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("Unrealized P/L", justify="right")
    table.add_column("Unrealized %", justify="right")

    count = 0
    for pos in positions:
        count += 1
        table.add_row(
            pos.symbol,
            f"{pos.qty}",
            f"{pos.avg_entry_price:,.2f}",
            f"{pos.current_price:,.2f}",
            f"{pos.market_value:,.2f}",
            f"{pos.unrealized_pl:,.2f}",
            f"{pos.unrealized_plpc:.2%}",
        )

    if count == 0:
        console.print("No open positions.")
    else:
        console.print(table)


def render_orders(console: Console, orders: Iterable[Order], status: Optional[str]) -> None:
    scoped = f" ({status})" if status else ""
    table = Table(title=f"Orders{scoped}")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Filled Qty")
    table.add_column("Filled Avg")
    table.add_column("Submitted")

    count = 0
    for order in orders:
        count += 1
        table.add_row(
            order.id,
            order.symbol,
            order.side,
            f"{order.qty}",
            order.type,
            order.status,
            f"{order.filled_qty}" if order.filled_qty is not None else "-",
            f"{order.filled_avg_price}" if order.filled_avg_price is not None else "-",
            str(order.submitted_at or "-"),
        )

    if count == 0:
        console.print("No orders found.")
    else:
        console.print(table)


def render_quote(console: Console, symbol: str, quote: dict[str, object]) -> None:
    table = Table(title=f"Quote {symbol.upper()}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in quote.items():
        table.add_row(key, str(value))
    console.print(table)


class AtlasTerminal(cmd.Cmd):
    """A small REPL for interacting with an Alpaca trading account."""

    intro = "Welcome to Atlas. Type 'help' to list commands, 'quit' to exit."
    prompt = "atlas> "

    def __init__(self, broker: AlpacaBroker, environment: str, console: Optional[Console] = None) -> None:
        super().__init__()
        self._broker = broker
        self._env = environment
        self.console = console or Console()

    def _handle_error(self, exc: Exception) -> None:
        message = str(exc)
        if not message:
            message = exc.__class__.__name__
        self.console.print(f"[red]{message}[/red]")

    # ------------------------------------------------------------------
    # Commands
    def do_account(self, _: str) -> None:
        """Show account status and buying power."""
        try:
            account = self._broker.get_account()
            render_account(self.console, self._env, account)
        except Exception as exc:
            self._handle_error(exc)

    def do_positions(self, _: str) -> None:
        """List open positions."""
        try:
            positions = self._broker.get_positions()
            render_positions(self.console, positions)
        except Exception as exc:
            self._handle_error(exc)

    def do_orders(self, arg: str) -> None:
        """orders [open|closed|all] -- list orders (default: open)."""
        status = None
        if arg.strip():
            tokens = shlex.split(arg)
            if tokens:
                status = tokens[0]
        try:
            orders = self._broker.get_orders(status)
            render_orders(self.console, orders, status)
        except Exception as exc:
            self._handle_error(exc)

    def _place_order(self, side: str, arg: str) -> None:
        try:
            symbol, qty = shlex.split(arg)
        except ValueError:
            self.console.print("Usage: buy SYMBOL QTY")
            return

        try:
            quantity = float(qty)
        except ValueError:
            self.console.print("Quantity must be numeric (e.g. 1 or 0.5).")
            return

        try:
            order = self._broker.submit_market_order(symbol, quantity, side)
            summary = {
                k: v
                for k, v in asdict(order).items()
                if k in {"id", "symbol", "qty", "side", "status"}
            }
            self.console.print(f"[green]Submitted {side.upper()} order:[/green] {summary}")
        except Exception as exc:
            self._handle_error(exc)

    def do_buy(self, arg: str) -> None:
        """buy SYMBOL QTY -- submit a market buy order."""
        self._place_order("buy", arg)

    def do_sell(self, arg: str) -> None:
        """sell SYMBOL QTY -- submit a market sell order."""
        self._place_order("sell", arg)

    def do_cancel(self, arg: str) -> None:
        """cancel ORDER_ID -- cancel an existing order."""
        order_id = arg.strip()
        if not order_id:
            self.console.print("Usage: cancel ORDER_ID")
            return
        try:
            self._broker.cancel_order(order_id)
            self.console.print(f"[green]Canceled order {order_id}.[/green]")
        except Exception as exc:
            self._handle_error(exc)

    def do_quote(self, arg: str) -> None:
        """quote SYMBOL -- fetch the latest quote."""
        symbol = arg.strip().upper()
        if not symbol:
            self.console.print("Usage: quote SYMBOL")
            return
        try:
            quote = self._broker.get_latest_quote(symbol)
            render_quote(self.console, symbol, quote)
        except Exception as exc:
            self._handle_error(exc)

    def do_env(self, _: str) -> None:
        """Show the current trading environment."""
        self.console.print(f"Environment: {self._env}")

    def do_quit(self, _: str) -> bool:  # type: ignore[override]
        """Exit the terminal."""
        self.console.print("Bye.")
        return True

    def do_exit(self, arg: str) -> bool:  # type: ignore[override]
        return self.do_quit(arg)

    def emptyline(self) -> None:  # type: ignore[override]
        # Avoid repeating the last command
        pass


def run_terminal(broker: AlpacaBroker, environment: str, console: Optional[Console] = None) -> None:
    """Launch the Atlas interactive terminal."""
    AtlasTerminal(broker, environment, console=console).cmdloop()
