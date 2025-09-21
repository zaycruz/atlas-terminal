"""Interactive terminal experience for the Atlas CLI."""
from __future__ import annotations

import atexit
import argparse
import cmd
import readline
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.console import Console
from rich.table import Table
from termcolor import colored

from .brokers import Account, AlpacaBroker, Order, Position
from .brokers.base import BrokerError
from .environment import APP_DIR, get_ai_model, get_ai_system_prompt, get_ollama_host

HISTORY_FILE = APP_DIR / "history.txt"


# ---------------------------------------------------------------------------
# Shared render helpers (also used by the CLI commands)

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
        console.print(colored("No open positions.", "yellow"))
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
        console.print(colored("No orders found.", "yellow"))
    else:
        console.print(table)




def render_option_chain(console: Console, chain: dict[str, Any]) -> None:
    """Pretty-print a small option chain snapshot."""
    title = f"Options {chain['symbol']} {chain['expiration']}"
    if chain.get("requested_expiration") and chain["requested_expiration"] and chain["requested_expiration"] != chain["expiration"]:
        console.print(colored(
            f"Requested expiration {chain['requested_expiration']} unavailable; showing {chain['expiration']}",
            "yellow",
        ))
    if chain.get("underlying_price"):
        console.print(f"Underlying: {chain['underlying_price']:.2f}")
    if chain.get("available_expirations"):
        console.print("Expirations: " + ", ".join(chain['available_expirations'][:8]))

    table = Table(title=title)
    table.add_column("Strike", justify="right")
    table.add_column("Call Bid", justify="right")
    table.add_column("Call Ask", justify="right")
    table.add_column("Call Last", justify="right")
    table.add_column("Call IV", justify="right")
    table.add_column("Put Bid", justify="right")
    table.add_column("Put Ask", justify="right")
    table.add_column("Put Last", justify="right")
    table.add_column("Put IV", justify="right")

    for row in chain["rows"]:
        call = row.get("call") or {}
        put = row.get("put") or {}

        def fmt(value: Optional[float], suffix: str = "") -> str:
            if value is None:
                return "-"
            if suffix:
                return f"{value:{suffix}}"
            return f"{value:.2f}"

        def fmt_iv(value: Optional[float]) -> str:
            return fmt(value * 100 if value is not None else None, ".1f") + "%" if value is not None else "-"

        table.add_row(
            f"{row['strike']:.2f}",
            fmt(call.get("bid")),
            fmt(call.get("ask")),
            fmt(call.get("last")),
            fmt_iv(call.get("iv")),
            fmt(put.get("bid")),
            fmt(put.get("ask")),
            fmt(put.get("last")),
            fmt_iv(put.get("iv")),
        )

    console.print(table)
def render_quote(console: Console, symbol: str, quote: dict[str, object]) -> None:
    table = Table(title=f"Quote {symbol.upper()}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in quote.items():
        table.add_row(key, str(value))
    console.print(table)


# ---------------------------------------------------------------------------
# Terminal implementation

class AtlasTerminal(cmd.Cmd):
    """REPL for interacting with an Alpaca trading account."""

    def __init__(self, broker: AlpacaBroker, environment: str, console: Optional[Console] = None) -> None:
        super().__init__()
        self._broker = broker
        self._env = environment
        self.console = console or Console()
        self.prompt = colored("atlas: ", "green")
        self.intro = self._create_welcome_banner()
        self.history_file = HISTORY_FILE
        self._load_history()
        atexit.register(self._save_history)

    # ------------------------------------------------------------------
    # Setup helpers
    def _load_history(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass

    def _save_history(self) -> None:
        try:
            readline.write_history_file(self.history_file)
        except FileNotFoundError:
            pass

    def _create_welcome_banner(self) -> str:
        banner_lines = [
            "    ╭─────────────────────────────────────────────────────────╮",
            "    │                                                         │",
            "    │     █████╗ ████████╗██╗      █████╗ ███████╗            │",
            "    │    ██╔══██╗╚══██╔══╝██║     ██╔══██╗██╔════╝            │",
            "    │    ███████║   ██║   ██║     ███████║███████╗            │",
            "    │    ██╔══██║   ██║   ██║     ██╔══██║╚════██║            │",
            "    │    ██║  ██║   ██║   ███████╗██║  ██║███████║            │",
            "    │    ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝            │",
            "    │                                                         │",
            "    │           🚀 Advanced Trading Terminal 🚀               │",
            "    ╰─────────────────────────────────────────────────────────╯",
        ]
        colored_lines = [colored(line, "cyan") for line in banner_lines]
        tips = [
            "",
            colored("Tips:", "yellow"),
            colored("1. Type 'help' to list commands.", "white"),
            colored("2. Use 'account', 'positions', 'orders' to explore.", "white"),
            colored("3. 'buy' and 'sell' submit market orders.", "white"),
            colored("4. 'quote' fetches the latest bid/ask.", "white"),
            colored("5. 'env' shows whether you are in paper or live mode.", "white"),
            "",
        ]
        return "\n".join(colored_lines + tips)

    def _handle_error(self, exc: Exception) -> None:
        message = str(exc) or exc.__class__.__name__
        self.console.print(colored(message, "red"))

    # ------------------------------------------------------------------
    # Built-in hooks
    def preloop(self) -> None:  # type: ignore[override]
        if self.intro:
            print(self.intro)

    def emptyline(self) -> None:  # type: ignore[override]
        # Avoid repeating the previous command automatically
        pass

    def default(self, line: str) -> None:  # type: ignore[override]
        self.console.print(colored(f"Unknown command: {line}", "yellow"))

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
            self.console.print(colored("Usage: buy SYMBOL QTY", "yellow"))
            return

        try:
            quantity = float(qty)
        except ValueError:
            self.console.print(colored("Quantity must be numeric (e.g. 1 or 0.5).", "yellow"))
            return

        try:
            order = self._broker.submit_market_order(symbol, quantity, side)
            summary = {
                k: v
                for k, v in asdict(order).items()
                if k in {"id", "symbol", "qty", "side", "status"}
            }
            self.console.print(colored(f"Submitted {side.upper()} order", "green"))
            self.console.print(summary)
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
            self.console.print(colored("Usage: cancel ORDER_ID", "yellow"))
            return
        try:
            self._broker.cancel_order(order_id)
            self.console.print(colored(f"Canceled order {order_id}", "green"))
        except Exception as exc:
            self._handle_error(exc)

    def do_options(self, arg: str) -> None:
        """options SYMBOL [--expiration YYYY-MM-DD] [--width N] [--type call|put]"""
        parser = argparse.ArgumentParser(prog="options", add_help=False)
        parser.add_argument("symbol")
        parser.add_argument("--expiration")
        parser.add_argument("--width", type=int, default=5)
        parser.add_argument("--type", choices=["call", "put"])
        try:
            args = parser.parse_args(shlex.split(arg))
        except SystemExit:
            self.console.print("Usage: options SYMBOL [--expiration YYYY-MM-DD] [--width N] [--type call|put]")
            return
        try:
            chain = self._broker.get_option_chain(
                args.symbol,
                expiration=args.expiration,
                strikes=args.width,
                option_type=args.type,
            )
            render_option_chain(self.console, chain)
        except Exception as exc:
            self._handle_error(exc)

    def do_quote(self, arg: str) -> None:
        """quote SYMBOL -- fetch the latest quote."""
        symbol = arg.strip().upper()
        if not symbol:
            self.console.print(colored("Usage: quote SYMBOL", "yellow"))
            return
        try:
            quote = self._broker.get_latest_quote(symbol)
            render_quote(self.console, symbol, quote)
        except Exception as exc:
            self._handle_error(exc)

    def do_ai(self, arg: str) -> None:
        """ai [MODEL] -- enter AI chat mode (optional model override)."""
        model_override = arg.strip() or None
        from .ai import AIChatConfig, run_chat  # local import to avoid circular dependency

        config = AIChatConfig(
            host=get_ollama_host(),
            model=get_ai_model(model_override),
            system_prompt=get_ai_system_prompt(),
            environment=self._env,
        )
        run_chat(self._broker, config, console=self.console)

    def do_optionorder(self, arg: str) -> None:
        """optionorder SYMBOL QTY --side buy|sell [--intent ...] [--type market|limit] [--limit PRICE] [--tif day]"""
        parser = argparse.ArgumentParser(prog="optionorder", add_help=False)
        parser.add_argument("symbol")
        parser.add_argument("qty", type=float)
        parser.add_argument("--side", choices=["buy", "sell"], required=True)
        parser.add_argument("--intent", choices=["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"])
        parser.add_argument("--type", choices=["market", "limit"], default="market")
        parser.add_argument("--limit", type=float)
        parser.add_argument("--tif", default="day")
        try:
            args = parser.parse_args(shlex.split(arg))
        except SystemExit:
            self.console.print("Usage: optionorder SYMBOL QTY --side buy|sell [--intent ...] [--type market|limit] [--limit PRICE] [--tif day]")
            return
        try:
            order = self._broker.submit_option_order(
                option_symbol=args.symbol,
                qty=args.qty,
                side=args.side,
                intent=args.intent,
                order_type=args.type,
                limit_price=args.limit,
                time_in_force=args.tif,
            )
            self.console.print(colored(
                f"Submitted option order {order.symbol} qty={order.qty} side={order.side} status={order.status} id={order.id}",
                "green",
            ))
        except Exception as exc:
            self._handle_error(exc)

    def do_env(self, _: str) -> None:
        """Show the current trading environment."""
        self.console.print(colored(f"Environment: {self._env}", "cyan"))

    def do_quit(self, _: str) -> bool:  # type: ignore[override]
        """Exit the terminal."""
        self.console.print(colored("Bye.", "cyan"))
        return True

    def do_exit(self, arg: str) -> bool:  # type: ignore[override]
        return self.do_quit(arg)


def run_terminal(broker: AlpacaBroker, environment: str, console: Optional[Console] = None) -> None:
    """Launch the Atlas interactive terminal."""
    AtlasTerminal(broker, environment, console=console).cmdloop()
