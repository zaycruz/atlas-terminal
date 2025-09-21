"""Command-line entry point for the Atlas Alpaca terminal."""
from __future__ import annotations

import argparse
from typing import Callable, Optional

from rich.console import Console
from termcolor import colored

from .brokers import AlpacaBroker, AlpacaConfig
from .brokers.base import BrokerError
from .environment import (
    get_alpaca_credentials,
    get_ai_model,
    get_ai_system_prompt,
    get_ollama_host,
    load_dotenv,
    resolve_environment,
)
from .terminal import (
    render_account,
    render_orders,
    render_positions,
    render_quote,
    render_option_chain,
    run_terminal,
)
from .ai import AIChatConfig, run_chat

ConsoleAction = Callable[[AlpacaBroker, str, argparse.Namespace], None]

console = Console()


def create_broker(environment: str) -> AlpacaBroker:
    api_key, secret_key = get_alpaca_credentials()
    config = AlpacaConfig(
        api_key=api_key,
        secret_key=secret_key,
        paper=(environment == "paper"),
    )
    return AlpacaBroker(config)


def run_action(args: argparse.Namespace, action: ConsoleAction) -> int:
    env = resolve_environment(getattr(args, "env", None))
    try:
        broker = create_broker(env)
    except Exception as exc:
        message = str(exc) or exc.__class__.__name__
        console.print(colored(message, "red"))
        return 2

    try:
        action(broker, env, args)
        return 0
    except (BrokerError, ValueError) as exc:
        message = str(exc) or exc.__class__.__name__
        console.print(colored(message, "red"))
        return 1


# ----------------------------------------------------------------------
# Command handlers

def handle_account(broker: AlpacaBroker, env: str, _: argparse.Namespace) -> None:
    account = broker.get_account()
    render_account(console, env, account)


def handle_positions(broker: AlpacaBroker, _: str, __: argparse.Namespace) -> None:
    positions = broker.get_positions()
    render_positions(console, positions)


def handle_orders(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    orders = broker.get_orders(args.status)
    render_orders(console, orders, args.status)


def _parse_quantity(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("Quantity must be numeric (e.g. 1 or 0.5).") from exc


def handle_buy(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    qty = _parse_quantity(args.qty)
    order = broker.submit_market_order(args.symbol, qty, "buy")
    console.print(colored(
        f"Submitted BUY order {order.symbol} qty={order.qty} status={order.status} id={order.id}",
        "green",
    ))


def handle_sell(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    qty = _parse_quantity(args.qty)
    order = broker.submit_market_order(args.symbol, qty, "sell")
    console.print(colored(
        f"Submitted SELL order {order.symbol} qty={order.qty} status={order.status} id={order.id}",
        "green",
    ))


def handle_cancel(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    broker.cancel_order(args.order_id)
    console.print(colored(f"Canceled order {args.order_id}", "green"))


def handle_quote(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    quote = broker.get_latest_quote(args.symbol)
    render_quote(console, args.symbol, quote)



def handle_option_order(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    order = broker.submit_option_order(
        option_symbol=args.symbol,
        qty=args.qty,
        side=args.side,
        intent=args.intent,
        order_type=args.type,
        limit_price=args.limit,
        time_in_force=args.tif,
    )
    console.print(colored(
        f"Submitted option order {order.symbol} qty={order.qty} side={order.side} status={order.status} id={order.id}",
        "green",
    ))

def handle_options(broker: AlpacaBroker, _: str, args: argparse.Namespace) -> None:
    chain = broker.get_option_chain(
        args.symbol,
        expiration=args.expiration,
        strikes=args.width,
        option_type=args.type,
    )
    render_option_chain(console, chain)



def handle_terminal(broker: AlpacaBroker, env: str, _: argparse.Namespace) -> None:
    run_terminal(broker, env, console=console)


def handle_ai(broker: AlpacaBroker, env: str, args: argparse.Namespace) -> None:
    model = get_ai_model(getattr(args, "model", None))
    config = AIChatConfig(
        host=get_ollama_host(),
        model=model,
        system_prompt=get_ai_system_prompt(),
        environment=env,
    )
    run_chat(broker, config, console=console)


# ----------------------------------------------------------------------
# CLI wiring

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas terminal for Alpaca paper/live trading",
    )
    parser.add_argument(
        "--env",
        choices=["paper", "live"],
        help="Target environment (defaults to paper or $ATLAS_ENV)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("account", help="Show account overview").set_defaults(func=handle_account)
    sub.add_parser("positions", help="List open positions").set_defaults(func=handle_positions)

    orders_parser = sub.add_parser("orders", help="List recent orders")
    orders_parser.add_argument("--status", choices=["open", "closed", "all"], default="open")
    orders_parser.set_defaults(func=handle_orders)

    buy_parser = sub.add_parser("buy", help="Submit a market BUY order")
    buy_parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    buy_parser.add_argument("qty", help="Quantity (supports fractional)")
    buy_parser.set_defaults(func=handle_buy)

    sell_parser = sub.add_parser("sell", help="Submit a market SELL order")
    sell_parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    sell_parser.add_argument("qty", help="Quantity (supports fractional)")
    sell_parser.set_defaults(func=handle_sell)

    cancel_parser = sub.add_parser("cancel", help="Cancel an order by id")
    cancel_parser.add_argument("order_id", help="Order identifier")
    cancel_parser.set_defaults(func=handle_cancel)

    quote_parser = sub.add_parser("quote", help="Fetch the latest quote for a symbol")
    quote_parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    quote_parser.set_defaults(func=handle_quote)

    options_parser = sub.add_parser("options", help="View options chain slice")
    options_parser.add_argument("symbol", help="Underlying symbol, e.g. AAPL")
    options_parser.add_argument("--expiration", help="Expiration date YYYY-MM-DD")
    options_parser.add_argument("--width", type=int, default=5, help="Number of strikes each side of ATM")
    options_parser.add_argument("--type", choices=["call", "put"], help="Filter by option type")
    options_parser.set_defaults(func=handle_options)

    option_order_parser = sub.add_parser("option-order", help="Submit a simple option order")
    option_order_parser.add_argument("symbol", help="Option OCC symbol, e.g. AAPL240920C00200000")
    option_order_parser.add_argument("qty", type=float, help="Number of contracts")
    option_order_parser.add_argument("--side", choices=["buy", "sell"], required=True, help="Order side")
    option_order_parser.add_argument("--intent", choices=["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"], help="Position intent")
    option_order_parser.add_argument("--type", choices=["market", "limit"], default="market", help="Order type")
    option_order_parser.add_argument("--limit", type=float, help="Limit price (required for limit orders)")
    option_order_parser.add_argument("--tif", default="day", help="Time in force (day, gtc, opg, cls, ioc, fok)")
    option_order_parser.set_defaults(func=handle_option_order)

    sub.add_parser("terminal", help="Launch the interactive Atlas terminal").set_defaults(func=handle_terminal)

    ai_parser = sub.add_parser("ai", help="Launch Atlas AI chat mode")
    ai_parser.add_argument("--model", help="Override the Ollama model to use")
    ai_parser.set_defaults(func=handle_ai)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    func: ConsoleAction = args.func  # type: ignore[attr-defined]
    return run_action(args, func)


if __name__ == "__main__":  # Allows `python src/atlas/cli.py` for quick tests
    raise SystemExit(main())
