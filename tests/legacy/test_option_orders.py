import unittest
from unittest.mock import MagicMock

from atlas.brokers.alpaca import AlpacaBroker
from atlas.brokers.base import BrokerError
from atlas.brokers.models import Order
from alpaca.trading.enums import OrderType, OrderSide, PositionIntent


class StubAlpacaBroker(AlpacaBroker):
    def __init__(self):
        pass

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def get_latest_quote(self, symbol: str):  # pragma: no cover - stub
        return {}


class DummyOrder:
    def __init__(self):
        self.id = "id"
        self.symbol = "AAPL240920C00200000"
        self.side = OrderSide.BUY
        self.qty = 1
        self.type = OrderType.MARKET
        self.status = "accepted"
        self.submitted_at = None
        self.filled_qty = None
        self.filled_avg_price = None


class OptionOrderTests(unittest.TestCase):
    def _make_broker(self):
        broker = StubAlpacaBroker()
        broker._trading = MagicMock()
        broker._map_order = MagicMock(return_value=Order(
            id="id",
            symbol="OPT",
            qty=1,
            side="buy",
            type="market",
            status="accepted",
            submitted_at=None,
            filled_qty=None,
            filled_avg_price=None,
        ))
        return broker

    def test_market_order_builds_request(self):
        broker = self._make_broker()
        broker._trading.submit_order.return_value = DummyOrder()

        order = broker.submit_option_order(
            option_symbol="AAPL240920C00200000",
            qty=2,
            side="buy",
            intent="buy_to_open",
        )

        request = broker._trading.submit_order.call_args.args[0]
        self.assertEqual(request.type, OrderType.MARKET)
        self.assertEqual(request.symbol, "AAPL240920C00200000")
        self.assertEqual(request.qty, 2.0)
        self.assertEqual(request.side, OrderSide.BUY)
        self.assertEqual(request.position_intent, PositionIntent.BUY_TO_OPEN)
        self.assertEqual(order.symbol, "OPT")

    def test_limit_requires_price(self):
        broker = self._make_broker()
        with self.assertRaises(BrokerError):
            broker.submit_option_order(
                option_symbol="AAPL240920C00200000",
                qty=1,
                side="buy",
                order_type="limit",
            )

    def test_invalid_side_raises(self):
        broker = self._make_broker()
        with self.assertRaises(BrokerError):
            broker.submit_option_order("AAPL240920C00200000", 1, "hold")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
