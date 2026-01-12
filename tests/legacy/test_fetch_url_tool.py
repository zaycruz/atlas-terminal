import unittest
from unittest.mock import MagicMock, patch

from atlas.ai.tools import _fetch_url_tool
from atlas.brokers.alpaca import AlpacaBroker
from atlas.brokers.base import BrokerError


class DummyBroker(AlpacaBroker):
    def __init__(self):
        pass

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def get_latest_quote(self, symbol: str):  # pragma: no cover - stub
        return {}


class FetchUrlToolTests(unittest.TestCase):
    def test_fetch_success(self):
        broker = DummyBroker()
        html = """
        <html>
          <head><title>Example</title></head>
          <body>
            <h1>Hello</h1>
            <p>World</p>
            <script>ignore()</script>
          </body>
        </html>
        """
        with patch('atlas.ai.tools.requests.get') as mock_get:
            response = MagicMock()
            response.status_code = 200
            response.text = html
            response.headers = {"Content-Type": "text/html"}
            mock_get.return_value = response

            result = _fetch_url_tool(broker, {"url": "http://example.com"})

        mock_get.assert_called_once()
        self.assertTrue(result.success)
        self.assertEqual(result.data["title"], "Example")
        self.assertIn("Hello", result.data["text"]) 

    def test_requires_url(self):
        broker = DummyBroker()
        with self.assertRaises(BrokerError):
            _fetch_url_tool(broker, {})


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
