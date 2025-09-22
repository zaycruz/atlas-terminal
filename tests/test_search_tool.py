import unittest
from unittest.mock import MagicMock, patch

from atlas.ai.tools import _search_tool, ToolResult
from atlas.brokers.alpaca import AlpacaBroker
from atlas.brokers.base import BrokerError


class DummyBroker(AlpacaBroker):
    def __init__(self):
        pass

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def get_latest_quote(self, symbol: str):  # pragma: no cover - stub
        return {}


class SearchToolTests(unittest.TestCase):
    def test_search_returns_results(self):
        broker = DummyBroker()
        with patch('atlas.ai.tools.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"title": "Result 1", "url": "http://example.com", "content": "snippet"}
                ]
            }
            mock_get.return_value = mock_response

            result = _search_tool(broker, {"query": "test", "max_results": 3})

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        headers = kwargs["headers"]
        self.assertEqual(headers.get("X-Forwarded-For"), "127.0.0.1")
        self.assertEqual(headers.get("Accept"), "application/json")
        self.assertEqual(headers.get("Referer"), "http://localhost")
        self.assertEqual(kwargs["params"].get("max_results"), 3)
        self.assertIsInstance(result, ToolResult)
        self.assertEqual(len(result.data), 1)
        self.assertEqual(result.data[0]["title"], "Result 1")

    def test_search_without_query_raises(self):
        broker = DummyBroker()
        with self.assertRaises(BrokerError):
            _search_tool(broker, {})


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
