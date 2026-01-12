import unittest

from atlas.brokers.alpaca import _parse_option_symbol
from atlas.brokers.base import BrokerError


class OptionSymbolParsingTests(unittest.TestCase):
    def test_parses_call_symbol(self):
        exp, opt_type, strike = _parse_option_symbol("AAPL240920C00190000")
        self.assertEqual(exp, "2024-09-20")
        self.assertEqual(opt_type, "call")
        self.assertAlmostEqual(strike, 190.0)

    def test_parses_put_symbol(self):
        exp, opt_type, strike = _parse_option_symbol("TSLA250117P00500000")
        self.assertEqual(exp, "2025-01-17")
        self.assertEqual(opt_type, "put")
        self.assertAlmostEqual(strike, 500.0)

    def test_invalid_symbol_raises(self):
        with self.assertRaises(BrokerError):
            _parse_option_symbol("INVALID")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
