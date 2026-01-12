import unittest

from atlas.cli import _parse_quantity


class CliHelpersTests(unittest.TestCase):
    def test_parse_quantity_handles_numeric_strings(self) -> None:
        self.assertEqual(_parse_quantity("1"), 1.0)
        self.assertAlmostEqual(_parse_quantity("0.5"), 0.5)

    def test_parse_quantity_rejects_non_numeric(self) -> None:
        with self.assertRaises(ValueError):
            _parse_quantity("nope")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
