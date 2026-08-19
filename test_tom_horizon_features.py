import unittest

import pandas as pd

from src.tom.evaluation.harness import EvalConfig, EvaluationHarness


class TestTomHorizonFeatures(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = EvaluationHarness(EvalConfig())

    def test_15m_series_uses_4_bars_for_1h(self) -> None:
        index = pd.date_range("2026-01-01 00:00:00", periods=6, freq="15min")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)

        change = self.harness._pct_change_over_horizon(close, pd.Timedelta(hours=1))
        expected = (105.0 / 101.0) - 1.0
        self.assertAlmostEqual(change, expected, places=12)

    def test_1h_series_uses_1_bar_for_1h(self) -> None:
        index = pd.date_range("2026-01-01 00:00:00", periods=6, freq="1h")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index)

        change = self.harness._pct_change_over_horizon(close, pd.Timedelta(hours=1))
        expected = (105.0 / 104.0) - 1.0
        self.assertAlmostEqual(change, expected, places=12)

    def test_insufficient_history_returns_zero(self) -> None:
        index = pd.date_range("2026-01-01 00:00:00", periods=2, freq="1h")
        close = pd.Series([100.0, 101.0], index=index)

        change = self.harness._pct_change_over_horizon(close, pd.Timedelta(days=7))
        self.assertEqual(change, 0.0)


if __name__ == "__main__":
    unittest.main()
