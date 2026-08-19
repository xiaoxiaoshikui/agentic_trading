import unittest

import pandas as pd

from src.tom.agent import TheoryOfMindAgent
from src.tom.evaluation.harness import EvalConfig, EvaluationHarness
from src.tom.opponents.base import AgentAction, OpponentModel, OpponentPrediction


class AlwaysBuyModel(OpponentModel):
    @property
    def name(self) -> str:
        return "always_buy"

    @property
    def market_share(self) -> float:
        return 0.3

    def predict(self, state):
        return OpponentPrediction(
            action=AgentAction.BUY,
            confidence=1.0,
            reasoning="always buy",
            volume_estimate=1.0,
            price_impact=0.001,
        )

    def get_impact(self, prediction: OpponentPrediction) -> float:
        return prediction.price_impact * self.market_share


class AlwaysSellModel(OpponentModel):
    @property
    def name(self) -> str:
        return "always_sell"

    @property
    def market_share(self) -> float:
        return 0.3

    def predict(self, state):
        return OpponentPrediction(
            action=AgentAction.SELL,
            confidence=1.0,
            reasoning="always sell",
            volume_estimate=1.0,
            price_impact=-0.001,
        )

    def get_impact(self, prediction: OpponentPrediction) -> float:
        return prediction.price_impact * self.market_share


class TestTomCalibration(unittest.TestCase):
    def test_calibration_shifts_weight_to_better_model(self) -> None:
        idx = pd.date_range("2026-01-01", periods=320, freq="15min")
        close = pd.Series([100.0 + i * 0.1 for i in range(len(idx))], index=idx)
        df = pd.DataFrame(
            {
                "open": close.values,
                "high": (close * 1.001).values,
                "low": (close * 0.999).values,
                "close": close.values,
                "volume": [1000.0] * len(idx),
                "funding_rate": [0.0] * len(idx),
                "open_interest": [0.0] * len(idx),
            },
            index=idx,
        )

        harness = EvaluationHarness(EvalConfig(warmup_bars=50))
        agent = TheoryOfMindAgent(opponents=[AlwaysBuyModel(), AlwaysSellModel()])

        info = agent.calibrate_on_history(
            {"BTCUSDT": df},
            state_builder=harness._build_state,
            warmup_bars=50,
            max_samples_per_symbol=250,
        )

        weights = info["weights"]
        self.assertGreater(weights["always_buy"], weights["always_sell"])
        self.assertGreater(info["n_samples"], 0)


if __name__ == "__main__":
    unittest.main()
