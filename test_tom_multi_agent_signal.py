import unittest
from datetime import datetime

import pandas as pd

from src.tom.agent import TechnicalExpertVote, TheoryOfMindAgent
from src.tom.base import MarketState


def _make_state(
    *,
    ema50: float,
    ema200: float,
    rsi: float,
    atr: float,
    price: float,
    pc1h: float,
    pc24h: float,
    vol_change: float,
) -> MarketState:
    return MarketState(
        symbol="BTCUSDT",
        df=pd.DataFrame(),
        current_price=price,
        price_change_1h=pc1h,
        price_change_24h=pc24h,
        price_change_7d=0.0,
        volume=1000.0,
        volume_change=vol_change,
        indicators={"ema_50": ema50, "ema_200": ema200, "rsi": rsi, "atr": atr},
        funding_rate=0.0,
        open_interest=0.0,
        timestamp=datetime.utcnow(),
    )


class TestTomMultiAgentSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = TheoryOfMindAgent(technical_mode="multi")

    def test_no_dynamic_strong_expert_allows_entry_with_capped_conf(self) -> None:
        state = _make_state(
            ema50=103.0,
            ema200=99.0,
            rsi=58.0,
            atr=1.0,
            price=100.0,
            pc1h=0.004,
            pc24h=0.03,
            vol_change=0.02,
        )
        sig = self.agent._technical_signal(state)
        self.assertEqual(sig.action, "LONG")
        self.assertLessEqual(sig.confidence, 0.30)

    def test_no_dynamic_high_vol_stays_flat(self) -> None:
        state = _make_state(
            ema50=110.0,
            ema200=100.0,
            rsi=55.0,
            atr=3.0,
            price=100.0,
            pc1h=0.03,
            pc24h=0.03,
            vol_change=0.2,
        )
        sig = self.agent._technical_signal(state)
        self.assertEqual(sig.action, "FLAT")

    def test_dynamic_alignment_prefers_long(self) -> None:
        state = _make_state(
            ema50=103.0,
            ema200=99.0,
            rsi=58.0,
            atr=1.0,
            price=100.0,
            pc1h=0.004,
            pc24h=0.03,
            vol_change=0.02,
        )
        self.agent._dynamic_strategy_vote = lambda _: TechnicalExpertVote(
            name="dynamic",
            score=0.70,
            confidence=0.70,
            reasoning="forced_test_alignment",
        )
        sig = self.agent._technical_signal(state)
        self.assertEqual(sig.action, "LONG")
        self.assertGreater(sig.confidence, 0.2)

    def test_dynamic_alignment_prefers_short(self) -> None:
        state = _make_state(
            ema50=99.0,
            ema200=103.0,
            rsi=42.0,
            atr=1.0,
            price=100.0,
            pc1h=-0.004,
            pc24h=-0.03,
            vol_change=0.02,
        )
        self.agent._dynamic_strategy_vote = lambda _: TechnicalExpertVote(
            name="dynamic",
            score=-0.72,
            confidence=0.72,
            reasoning="forced_test_alignment",
        )
        sig = self.agent._technical_signal(state)
        self.assertEqual(sig.action, "SHORT")

    def test_dynamic_conflict_flattens_signal(self) -> None:
        state = _make_state(
            ema50=106.0,
            ema200=100.0,
            rsi=20.0,
            atr=1.0,
            price=100.0,
            pc1h=0.002,
            pc24h=0.01,
            vol_change=0.01,
        )

        self.agent._dynamic_strategy_vote = lambda _: TechnicalExpertVote(
            name="dynamic",
            score=-0.80,
            confidence=0.80,
            reasoning="forced_test_conflict",
        )
        sig = self.agent._technical_signal(state)
        self.assertEqual(sig.action, "FLAT")

    def test_dynamic_conflict_trend_biases_dynamic(self) -> None:
        state = _make_state(
            ema50=106.0,
            ema200=100.0,
            rsi=20.0,
            atr=1.0,
            price=100.0,
            pc1h=0.002,
            pc24h=0.03,
            vol_change=0.01,
        )

        self.agent._dynamic_strategy_vote = lambda _: TechnicalExpertVote(
            name="dynamic",
            score=-0.80,
            confidence=0.80,
            reasoning="forced_test_conflict_trend",
        )
        sig = self.agent._technical_signal(state)
        self.assertEqual(sig.action, "SHORT")
        self.assertTrue(
            ("trend_conflict_dynamic_bias" in sig.reasoning) or ("dynamic_direct_trend" in sig.reasoning)
        )


if __name__ == "__main__":
    unittest.main()
