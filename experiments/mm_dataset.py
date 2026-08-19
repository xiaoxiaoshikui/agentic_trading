"""
Multimodal dataset builder for offline market-intelligence experiments.

This module aligns cached price data with locally recorded web-intelligence
history from ``data/history/*.jsonl`` and exports a leakage-safe tabular
dataset for predictive experiments.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import DataLoader
from .mm_quality import (
    build_text_quality_fields,
    clean_text_fragment,
    collect_text_fragments,
    count_error_markers,
)

logger = logging.getLogger(__name__)


PRICE_FEATURES = [
    "ret_1",
    "ret_4",
    "ret_16",
    "ret_64",
    "range_16",
    "close_vs_sma_16",
    "close_vs_sma_64",
    "volume_z_16",
    "realized_vol_16",
    "atr_pct_14",
    "rsi_14",
]

WEB_FEATURES = [
    "news_direction_score",
    "news_confidence",
    "news_count",
    "fear_greed_norm",
    "social_sentiment_score",
    "whale_flow_score",
    "whale_activity_score",
    "web_context_strength",
    "modality_age_minutes",
    "has_news",
    "has_fear_greed",
    "has_social",
    "has_whale",
]

TEXT_AUX_FEATURES = [
    "text_char_len",
    "text_source_count",
    "text_error_count",
    "text_has_content",
    "text_fragment_count",
    "text_template_count",
    "text_template_ratio",
    "text_quality_score",
]

ALL_FEATURES = PRICE_FEATURES + WEB_FEATURES


@dataclass
class MultimodalDatasetConfig:
    symbol: str = "BTCUSDT"
    symbols: Optional[List[str]] = None
    interval: str = "15m"
    n_bars: int = 12000
    end_time: Optional[str] = "2026-02-07T00:00:00Z"
    cache_dir: str = "experiments/data_cache"
    history_glob: str = "data/history/*.jsonl"
    output_dir: str = "experiments/mm_cache"
    lookback_bars: int = 64
    horizon_bars: int = 4
    max_history_lag_minutes: int = 180
    alignment_mode: str = "bar"
    require_web_context: bool = True
    min_web_context_strength: float = 0.0
    min_text_char_len: int = 0
    min_text_source_count: int = 0
    max_text_error_count: int = 999
    max_text_template_ratio: float = 1.0
    min_text_quality_score: float = 0.0
    cached_only: bool = True
    force_download: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalDatasetConfig":
        return cls(**data)


class MultimodalDatasetBuilder:
    def __init__(self, config: MultimodalDatasetConfig):
        self.config = config
        self.data_loader = DataLoader(cache_dir=config.cache_dir)
        self.price_source = "cache"

    def build(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        raw_records = self._load_raw_history_records()
        symbols = self._resolved_symbols()
        price_frames = self._load_price_frames(raw_records)
        history_df = self._load_history_features(raw_records)

        datasets: List[pd.DataFrame] = []
        for symbol in symbols:
            price_df = price_frames.get(symbol)
            if price_df is None or price_df.empty:
                logger.warning("Skipping %s because no price frame was loaded", symbol)
                continue
            symbol_history = history_df[history_df["symbol"] == symbol].copy() if not history_df.empty else history_df
            feature_df = self._build_price_feature_frame(price_df, symbol=symbol)
            merged = self._align_modalities(feature_df, symbol_history)
            dataset = self._finalize_dataset(merged)
            if not dataset.empty:
                datasets.append(dataset)

        if not datasets:
            raise ValueError("No multimodal samples were produced for the requested symbols.")

        dataset = pd.concat(datasets, ignore_index=True).sort_values(["window_end", "symbol"]).reset_index(drop=True)
        metadata = self._build_metadata(price_frames, history_df, dataset)
        return dataset, metadata

    def save(self, dataset: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, str]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        stem = (
            f"{self._symbol_tag()}_{self.config.interval}_"
            f"lb{self.config.lookback_bars}_hz{self.config.horizon_bars}"
        )
        if self.config.alignment_mode != "bar":
            stem = f"{stem}_{self.config.alignment_mode}"
        dataset_path = os.path.join(self.config.output_dir, f"{stem}.parquet")
        metadata_path = os.path.join(self.config.output_dir, f"{stem}_metadata.json")
        try:
            dataset.to_parquet(dataset_path, index=False)
        except Exception:
            dataset_path = os.path.join(self.config.output_dir, f"{stem}.csv")
            dataset.to_csv(dataset_path, index=False)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return {"dataset": dataset_path, "metadata": metadata_path}

    def _load_raw_history_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        allowed_symbols = set(self._resolved_symbols())
        for path in self._resolve_history_paths():
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception as exc:
                        logger.warning("Skipping malformed history line %s:%s (%s)", path, line_no, exc)
                        continue
                    web = record.get("web_intelligence") or {}
                    symbol = web.get("symbol") or record.get("symbol")
                    if symbol not in allowed_symbols:
                        continue
                    if self._coerce_record_time(record) is None:
                        continue
                    records.append(record)
        return records

    def _load_price_frames(self, raw_records: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        end_time_ms = self.data_loader._normalize_end_time(self.config.end_time)
        symbols = self._resolved_symbols()
        if self.config.cached_only:
            for symbol in symbols:
                cache_path = self.data_loader.get_cache_path(
                    symbol,
                    self.config.interval,
                    self.config.n_bars,
                    end_time_ms=end_time_ms,
                )
                if not os.path.exists(cache_path):
                    raise FileNotFoundError(
                        f"Missing cached data for {symbol} at {cache_path}. "
                        "Set cached_only=false or populate the cache first."
                    )

        try:
            data = self.data_loader.load_data(
                symbols=symbols,
                interval=self.config.interval,
                n_bars=self.config.n_bars,
                force_download=self.config.force_download,
                end_time=self.config.end_time,
            )
            self.price_source = "cache"
            frames: Dict[str, pd.DataFrame] = {}
            for symbol in symbols:
                if symbol not in data:
                    raise ValueError(f"Failed to load price data for {symbol}")
                df = data[symbol].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                frames[symbol] = df.sort_index()
            return frames
        except Exception as exc:
            logger.warning("Falling back to history-derived price proxy: %s", exc)
            self.price_source = "history_proxy"
            return {
                symbol: self._reconstruct_price_frame_from_history(raw_records, symbol)
                for symbol in symbols
            }

    def _load_history_features(self, raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in raw_records:
            row = self._extract_history_row(record)
            if row is not None:
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["symbol", "event_time"] + WEB_FEATURES)

        history_df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "event_time"]).sort_values(
            ["symbol", "event_time"]
        )
        history_df["event_time"] = pd.to_datetime(history_df["event_time"]).dt.tz_localize(None)
        return history_df

    def _reconstruct_price_frame_from_history(
        self,
        raw_records: List[Dict[str, Any]],
        symbol: str,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in raw_records:
            record_symbol = (record.get("web_intelligence") or {}).get("symbol") or record.get("symbol")
            if record_symbol != symbol:
                continue
            event_time = self._coerce_record_time(record)
            price = self._bounded_float(record.get("entry_price"), default=np.nan)
            if event_time is None or not np.isfinite(price) or price <= 0:
                continue
            volume_strength = self._extract_numeric(
                (record.get("indicators") or {}).get("volume", {}),
                ["strength"],
                default=0.0,
            )
            rows.append(
                {
                    "open_time": event_time,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": max(1.0, abs(volume_strength) * 100.0),
                    "funding_rate": 0.0,
                    "open_interest": 0.0,
                }
            )

        if not rows:
            raise ValueError("History logs do not contain enough price information to build a fallback dataset.")

        df = pd.DataFrame(rows)
        df["open_time"] = pd.to_datetime(df["open_time"]).dt.tz_localize(None)
        df = df.groupby("open_time", as_index=False).last().set_index("open_time").sort_index()
        return df

    def _extract_history_row(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        web = record.get("web_intelligence") or {}
        symbol = web.get("symbol")
        if symbol not in set(self._resolved_symbols()):
            return None

        event_time = self._coerce_record_time(record)
        if event_time is None:
            return None

        overall = web.get("overall_assessment") or {}
        news_summary = web.get("news_summary") or {}
        fear_greed = web.get("fear_greed_index") or {}
        whale = self._normalize_blob(web.get("whale_movements"))
        social = self._normalize_blob(web.get("social_sentiment"))

        news_direction = self._direction_to_score(
            overall.get("direction")
            or news_summary.get("overall_sentiment")
            or news_summary.get("sentiment_overall")
        )
        news_conf = self._bounded_float(overall.get("confidence"), 0.5)
        news_count = self._count_news_items(news_summary)

        fear_value = self._extract_numeric(
            fear_greed,
            ["current_value", "index", "value", "currentValue"],
            default=50.0,
        )
        fear_norm = float(np.clip((fear_value - 50.0) / 50.0, -1.0, 1.0))

        social_score = self._extract_numeric(social, ["sentiment_score"], default=np.nan)
        if not np.isfinite(social_score):
            social_score = 100.0 * self._direction_to_score(
                social.get("overall_sentiment") if isinstance(social, dict) else None
            )
        social_score = float(np.clip(social_score / 100.0, -1.0, 1.0))

        whale_flow_score = self._flow_to_score(
            whale.get("net_flow") if isinstance(whale, dict) else None
        )
        whale_activity = self._activity_to_score(
            whale.get("whale_activity") if isinstance(whale, dict) else None
        )

        has_news = 1 if news_count > 0 else 0
        has_fear = 1 if fear_greed else 0
        has_social = 1 if social else 0
        has_whale = 1 if whale else 0

        strength_components = [
            abs(news_direction) if has_news else 0.0,
            abs(fear_norm) if has_fear else 0.0,
            abs(social_score) if has_social else 0.0,
            abs(whale_flow_score) if has_whale else 0.0,
        ]

        return {
            "symbol": symbol,
            "event_time": event_time,
            "news_direction_score": float(news_direction),
            "news_confidence": float(np.clip(news_conf, 0.0, 1.0)),
            "news_count": int(news_count),
            "fear_greed_norm": float(fear_norm),
            "social_sentiment_score": float(social_score),
            "whale_flow_score": float(whale_flow_score),
            "whale_activity_score": float(whale_activity),
            "web_context_strength": float(np.mean(strength_components)) if strength_components else 0.0,
            "has_news": int(has_news),
            "has_fear_greed": int(has_fear),
            "has_social": int(has_social),
            "has_whale": int(has_whale),
            **self._build_text_fields(
                news_summary=news_summary,
                fear_greed=fear_greed,
                social=social,
                whale=whale,
                overall=overall,
            ),
        }

    def _build_price_feature_frame(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        interval_minutes = self._interval_to_minutes(self.config.interval)
        window_15m = self._bars_for_minutes(15, interval_minutes)
        window_1h = self._bars_for_minutes(60, interval_minutes)
        window_4h = self._bars_for_minutes(240, interval_minutes)
        window_16h = self._bars_for_minutes(960, interval_minutes)
        atr_window = self._bars_for_minutes(210, interval_minutes)
        rsi_window = self._bars_for_minutes(210, interval_minutes)

        ret_1 = close.pct_change(window_15m)
        ret_4 = close.pct_change(window_1h)
        ret_16 = close.pct_change(window_4h)
        ret_64 = close.pct_change(window_16h)
        range_16 = (high.rolling(window_4h).max() - low.rolling(window_4h).min()) / close
        sma_16 = close.rolling(window_4h).mean()
        sma_64 = close.rolling(window_16h).mean()
        volume_mean_16 = volume.rolling(window_4h).mean()
        volume_std_16 = volume.rolling(window_4h).std().replace(0.0, np.nan)
        volume_z_16 = (volume - volume_mean_16) / volume_std_16
        realized_vol_16 = close.pct_change().rolling(window_4h).std()

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct_14 = tr.rolling(atr_window).mean() / close
        rsi_14 = self._compute_rsi(close, window=rsi_window)

        future_return = close.shift(-self.config.horizon_bars) / close - 1.0
        label_up = (future_return > 0).astype(int)

        feature_df = pd.DataFrame(
            {
                "window_end": pd.to_datetime(df.index),
                "symbol": symbol,
                "close": close.values,
                "future_return": future_return.values,
                "label_up": label_up.values,
                "ret_1": ret_1.values,
                "ret_4": ret_4.values,
                "ret_16": ret_16.values,
                "ret_64": ret_64.values,
                "range_16": range_16.values,
                "close_vs_sma_16": (close / sma_16 - 1.0).values,
                "close_vs_sma_64": (close / sma_64 - 1.0).values,
                "volume_z_16": volume_z_16.values,
                "realized_vol_16": realized_vol_16.values,
                "atr_pct_14": atr_pct_14.values,
                "rsi_14": (rsi_14 / 100.0).values,
            }
        )
        feature_df["window_end"] = feature_df["window_end"].dt.tz_localize(None)
        feature_df["has_full_price_context"] = (
            feature_df[PRICE_FEATURES].notna().all(axis=1) & feature_df["future_return"].notna()
        ).astype(int)
        return feature_df

    def _align_modalities(self, feature_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        if self.config.alignment_mode == "event":
            return self._align_events_to_price(feature_df, history_df)
        left = feature_df.copy()
        left["window_end"] = pd.to_datetime(left["window_end"]).astype("datetime64[ns]")
        left = left.sort_values("window_end")
        if history_df.empty:
            merged = left.copy()
            for col in WEB_FEATURES:
                merged[col] = 0.0 if col not in {"has_news", "has_fear_greed", "has_social", "has_whale"} else 0
            merged["event_time"] = pd.NaT
            merged["modality_age_minutes"] = float(self.config.max_history_lag_minutes + 1)
            merged["has_web_context"] = 0
            return merged

        tolerance = pd.Timedelta(minutes=self.config.max_history_lag_minutes)
        right = history_df.copy()
        right["event_time"] = pd.to_datetime(right["event_time"]).astype("datetime64[ns]")
        right = right.sort_values("event_time").drop(columns=["symbol"], errors="ignore")
        merged = pd.merge_asof(
            left,
            right,
            left_on="window_end",
            right_on="event_time",
            direction="backward",
            tolerance=tolerance,
        )
        merged["has_web_context"] = merged["event_time"].notna().astype(int)
        age = merged["window_end"] - merged["event_time"]
        merged["modality_age_minutes"] = age.dt.total_seconds().div(60.0)
        merged.loc[merged["has_web_context"] == 0, "modality_age_minutes"] = float(
            self.config.max_history_lag_minutes + 1
        )
        return merged

    def _align_events_to_price(self, feature_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            columns = ["event_time"] + WEB_FEATURES + list(feature_df.columns)
            return pd.DataFrame(columns=columns)

        tolerance = pd.Timedelta(minutes=self.config.max_history_lag_minutes)
        left = history_df.copy()
        left["event_time"] = pd.to_datetime(left["event_time"]).astype("datetime64[ns]")
        left = left.sort_values("event_time").drop(columns=["symbol"], errors="ignore")
        right = feature_df.copy()
        right["window_end"] = pd.to_datetime(right["window_end"]).astype("datetime64[ns]")
        right = right.sort_values("window_end")
        merged = pd.merge_asof(
            left,
            right,
            left_on="event_time",
            right_on="window_end",
            direction="backward",
            tolerance=tolerance,
        )
        merged["has_web_context"] = 1
        age = merged["event_time"] - merged["window_end"]
        merged["modality_age_minutes"] = age.dt.total_seconds().div(60.0)
        return merged

    def _finalize_dataset(self, merged: pd.DataFrame) -> pd.DataFrame:
        dataset = merged.copy()
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset[dataset["has_full_price_context"] == 1].copy()

        for col in WEB_FEATURES:
            if col not in dataset.columns:
                dataset[col] = 0.0

        for col in WEB_FEATURES:
            dataset[col] = dataset[col].fillna(0.0)

        for col in PRICE_FEATURES:
            dataset[col] = dataset[col].fillna(0.0)

        for col in TEXT_AUX_FEATURES:
            if col not in dataset.columns:
                dataset[col] = 0.0
            dataset[col] = dataset[col].fillna(0.0)

        if "text_blob" not in dataset.columns:
            dataset["text_blob"] = ""
        dataset["text_blob"] = dataset["text_blob"].fillna("").astype(str)

        dataset["label_up"] = dataset["label_up"].astype(int)
        dataset["window_end"] = pd.to_datetime(dataset["window_end"]).dt.tz_localize(None)
        dataset["event_time"] = pd.to_datetime(dataset["event_time"]).dt.tz_localize(None)

        if self.config.require_web_context:
            dataset = dataset[dataset["has_web_context"] == 1].copy()

        if self.config.min_web_context_strength > 0.0:
            before = len(dataset)
            dataset = dataset[dataset["web_context_strength"] >= self.config.min_web_context_strength].copy()
            logger.info(
                "min_web_context_strength=%.2f: kept %d / %d samples",
                self.config.min_web_context_strength,
                len(dataset),
                before,
            )

        if self.config.min_text_char_len > 0:
            before = len(dataset)
            dataset = dataset[dataset["text_char_len"] >= float(self.config.min_text_char_len)].copy()
            logger.info(
                "min_text_char_len=%d: kept %d / %d samples",
                self.config.min_text_char_len,
                len(dataset),
                before,
            )

        if self.config.min_text_source_count > 0:
            before = len(dataset)
            dataset = dataset[dataset["text_source_count"] >= float(self.config.min_text_source_count)].copy()
            logger.info(
                "min_text_source_count=%d: kept %d / %d samples",
                self.config.min_text_source_count,
                len(dataset),
                before,
            )

        if self.config.max_text_error_count < 999:
            before = len(dataset)
            dataset = dataset[dataset["text_error_count"] <= float(self.config.max_text_error_count)].copy()
            logger.info(
                "max_text_error_count=%d: kept %d / %d samples",
                self.config.max_text_error_count,
                len(dataset),
                before,
            )

        if self.config.max_text_template_ratio < 1.0:
            before = len(dataset)
            dataset = dataset[dataset["text_template_ratio"] <= float(self.config.max_text_template_ratio)].copy()
            logger.info(
                "max_text_template_ratio=%.2f: kept %d / %d samples",
                self.config.max_text_template_ratio,
                len(dataset),
                before,
            )

        if self.config.min_text_quality_score > 0.0:
            before = len(dataset)
            dataset = dataset[dataset["text_quality_score"] >= float(self.config.min_text_quality_score)].copy()
            logger.info(
                "min_text_quality_score=%.2f: kept %d / %d samples",
                self.config.min_text_quality_score,
                len(dataset),
                before,
            )

        dataset.reset_index(drop=True, inplace=True)
        return dataset

    def _build_metadata(
        self,
        price_frames: Dict[str, pd.DataFrame],
        history_df: pd.DataFrame,
        dataset: pd.DataFrame,
    ) -> Dict[str, Any]:
        if dataset.empty:
            raise ValueError("No multimodal samples were produced. Check history coverage or lag tolerance.")

        symbols = self._resolved_symbols()
        return {
            "config": self.config.to_dict(),
            "price_source": self.price_source,
            "alignment_mode": self.config.alignment_mode,
            "symbols": symbols,
            "n_symbols": int(len(symbols)),
            "n_price_bars": int(sum(len(df) for df in price_frames.values())),
            "n_price_bars_by_symbol": {symbol: int(len(price_frames[symbol])) for symbol in symbols if symbol in price_frames},
            "n_history_rows": int(len(history_df)),
            "n_history_rows_by_symbol": {
                symbol: int((history_df["symbol"] == symbol).sum()) for symbol in symbols
            }
            if not history_df.empty
            else {symbol: 0 for symbol in symbols},
            "n_samples": int(len(dataset)),
            "n_samples_by_symbol": {
                symbol: int((dataset["symbol"] == symbol).sum()) for symbol in symbols
            },
            "sample_start": str(
                dataset["event_time"].iloc[0]
                if self.config.alignment_mode == "event"
                else dataset["window_end"].iloc[0]
            ),
            "sample_end": str(
                dataset["event_time"].iloc[-1]
                if self.config.alignment_mode == "event"
                else dataset["window_end"].iloc[-1]
            ),
            "positive_rate": float(dataset["label_up"].mean()),
            "mean_future_return": float(dataset["future_return"].mean()),
            "mean_abs_future_return": float(dataset["future_return"].abs().mean()),
            "mean_modality_age_minutes": float(dataset["modality_age_minutes"].mean()),
            "coverage": {
                "has_news": float(dataset["has_news"].mean()),
                "has_fear_greed": float(dataset["has_fear_greed"].mean()),
                "has_social": float(dataset["has_social"].mean()),
                "has_whale": float(dataset["has_whale"].mean()),
                "has_web_context": float(dataset["has_web_context"].mean()),
                "text_has_content": float(dataset["text_has_content"].mean()),
                "mean_text_char_len": float(dataset["text_char_len"].mean()),
                "mean_text_source_count": float(dataset["text_source_count"].mean()),
                "mean_text_template_ratio": float(dataset["text_template_ratio"].mean()),
                "mean_text_quality_score": float(dataset["text_quality_score"].mean()),
            },
            "feature_sets": {
                "price": PRICE_FEATURES,
                "web": WEB_FEATURES,
                "all": ALL_FEATURES,
                "text_aux": TEXT_AUX_FEATURES,
            },
            "text_stats": {
                "mean_text_char_len": float(dataset["text_char_len"].mean()),
                "mean_text_source_count": float(dataset["text_source_count"].mean()),
                "mean_text_error_count": float(dataset["text_error_count"].mean()),
            },
            "feature_windows": {
                "interval_minutes": self._interval_to_minutes(self.config.interval),
                "ret_1_bars": self._bars_for_minutes(15, self._interval_to_minutes(self.config.interval)),
                "ret_4_bars": self._bars_for_minutes(60, self._interval_to_minutes(self.config.interval)),
                "ret_16_bars": self._bars_for_minutes(240, self._interval_to_minutes(self.config.interval)),
                "ret_64_bars": self._bars_for_minutes(960, self._interval_to_minutes(self.config.interval)),
                "atr_pct_14_bars": self._bars_for_minutes(210, self._interval_to_minutes(self.config.interval)),
                "rsi_14_bars": self._bars_for_minutes(210, self._interval_to_minutes(self.config.interval)),
            },
        }

    def _resolved_symbols(self) -> List[str]:
        if self.config.symbols:
            return list(dict.fromkeys(str(symbol).strip() for symbol in self.config.symbols if str(symbol).strip()))
        return [self.config.symbol]

    def _symbol_tag(self) -> str:
        symbols = self._resolved_symbols()
        if len(symbols) == 1:
            return symbols[0]
        return "_".join(symbols)

    def _resolve_history_paths(self) -> List[str]:
        patterns = self.config.history_glob
        if isinstance(patterns, str):
            patterns = [patterns]
        paths: List[str] = []
        for pattern in patterns:
            paths.extend(glob.glob(pattern))
        return sorted(dict.fromkeys(paths))

    @staticmethod
    def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    @staticmethod
    def _coerce_record_time(record: Dict[str, Any]) -> Optional[pd.Timestamp]:
        ts = record.get("timestamp_ts")
        if ts is not None:
            try:
                return pd.to_datetime(float(ts), unit="s")
            except Exception:
                pass

        iso_ts = record.get("timestamp")
        if iso_ts:
            try:
                return pd.to_datetime(iso_ts)
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_blob(blob: Any) -> Dict[str, Any]:
        if isinstance(blob, dict):
            raw = blob.get("raw_response")
            parsed = MultimodalDatasetBuilder._extract_json_object(raw) if isinstance(raw, str) else None
            if parsed:
                merged = dict(parsed)
                for k, v in blob.items():
                    merged.setdefault(k, v)
                return merged
            return blob
        if isinstance(blob, str):
            parsed = MultimodalDatasetBuilder._extract_json_object(blob)
            return parsed if parsed else {}
        return {}

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @classmethod
    def _build_text_fields(
        cls,
        news_summary: Dict[str, Any],
        fear_greed: Dict[str, Any],
        social: Dict[str, Any],
        whale: Dict[str, Any],
        overall: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_text_quality_fields(
            news_summary=news_summary,
            fear_greed=fear_greed,
            social=social,
            whale=whale,
            overall=overall,
        )

    @classmethod
    def _collect_text_fragments(cls, value: Any, fragments: List[str], depth: int = 0) -> None:
        collect_text_fragments(value, fragments, depth)

    @staticmethod
    def _clean_text_fragment(text: str) -> str:
        return clean_text_fragment(text)

    @classmethod
    def _count_error_markers(cls, value: Any, depth: int = 0) -> int:
        return count_error_markers(value, depth)

    @staticmethod
    def _interval_to_minutes(interval: str) -> int:
        match = re.fullmatch(r"(\d+)([mhd])", str(interval).strip().lower())
        if not match:
            raise ValueError(f"Unsupported interval format: {interval}")
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "m":
            return value
        if unit == "h":
            return value * 60
        if unit == "d":
            return value * 1440
        raise ValueError(f"Unsupported interval unit: {interval}")

    @staticmethod
    def _bars_for_minutes(target_minutes: int, interval_minutes: int) -> int:
        return max(1, int(round(target_minutes / float(interval_minutes))))

    @staticmethod
    def _bounded_float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
            return result if np.isfinite(result) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _extract_numeric(blob: Dict[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
        if not isinstance(blob, dict):
            return float(default)
        for key in keys:
            if key in blob:
                return MultimodalDatasetBuilder._bounded_float(blob.get(key), default=default)
        return float(default)

    @staticmethod
    def _direction_to_score(value: Any) -> float:
        if value is None:
            return 0.0
        text = str(value).strip().lower()
        if not text:
            return 0.0
        bullish_terms = ["bullish", "positive", "long", "up", "看多"]
        bearish_terms = ["bearish", "negative", "short", "down", "看空"]
        if any(term in text for term in bullish_terms):
            return 1.0
        if any(term in text for term in bearish_terms):
            return -1.0
        return 0.0

    @staticmethod
    def _activity_to_score(value: Any) -> float:
        if value is None:
            return 0.0
        text = str(value).strip().lower()
        if "high" in text:
            return 1.0
        if "medium" in text:
            return 0.5
        if "low" in text:
            return 0.25
        return 0.0

    @staticmethod
    def _flow_to_score(value: Any) -> float:
        if value is None:
            return 0.0
        text = str(value).strip().lower()
        if "outflow" in text:
            return 1.0
        if "inflow" in text:
            return -1.0
        return 0.0

    @staticmethod
    def _count_news_items(news_summary: Any) -> int:
        if not isinstance(news_summary, dict):
            return 0
        for key in ("news_summary", "news", "新闻摘要"):
            value = news_summary.get(key)
            if isinstance(value, list):
                return len(value)
        for value in news_summary.values():
            if isinstance(value, list):
                return len(value)
        return 0
