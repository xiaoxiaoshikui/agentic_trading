"""
Deep multimodal dataset helpers for ACM MM style experiments.

This module keeps the original event-aligned dataset rows, but augments each
sample with a raw price sequence window so sequence models can operate on
structured temporal inputs instead of hand-crafted indicators only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .mm_dataset import MultimodalDatasetBuilder, MultimodalDatasetConfig, WEB_FEATURES


SEQUENCE_FEATURES = [
    "open_rel_last",
    "high_rel_last",
    "low_rel_last",
    "close_rel_last",
    "volume_z_window",
    "return_1",
]


@dataclass
class DeepMultimodalArtifacts:
    dataset: pd.DataFrame
    price_sequences: np.ndarray
    web_features: np.ndarray
    metadata: Dict[str, Any]


class DeepMultimodalDatasetBuilder:
    def __init__(self, config: MultimodalDatasetConfig):
        self.config = config
        self.base_builder = MultimodalDatasetBuilder(config)

    def build(self) -> DeepMultimodalArtifacts:
        raw_records = self.base_builder._load_raw_history_records()
        price_frames = self.base_builder._load_price_frames(raw_records)
        history_df = self.base_builder._load_history_features(raw_records)
        datasets = []
        for symbol in self.base_builder._resolved_symbols():
            price_df = price_frames.get(symbol)
            if price_df is None or price_df.empty:
                continue
            symbol_history = history_df[history_df["symbol"] == symbol].copy() if not history_df.empty else history_df
            feature_df = self.base_builder._build_price_feature_frame(price_df, symbol=symbol)
            merged = self.base_builder._align_modalities(feature_df, symbol_history)
            dataset = self.base_builder._finalize_dataset(merged)
            if not dataset.empty:
                datasets.append(dataset)
        if not datasets:
            raise ValueError("No multimodal samples were produced for deep experiments.")
        dataset = pd.concat(datasets, ignore_index=True).sort_values(["window_end", "symbol"]).reset_index(drop=True)
        metadata = self.base_builder._build_metadata(price_frames, history_df, dataset)

        price_sequences = self._extract_price_sequences(price_frames, dataset[["symbol", "window_end"]])
        web_features = dataset[WEB_FEATURES].to_numpy(dtype=float)
        metadata["sequence"] = {
            "lookback_bars": int(self.config.lookback_bars),
            "sequence_features": SEQUENCE_FEATURES,
            "sequence_shape": [int(price_sequences.shape[1]), int(price_sequences.shape[2])],
        }
        return DeepMultimodalArtifacts(
            dataset=dataset,
            price_sequences=price_sequences,
            web_features=web_features,
            metadata=metadata,
        )

    def save(self, artifacts: DeepMultimodalArtifacts, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        stem = (
            f"{self.base_builder._symbol_tag()}_{self.config.interval}_"
            f"lb{self.config.lookback_bars}_hz{self.config.horizon_bars}_"
            f"{self.config.alignment_mode}_deep"
        )
        dataset_path = os.path.join(output_dir, f"{stem}.parquet")
        sequence_path = os.path.join(output_dir, f"{stem}_price_sequences.npy")
        metadata_path = os.path.join(output_dir, f"{stem}_metadata.json")
        artifacts.dataset.to_parquet(dataset_path, index=False)
        np.save(sequence_path, artifacts.price_sequences)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(artifacts.metadata, f, indent=2)
        return {
            "dataset": dataset_path,
            "price_sequences": sequence_path,
            "metadata": metadata_path,
        }

    def _extract_price_sequences(
        self,
        price_frames: Dict[str, pd.DataFrame],
        samples: pd.DataFrame,
    ) -> np.ndarray:
        prepared: Dict[str, Dict[str, Any]] = {}
        for symbol, price_df in price_frames.items():
            frame = price_df.sort_index().copy()
            close = frame["close"].astype(float)
            volume = frame["volume"].astype(float)
            ret_1 = close.pct_change().fillna(0.0)
            volume_mean = volume.rolling(32, min_periods=1).mean()
            volume_std = volume.rolling(32, min_periods=1).std().replace(0.0, np.nan)
            volume_z = ((volume - volume_mean) / volume_std).fillna(0.0)
            prepared[symbol] = {
                "frame": frame,
                "close": close,
                "ret_1": ret_1,
                "volume_z": volume_z,
                "timestamp_to_idx": {ts: idx for idx, ts in enumerate(frame.index)},
            }

        sequences = []
        for sample in samples.itertuples(index=False):
            symbol = str(sample.symbol)
            window_end = pd.Timestamp(sample.window_end)
            asset = prepared.get(symbol)
            if asset is None:
                raise KeyError(f"Missing price frame for symbol={symbol}")
            idx = asset["timestamp_to_idx"].get(window_end)
            if idx is None:
                raise KeyError(f"Missing price bar for symbol={symbol}, window_end={window_end}")
            start = idx - self.config.lookback_bars + 1
            if start < 0:
                raise ValueError(f"Not enough price history for symbol={symbol}, window_end={window_end}")
            window = asset["frame"].iloc[start : idx + 1]
            window_close = asset["close"].iloc[start : idx + 1]
            window_volume_z = asset["volume_z"].iloc[start : idx + 1]
            window_ret_1 = asset["ret_1"].iloc[start : idx + 1]

            last_close = float(window_close.iloc[-1])
            seq = np.column_stack(
                [
                    window["open"].to_numpy(dtype=float) / last_close - 1.0,
                    window["high"].to_numpy(dtype=float) / last_close - 1.0,
                    window["low"].to_numpy(dtype=float) / last_close - 1.0,
                    window["close"].to_numpy(dtype=float) / last_close - 1.0,
                    window_volume_z.to_numpy(dtype=float),
                    window_ret_1.to_numpy(dtype=float),
                ]
            ).astype(np.float32)
            sequences.append(seq)

        return np.stack(sequences, axis=0).astype(np.float32)
