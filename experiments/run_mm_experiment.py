#!/usr/bin/env python
"""
Run a minimal offline multimodal market-intelligence experiment.

This script builds a leakage-safe dataset from cached price data plus locally
recorded web-intelligence history, then evaluates linear baselines over
different modality subsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.mm_dataset import (
    ALL_FEATURES,
    PRICE_FEATURES,
    TEXT_AUX_FEATURES,
    WEB_FEATURES,
    MultimodalDatasetBuilder,
    MultimodalDatasetConfig,
)
from experiments.mm_metrics import (
    compute_classification_metrics,
    compute_downstream_metrics,
    sigmoid,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    feature_group: str
    kind: str = "ridge"
    alpha: float = 1.0
    web_alpha: float = 3.0
    text_alpha: float = 1.0
    text_dim: int = 256


class RidgeBinaryClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2:
            raise ValueError("Expected a 2D feature matrix")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Feature and label counts do not match")

        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        x_std = (x - self.mean_) / self.scale_
        x_aug = np.column_stack([np.ones(len(x_std)), x_std])
        target = 2.0 * y - 1.0
        reg = self.alpha * np.eye(x_aug.shape[1], dtype=float)
        reg[0, 0] = 0.0
        self.weights_ = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ target)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise RuntimeError("Model has not been fit")
        x = np.asarray(x, dtype=float)
        x_std = (x - self.mean_) / self.scale_
        x_aug = np.column_stack([np.ones(len(x_std)), x_std])
        return sigmoid(x_aug @ self.weights_)


class HashingTextEncoder:
    def __init__(self, n_features: int = 256, ngram_min: int = 3, ngram_max: int = 5):
        self.n_features = int(n_features)
        self.ngram_min = int(ngram_min)
        self.ngram_max = int(ngram_max)

    def transform(self, texts: List[str] | pd.Series) -> np.ndarray:
        rows = np.zeros((len(texts), self.n_features), dtype=float)
        for row_idx, raw_text in enumerate(texts):
            text = self._normalize_text(raw_text)
            if not text:
                continue
            for ngram_size in range(self.ngram_min, self.ngram_max + 1):
                if len(text) < ngram_size:
                    continue
                for start in range(len(text) - ngram_size + 1):
                    ngram = text[start : start + ngram_size]
                    digest = hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest()
                    feature_idx = int.from_bytes(digest[:4], "little") % self.n_features
                    sign = 1.0 if digest[4] % 2 == 0 else -1.0
                    rows[row_idx, feature_idx] += sign
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return rows / norms

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = "" if value is None else str(value)
        return " ".join(text.split())


class TfidfTextEncoder:
    def __init__(self, max_features: int = 512):
        self.max_features = int(max_features)
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            preprocessor=None,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=2,
            sublinear_tf=True,
        )
        self.fitted = False

    def fit_transform(self, texts: List[str] | pd.Series) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(list(texts))
        self.fitted = True
        return matrix.toarray().astype(float)

    def transform(self, texts: List[str] | pd.Series) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfTextEncoder has not been fit")
        return self.vectorizer.transform(list(texts)).toarray().astype(float)

    def get_feature_names(self) -> List[str]:
        if not self.fitted:
            return []
        return [f"tfidf::{name}" for name in self.vectorizer.get_feature_names_out().tolist()]

    @classmethod
    def _tokenize(cls, raw_text: Any) -> List[str]:
        text = HashingTextEncoder._normalize_text(raw_text)
        if not text:
            return []
        tokens: List[str] = []
        for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text):
            if re.fullmatch(r"[A-Za-z0-9_]+", part):
                tokens.append(part.lower())
            else:
                tokens.extend(cls._cjk_ngrams(part, 2))
                tokens.extend(cls._cjk_ngrams(part, 3))
        return tokens

    @staticmethod
    def _cjk_ngrams(text: str, n: int) -> List[str]:
        if len(text) <= n:
            return [text] if text else []
        return [text[idx : idx + n] for idx in range(len(text) - n + 1)]


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_experiment_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_features(feature_group: str) -> List[str]:
    if feature_group == "price":
        return PRICE_FEATURES
    if feature_group == "web":
        return WEB_FEATURES
    if feature_group == "text_aux":
        return TEXT_AUX_FEATURES
    if feature_group == "all":
        return ALL_FEATURES
    raise ValueError(f"Unknown feature group: {feature_group}")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def split_dataset(dataset: pd.DataFrame, train_ratio: float, val_ratio: float) -> Dict[str, pd.DataFrame]:
    sort_cols = ["window_end"]
    if "event_time" in dataset.columns and dataset["event_time"].notna().any():
        sort_cols = ["event_time", "window_end"]
    dataset = dataset.sort_values(sort_cols).reset_index(drop=True)

    unique_windows = dataset["window_end"].drop_duplicates().tolist()
    n = len(unique_windows)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    val_end = min(val_end, n - 1)
    train_windows = set(unique_windows[:train_end])
    val_windows = set(unique_windows[train_end:val_end])
    test_windows = set(unique_windows[val_end:])
    return {
        "train": dataset[dataset["window_end"].isin(train_windows)].copy(),
        "val": dataset[dataset["window_end"].isin(val_windows)].copy(),
        "test": dataset[dataset["window_end"].isin(test_windows)].copy(),
    }


def build_rolling_splits(
    dataset: pd.DataFrame,
    min_train_windows: int,
    val_windows: int,
    test_windows: int,
    step_windows: int,
) -> List[Dict[str, pd.DataFrame]]:
    sort_cols = ["window_end"]
    if "event_time" in dataset.columns and dataset["event_time"].notna().any():
        sort_cols = ["event_time", "window_end"]
    dataset = dataset.sort_values(sort_cols).reset_index(drop=True)
    unique_windows = dataset["window_end"].drop_duplicates().tolist()

    splits: List[Dict[str, pd.DataFrame]] = []
    start = int(min_train_windows)
    while start + val_windows + test_windows <= len(unique_windows):
        train_windows = set(unique_windows[:start])
        val_windows_set = set(unique_windows[start : start + val_windows])
        test_windows_set = set(unique_windows[start + val_windows : start + val_windows + test_windows])
        splits.append(
            {
                "train": dataset[dataset["window_end"].isin(train_windows)].copy(),
                "val": dataset[dataset["window_end"].isin(val_windows_set)].copy(),
                "test": dataset[dataset["window_end"].isin(test_windows_set)].copy(),
            }
        )
        start += step_windows

    if not splits:
        raise ValueError("Rolling split configuration produced zero folds. Reduce window sizes or expand the dataset.")
    return splits


def select_trade_thresholds(
    y_prob: np.ndarray,
    future_returns: np.ndarray,
    cost_bps: float = 0.0,
) -> Tuple[float, float]:
    best_thresholds = (0.55, 0.45)
    best_objective = -1e9
    for long_th in np.arange(0.52, 0.71, 0.03):
        for short_th in np.arange(0.29, 0.49, 0.03):
            if short_th >= long_th:
                continue
            metrics = compute_downstream_metrics(
                future_returns=future_returns,
                y_prob=y_prob,
                long_threshold=float(long_th),
                short_threshold=float(short_th),
                cost_bps=cost_bps,
            )
            if metrics.n_trades < max(5, int(0.05 * len(y_prob))):
                continue
            objective = metrics.sharpe_like - 0.5 * metrics.max_drawdown
            if objective > best_objective:
                best_objective = objective
                best_thresholds = (float(long_th), float(short_th))
    return best_thresholds


def aggregate_for_downstream(split_df: pd.DataFrame, y_prob: np.ndarray) -> pd.DataFrame:
    group_cols = ["window_end"]
    if "symbol" in split_df.columns:
        group_cols = ["symbol", "window_end"]
    downstream_df = split_df[group_cols + ["future_return"]].copy()
    downstream_df["y_prob"] = np.asarray(y_prob, dtype=float)
    downstream_df = downstream_df.sort_values(group_cols)
    return (
        downstream_df.groupby(group_cols, as_index=False)
        .agg({"future_return": "last", "y_prob": "mean"})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_text_aux_matrix(df: pd.DataFrame) -> np.ndarray:
    aux = df[TEXT_AUX_FEATURES].to_numpy(dtype=float).copy()
    aux[:, 0] = np.log1p(aux[:, 0]) / 8.0
    aux[:, 1] = np.log1p(aux[:, 1])
    aux[:, 2] = np.log1p(aux[:, 2])
    return aux


def build_text_matrix(df: pd.DataFrame, encoder: HashingTextEncoder) -> np.ndarray:
    hashed = encoder.transform(df["text_blob"].fillna("").astype(str).tolist())
    aux = build_text_aux_matrix(df)
    return np.hstack([hashed, aux])


def text_feature_names(text_dim: int) -> List[str]:
    return [f"text_hash_{idx}" for idx in range(int(text_dim))] + TEXT_AUX_FEATURES


def build_tfidf_text_matrix(
    df: pd.DataFrame,
    encoder: TfidfTextEncoder,
    fit: bool = False,
) -> np.ndarray:
    texts = df["text_blob"].fillna("").astype(str).tolist()
    text_matrix = encoder.fit_transform(texts) if fit else encoder.transform(texts)
    aux = build_text_aux_matrix(df)
    return np.hstack([text_matrix, aux])


def select_fusion_weight(
    y_true: np.ndarray,
    future_returns: np.ndarray,
    base_prob: np.ndarray,
    text_prob: np.ndarray,
    cost_bps: float = 0.0,
) -> float:
    best_weight = 0.5
    best_objective = -1e9
    for weight in np.linspace(0.0, 1.0, 11):
        y_prob = (1.0 - weight) * base_prob + weight * text_prob
        cls = compute_classification_metrics(y_true, y_prob)
        down = compute_downstream_metrics(
            future_returns=future_returns,
            y_prob=y_prob,
            long_threshold=0.55,
            short_threshold=0.45,
            cost_bps=cost_bps,
        )
        objective = cls.macro_f1 + 0.05 * down.sharpe_like
        if objective > best_objective:
            best_objective = objective
            best_weight = float(weight)
    return best_weight


def aggregate_numeric_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    aggregated: Dict[str, Any] = {}
    for key in keys:
        values = [d[key] for d in dicts if key in d]
        if not values:
            continue
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            arr = np.asarray(values, dtype=float)
            aggregated[key] = float(np.mean(arr))
            aggregated[f"{key}_std"] = float(np.std(arr))
        else:
            aggregated[key] = values[0]
    return aggregated


def aggregate_model_results(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not fold_results:
        raise ValueError("No fold results to aggregate")

    first = fold_results[0]
    aggregated = {
        "model": first["model"],
        "kind": first["kind"],
        "feature_group": first["feature_group"],
        "n_features": first["n_features"],
        "features": first["features"],
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "val_classification": aggregate_numeric_dicts([r["val_classification"] for r in fold_results]),
        "test_classification": aggregate_numeric_dicts([r["test_classification"] for r in fold_results]),
        "val_downstream": aggregate_numeric_dicts([r["val_downstream"] for r in fold_results]),
        "test_downstream": aggregate_numeric_dicts([r["test_downstream"] for r in fold_results]),
        "val_trade_windows": float(np.mean([r.get("val_trade_windows", 0) for r in fold_results])),
        "test_trade_windows": float(np.mean([r.get("test_trade_windows", 0) for r in fold_results])),
    }

    optional_numeric = ["gamma", "web_alpha", "text_alpha", "fusion_alpha", "text_dim"]
    for key in optional_numeric:
        values = [r[key] for r in fold_results if key in r]
        if values:
            aggregated[key] = float(np.mean(np.asarray(values, dtype=float)))
            aggregated[f"{key}_std"] = float(np.std(np.asarray(values, dtype=float)))
    return aggregated


def run_model(
    spec: ModelSpec,
    splits: Dict[str, pd.DataFrame],
    transaction_cost_bps: float = 0.0,
) -> Dict[str, Any]:
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    y_train = train["label_up"].to_numpy(dtype=int)
    y_val = val["label_up"].to_numpy(dtype=int)
    y_test = test["label_up"].to_numpy(dtype=int)

    extra: Dict[str, Any] = {}

    if spec.kind == "gated_residual":
        price_model = RidgeBinaryClassifier(alpha=spec.alpha)
        web_model = RidgeBinaryClassifier(alpha=spec.web_alpha)
        price_model.fit(train[PRICE_FEATURES].to_numpy(dtype=float), y_train)
        web_model.fit(train[WEB_FEATURES].to_numpy(dtype=float), y_train)

        val_price = price_model.predict_proba(val[PRICE_FEATURES].to_numpy(dtype=float))
        val_web = web_model.predict_proba(val[WEB_FEATURES].to_numpy(dtype=float))
        val_strength = val["web_context_strength"].to_numpy(dtype=float)

        best_gamma = 0.0
        best_score = -1e9
        for gamma in np.linspace(-1.5, 1.5, 31):
            val_prob = sigmoid(logit(val_price) + gamma * val_strength * (val_web - 0.5) * 2.0)
            cls = compute_classification_metrics(y_val, val_prob)
            down = compute_downstream_metrics(
                future_returns=val["future_return"].to_numpy(dtype=float),
                y_prob=val_prob,
                long_threshold=0.55,
                short_threshold=0.45,
                cost_bps=transaction_cost_bps,
            )
            objective = cls.macro_f1 + 0.05 * down.sharpe_like
            if objective > best_score:
                best_score = objective
                best_gamma = float(gamma)

        test_price = price_model.predict_proba(test[PRICE_FEATURES].to_numpy(dtype=float))
        test_web = web_model.predict_proba(test[WEB_FEATURES].to_numpy(dtype=float))
        test_strength = test["web_context_strength"].to_numpy(dtype=float)

        val_prob = sigmoid(logit(val_price) + best_gamma * val_strength * (val_web - 0.5) * 2.0)
        test_prob = sigmoid(logit(test_price) + best_gamma * test_strength * (test_web - 0.5) * 2.0)
        features = PRICE_FEATURES + WEB_FEATURES
        extra = {"gamma": best_gamma, "web_alpha": spec.web_alpha}
    elif spec.kind == "text_hash_ridge":
        encoder = HashingTextEncoder(n_features=spec.text_dim)
        x_train = build_text_matrix(train, encoder)
        x_val = build_text_matrix(val, encoder)
        x_test = build_text_matrix(test, encoder)

        model = RidgeBinaryClassifier(alpha=spec.text_alpha)
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)
        test_prob = model.predict_proba(x_test)
        features = text_feature_names(spec.text_dim)
        extra = {"text_dim": spec.text_dim, "text_alpha": spec.text_alpha}
    elif spec.kind == "text_tfidf_ridge":
        encoder = TfidfTextEncoder(max_features=spec.text_dim)
        x_train = build_tfidf_text_matrix(train, encoder, fit=True)
        x_val = build_tfidf_text_matrix(val, encoder, fit=False)
        x_test = build_tfidf_text_matrix(test, encoder, fit=False)

        model = RidgeBinaryClassifier(alpha=spec.text_alpha)
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)
        test_prob = model.predict_proba(x_test)
        features = encoder.get_feature_names() + TEXT_AUX_FEATURES
        extra = {"text_dim": spec.text_dim, "text_alpha": spec.text_alpha}
    elif spec.kind == "text_concat":
        numeric_features = resolve_features(spec.feature_group)
        encoder = HashingTextEncoder(n_features=spec.text_dim)
        x_train = np.hstack([train[numeric_features].to_numpy(dtype=float), build_text_matrix(train, encoder)])
        x_val = np.hstack([val[numeric_features].to_numpy(dtype=float), build_text_matrix(val, encoder)])
        x_test = np.hstack([test[numeric_features].to_numpy(dtype=float), build_text_matrix(test, encoder)])

        model = RidgeBinaryClassifier(alpha=spec.alpha)
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)
        test_prob = model.predict_proba(x_test)
        features = numeric_features + text_feature_names(spec.text_dim)
        extra = {"text_dim": spec.text_dim, "text_alpha": spec.text_alpha}
    elif spec.kind == "text_tfidf_concat":
        numeric_features = resolve_features(spec.feature_group)
        encoder = TfidfTextEncoder(max_features=spec.text_dim)
        x_train = np.hstack(
            [train[numeric_features].to_numpy(dtype=float), build_tfidf_text_matrix(train, encoder, fit=True)]
        )
        x_val = np.hstack(
            [val[numeric_features].to_numpy(dtype=float), build_tfidf_text_matrix(val, encoder, fit=False)]
        )
        x_test = np.hstack(
            [test[numeric_features].to_numpy(dtype=float), build_tfidf_text_matrix(test, encoder, fit=False)]
        )

        model = RidgeBinaryClassifier(alpha=spec.alpha)
        model.fit(x_train, y_train)
        val_prob = model.predict_proba(x_val)
        test_prob = model.predict_proba(x_test)
        features = numeric_features + encoder.get_feature_names() + TEXT_AUX_FEATURES
        extra = {"text_dim": spec.text_dim, "text_alpha": spec.text_alpha}
    elif spec.kind == "late_fusion":
        numeric_features = resolve_features(spec.feature_group)
        encoder = HashingTextEncoder(n_features=spec.text_dim)

        x_train_base = train[numeric_features].to_numpy(dtype=float)
        x_val_base = val[numeric_features].to_numpy(dtype=float)
        x_test_base = test[numeric_features].to_numpy(dtype=float)
        x_train_text = build_text_matrix(train, encoder)
        x_val_text = build_text_matrix(val, encoder)
        x_test_text = build_text_matrix(test, encoder)

        base_model = RidgeBinaryClassifier(alpha=spec.alpha)
        text_model = RidgeBinaryClassifier(alpha=spec.text_alpha)
        base_model.fit(x_train_base, y_train)
        text_model.fit(x_train_text, y_train)

        val_base_prob = base_model.predict_proba(x_val_base)
        test_base_prob = base_model.predict_proba(x_test_base)
        val_text_prob = text_model.predict_proba(x_val_text)
        test_text_prob = text_model.predict_proba(x_test_text)

        fusion_weight = select_fusion_weight(
            y_true=y_val,
            future_returns=val["future_return"].to_numpy(dtype=float),
            base_prob=val_base_prob,
            text_prob=val_text_prob,
            cost_bps=transaction_cost_bps,
        )
        val_prob = (1.0 - fusion_weight) * val_base_prob + fusion_weight * val_text_prob
        test_prob = (1.0 - fusion_weight) * test_base_prob + fusion_weight * test_text_prob
        features = numeric_features + text_feature_names(spec.text_dim)
        extra = {"fusion_alpha": fusion_weight, "text_dim": spec.text_dim, "text_alpha": spec.text_alpha}
    elif spec.kind == "tfidf_late_fusion":
        numeric_features = resolve_features(spec.feature_group)
        encoder = TfidfTextEncoder(max_features=spec.text_dim)

        x_train_base = train[numeric_features].to_numpy(dtype=float)
        x_val_base = val[numeric_features].to_numpy(dtype=float)
        x_test_base = test[numeric_features].to_numpy(dtype=float)
        x_train_text = build_tfidf_text_matrix(train, encoder, fit=True)
        x_val_text = build_tfidf_text_matrix(val, encoder, fit=False)
        x_test_text = build_tfidf_text_matrix(test, encoder, fit=False)

        base_model = RidgeBinaryClassifier(alpha=spec.alpha)
        text_model = RidgeBinaryClassifier(alpha=spec.text_alpha)
        base_model.fit(x_train_base, y_train)
        text_model.fit(x_train_text, y_train)

        val_base_prob = base_model.predict_proba(x_val_base)
        test_base_prob = base_model.predict_proba(x_test_base)
        val_text_prob = text_model.predict_proba(x_val_text)
        test_text_prob = text_model.predict_proba(x_test_text)

        fusion_weight = select_fusion_weight(
            y_true=y_val,
            future_returns=val["future_return"].to_numpy(dtype=float),
            base_prob=val_base_prob,
            text_prob=val_text_prob,
            cost_bps=transaction_cost_bps,
        )
        val_prob = (1.0 - fusion_weight) * val_base_prob + fusion_weight * val_text_prob
        test_prob = (1.0 - fusion_weight) * test_base_prob + fusion_weight * test_text_prob
        features = numeric_features + encoder.get_feature_names() + TEXT_AUX_FEATURES
        extra = {"fusion_alpha": fusion_weight, "text_dim": spec.text_dim, "text_alpha": spec.text_alpha}
    else:
        features = resolve_features(spec.feature_group)
        x_train = train[features].to_numpy(dtype=float)
        x_val = val[features].to_numpy(dtype=float)
        x_test = test[features].to_numpy(dtype=float)

        model = RidgeBinaryClassifier(alpha=spec.alpha)
        model.fit(x_train, y_train)

        val_prob = model.predict_proba(x_val)
        test_prob = model.predict_proba(x_test)

    val_trade_df = aggregate_for_downstream(val, val_prob)
    test_trade_df = aggregate_for_downstream(test, test_prob)
    long_th, short_th = select_trade_thresholds(
        val_trade_df["y_prob"].to_numpy(dtype=float),
        val_trade_df["future_return"].to_numpy(dtype=float),
        cost_bps=transaction_cost_bps,
    )

    val_cls = compute_classification_metrics(y_val, val_prob)
    test_cls = compute_classification_metrics(y_test, test_prob)
    val_downstream = compute_downstream_metrics(
        future_returns=val_trade_df["future_return"].to_numpy(dtype=float),
        y_prob=val_trade_df["y_prob"].to_numpy(dtype=float),
        long_threshold=long_th,
        short_threshold=short_th,
        cost_bps=transaction_cost_bps,
    )
    test_downstream = compute_downstream_metrics(
        future_returns=test_trade_df["future_return"].to_numpy(dtype=float),
        y_prob=test_trade_df["y_prob"].to_numpy(dtype=float),
        long_threshold=long_th,
        short_threshold=short_th,
        cost_bps=transaction_cost_bps,
    )

    result = {
        "model": spec.name,
        "kind": spec.kind,
        "feature_group": spec.feature_group,
        "n_features": len(features),
        "features": features,
        "val_classification": val_cls.to_dict(),
        "test_classification": test_cls.to_dict(),
        "val_downstream": val_downstream.to_dict(),
        "test_downstream": test_downstream.to_dict(),
        "val_trade_windows": int(len(val_trade_df)),
        "test_trade_windows": int(len(test_trade_df)),
    }
    result.update(extra)
    return result


def summarize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        cls = result["test_classification"]
        down = result["test_downstream"]
        rows.append(
            {
                "model": result["model"],
                "kind": result.get("kind", "ridge"),
                "feature_group": result["feature_group"],
                "n_features": result["n_features"],
                "n_folds": result.get("n_folds", 1),
                "accuracy": cls["accuracy"],
                "accuracy_std": cls.get("accuracy_std", 0.0),
                "macro_f1": cls["macro_f1"],
                "macro_f1_std": cls.get("macro_f1_std", 0.0),
                "auc": cls["auc"],
                "auc_std": cls.get("auc_std", 0.0),
                "brier": cls["brier"],
                "ece": cls["ece"],
                "trade_rate": down["trade_rate"],
                "trade_rate_std": down.get("trade_rate_std", 0.0),
                "test_trade_windows": result.get("test_trade_windows", 0),
                "hit_rate": down["hit_rate"],
                "downstream_sharpe": down["sharpe_like"],
                "downstream_sharpe_std": down.get("sharpe_like_std", 0.0),
                "downstream_mdd": down["max_drawdown"],
                "downstream_total_return": down["total_return"],
                "long_threshold": down["long_threshold"],
                "short_threshold": down["short_threshold"],
                "cost_bps": down.get("cost_bps", 0.0),
                "gamma": result.get("gamma", 0.0),
                "fusion_alpha": result.get("fusion_alpha", 0.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline multimodal experiments")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--build-only", action="store_true", help="Build and save dataset without training models")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = parser.parse_args()

    setup_logging(verbose=not args.quiet)
    config = load_experiment_config(args.config)

    dataset_config = MultimodalDatasetConfig.from_dict(config["dataset"])
    builder = MultimodalDatasetBuilder(dataset_config)
    dataset, metadata = builder.build()
    saved_paths = builder.save(dataset, metadata)
    logger.info("Saved dataset to %s", saved_paths["dataset"])

    if args.build_only:
        print(json.dumps({"dataset": saved_paths, "metadata": metadata}, indent=2))
        return

    experiment_name = config.get("name", "mm_experiment")
    output_dir = config.get("output_dir", "experiments/results_mm")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    split_mode = config.get("split_mode", "single")
    transaction_cost_bps = float(config.get("transaction_cost_bps", 0.0))
    if split_mode == "rolling":
        rolling_cfg = config.get("rolling", {})
        split_sets = build_rolling_splits(
            dataset,
            min_train_windows=int(rolling_cfg.get("min_train_windows", 40)),
            val_windows=int(rolling_cfg.get("val_windows", 16)),
            test_windows=int(rolling_cfg.get("test_windows", 12)),
            step_windows=int(rolling_cfg.get("step_windows", rolling_cfg.get("test_windows", 12))),
        )
    else:
        split_sets = [
            split_dataset(
                dataset,
                train_ratio=float(config.get("train_ratio", 0.6)),
                val_ratio=float(config.get("val_ratio", 0.2)),
            )
        ]

    model_specs = [
        ModelSpec(
            name=model_cfg["name"],
            feature_group=model_cfg["feature_group"],
            kind=model_cfg.get("kind", "ridge"),
            alpha=float(model_cfg.get("alpha", 1.0)),
            web_alpha=float(model_cfg.get("web_alpha", 3.0)),
            text_alpha=float(model_cfg.get("text_alpha", 1.0)),
            text_dim=int(model_cfg.get("text_dim", 256)),
        )
        for model_cfg in config.get(
            "models",
            [
                {"name": "price_only_linear", "feature_group": "price"},
                {"name": "web_only_linear", "feature_group": "web"},
                {"name": "multimodal_linear", "feature_group": "all"},
            ],
        )
    ]

    results = []
    for spec in model_specs:
        fold_results = [run_model(spec, splits, transaction_cost_bps=transaction_cost_bps) for splits in split_sets]
        if len(fold_results) == 1:
            result = fold_results[0]
            result["n_folds"] = 1
            result["fold_results"] = fold_results
        else:
            result = aggregate_model_results(fold_results)
        results.append(result)
    summary_df = summarize_results(results)

    result_payload = {
        "name": experiment_name,
        "config": config,
        "dataset_metadata": metadata,
        "saved_dataset": saved_paths,
        "split_mode": split_mode,
        "transaction_cost_bps": transaction_cost_bps,
        "split_sizes": [
            {"fold": idx, **{name: int(len(split_df)) for name, split_df in splits.items()}}
            for idx, splits in enumerate(split_sets)
        ],
        "results": results,
    }

    summary_path = os.path.join(run_dir, "summary.csv")
    results_path = os.path.join(run_dir, "results.json")
    metadata_path = os.path.join(run_dir, "dataset_metadata.json")

    summary_df.to_csv(summary_path, index=False)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved experiment outputs to %s", run_dir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
