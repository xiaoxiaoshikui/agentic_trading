#!/usr/bin/env python
"""
Run ACM MM style deep multimodal experiments with price sequences and raw text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from experiments.mm_dataset import MultimodalDatasetConfig, TEXT_AUX_FEATURES, WEB_FEATURES
from experiments.mm_deep_dataset import DeepMultimodalDatasetBuilder
from experiments.mm_deep_models import (
    BiLSTMFusion,
    ConditionallyGatedCrossModalFusion,
    CrossModalAttentionFusion,
    DLinearPriceClassifier,
    DualHeadConditionallyGatedCrossModalFusion,
    DualHeadTimesNetGatedCrossModalFusion,
    DualBranchConditionalLateFusion,
    EarlyFusionTransformer,
    GatedLateFusionModel,
    ITransformerPriceClassifier,
    LearnedGateCrossModalFusion,
    MultimodalTransformer,
    PatchTSTPriceClassifier,
    PriceWebLateFusion,
    PriceSequenceTransformer,
    SourceAwareQualityConditionalLateFusion,
    StaleInterventionCrossModalFusion,
    StalenessAwareCrossModalFusion,
    TensorFusionNetwork,
    TextEmbeddingMLP,
    TimesNetGatedCrossModalFusion,
    TimesNetLitePriceClassifier,
    TimesNetMixtureOfExpertsFusion,
)
from experiments.mm_metrics import (
    compute_classification_metrics,
    compute_downstream_metrics,
    compute_position_downstream_metrics,
)
from experiments.run_mm_experiment import (
    aggregate_for_downstream,
    aggregate_model_results,
    build_rolling_splits,
    select_trade_thresholds,
    split_dataset,
    summarize_results,
)

logger = logging.getLogger(__name__)

DIRECT_POSITION_KINDS = {
    "conditionally_gated_cross_modal_dual_head",
    "timesnet_gated_cross_modal_dual_head",
}

DOWNSTREAM_SELECTION_METRICS = {
    "downstream_sharpe",
    "cost_sharpe",
    "val_sharpe",
    "downstream_total_return",
    "val_total_return",
    "position_sharpe",
    "direct_position_sharpe",
    "val_position_sharpe",
    "position_total_return",
    "direct_position_total_return",
    "val_position_total_return",
}


@dataclass
class DeepModelSpec:
    name: str
    kind: str
    d_model: int = 96
    n_heads: int = 4
    price_layers: int = 2
    fusion_layers: int = 1
    hidden_dim: int = 192
    dropout: float = 0.1
    lag_decay: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 12
    patience: int = 4
    selection_metric: str = "auc"
    use_mixup: bool = False
    mixup_alpha: float = 0.4
    train_lag_min: float = 0.0
    train_lag_max: float = float("inf")
    # --- new training enhancements ---
    use_hybrid_text: bool = False        # concat fold-level TF-IDF features to text embeddings
    hybrid_tfidf_features: int = 50     # vocabulary size for TF-IDF extension
    lr_schedule: str = "none"           # "none" | "cosine"
    use_contrastive: bool = False        # add InfoNCE auxiliary loss between price/text reprs
    contrastive_weight: float = 0.1     # weight of the contrastive term
    contrastive_temperature: float = 0.07
    stale_threshold_hours: float = 1.5  # hard null replacement threshold for stale_intervention kind
    # --- P2: quality gate ---
    use_quality_gate: bool = False      # web-richness gate on text_context (33 extra params)
    # --- P4: pairwise ranking loss ---
    use_ranking_loss: bool = False      # add margin ranking loss alongside BCE
    ranking_margin: float = 0.1        # margin for pairwise ranking
    ranking_weight: float = 0.3        # weight of ranking term relative to BCE
    use_return_weighted_loss: bool = False
    return_weight_alpha: float = 200.0
    return_weight_clip: float = 5.0
    use_utility_loss: bool = False
    utility_weight: float = 0.1
    utility_mode: str = "mean"          # "mean" | "sharpe"
    utility_return_scale: float = 1000.0
    utility_warmup_epochs: int = 0
    position_temperature: float = 1.0
    use_direct_position: bool = False   # use a separate position head for trading
    direct_position_threshold: float = 0.05
    position_l1_weight: float = 0.0
    position_target_weight: float = 0.0
    position_target_threshold_bps: float = -1.0  # negative means use transaction_cost_bps
    # --- P5: two-phase text freeze ---
    freeze_text_epochs: int = 0        # freeze text_proj + text_to_price for first N epochs
    text_base_dim: int = 384           # dense semantic embedding width before optional TF-IDF append
    # --- Diversity-aware fold filter ---
    min_fold_text_diversity: float = 0.0  # min std of news_direction_score in train window; 0=disabled
    context_feature_set: str = "web"   # "web" | "web_text_aux"
    use_cross_attention: bool = True
    gate_mode: str = "vector"          # "vector" | "scalar" | "none"
    use_web_gate_feature: bool = True
    use_lag_gate_feature: bool = True
    export_gate_analysis: bool = False
    patch_len: int = 16
    stride: int = 8
    max_len: int = 256
    shuffle_lag: bool = False
    shuffle_text: bool = False
    shuffle_web: bool = False


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).strip().lower()
    valid = {"auto", "cpu", "mps", "cuda"}
    if requested not in valid:
        raise ValueError(f"Unsupported device '{requested}'. Choose one of {sorted(valid)}")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is false")
    return torch.device(requested)


def dataset_symbol_tag(dataset_config: MultimodalDatasetConfig) -> str:
    if dataset_config.symbols:
        symbols = [str(symbol).strip() for symbol in dataset_config.symbols if str(symbol).strip()]
        if symbols:
            return "_".join(symbols)
    return dataset_config.symbol


def cache_text_embeddings(
    texts: List[str],
    model_name: str,
    cache_dir: str,
    dataset_key: str,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{sanitize_name(dataset_key)}__{sanitize_name(model_name)}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.save(cache_path, embeddings)
    return embeddings


def standardize_sequences(
    train_seq: np.ndarray,
    val_seq: np.ndarray,
    test_seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_seq.mean(axis=(0, 1), keepdims=True)
    std = train_seq.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_seq - mean) / std, (val_seq - mean) / std, (test_seq - mean) / std


def standardize_matrix(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def build_model(spec: DeepModelSpec, price_dim: int, text_dim: int, web_dim: int) -> nn.Module:
    if spec.kind == "price_sequence_transformer":
        return PriceSequenceTransformer(
            price_dim=price_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            num_layers=spec.price_layers,
            dropout=spec.dropout,
        )
    if spec.kind == "patchtst_price":
        return PatchTSTPriceClassifier(
            price_dim=price_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            num_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            patch_len=spec.patch_len,
            stride=spec.stride,
        )
    if spec.kind == "itransformer_price":
        return ITransformerPriceClassifier(
            price_dim=price_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            num_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            max_len=spec.max_len,
        )
    if spec.kind == "dlinear_price":
        return DLinearPriceClassifier(
            price_dim=price_dim,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            max_len=spec.max_len,
        )
    if spec.kind == "timesnet_lite_price":
        return TimesNetLitePriceClassifier(
            price_dim=price_dim,
            d_model=spec.d_model,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "timesnet_gated_cross_modal":
        return TimesNetGatedCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            gate_mode=spec.gate_mode,
            use_web_gate_feature=spec.use_web_gate_feature,
            use_lag_gate_feature=spec.use_lag_gate_feature,
        )
    if spec.kind == "timesnet_gated_cross_modal_dual_head":
        return DualHeadTimesNetGatedCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            gate_mode=spec.gate_mode,
            use_web_gate_feature=spec.use_web_gate_feature,
            use_lag_gate_feature=spec.use_lag_gate_feature,
        )
    if spec.kind == "timesnet_moe_fusion":
        return TimesNetMixtureOfExpertsFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "text_embedding_mlp":
        return TextEmbeddingMLP(text_dim=text_dim, web_dim=web_dim, hidden_dim=spec.hidden_dim, dropout=spec.dropout)
    if spec.kind == "price_web_late_fusion":
        return PriceWebLateFusion(
            price_dim=price_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "cross_modal_attention":
        return CrossModalAttentionFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            fusion_layers=spec.fusion_layers,
            dropout=spec.dropout,
        )
    if spec.kind == "gated_late_fusion":
        return GatedLateFusionModel(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "dual_branch_conditional_late_fusion":
        return DualBranchConditionalLateFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            dense_text_dim=spec.text_base_dim,
        )
    if spec.kind == "source_aware_quality_conditional_late_fusion":
        return SourceAwareQualityConditionalLateFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            dense_text_dim=spec.text_base_dim,
        )
    if spec.kind == "bilstm_fusion":
        return BiLSTMFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "multimodal_transformer":
        return MultimodalTransformer(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "early_fusion":
        return EarlyFusionTransformer(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "conditionally_gated_cross_modal":
        return ConditionallyGatedCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            use_cross_attention=spec.use_cross_attention,
            gate_mode=spec.gate_mode,
            use_web_gate_feature=spec.use_web_gate_feature,
            use_lag_gate_feature=spec.use_lag_gate_feature,
        )
    if spec.kind == "conditionally_gated_cross_modal_dual_head":
        return DualHeadConditionallyGatedCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
            use_cross_attention=spec.use_cross_attention,
            gate_mode=spec.gate_mode,
            use_web_gate_feature=spec.use_web_gate_feature,
            use_lag_gate_feature=spec.use_lag_gate_feature,
        )
    if spec.kind == "tensor_fusion":
        return TensorFusionNetwork(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            hidden_dim=spec.hidden_dim,
            dropout=spec.dropout,
        )
    if spec.kind == "staleness_aware_cross_modal":
        return StalenessAwareCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            fusion_layers=spec.fusion_layers,
            dropout=spec.dropout,
            lag_decay=spec.lag_decay,
        )
    if spec.kind == "learned_gate_cross_modal":
        return LearnedGateCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            fusion_layers=spec.fusion_layers,
            dropout=spec.dropout,
            init_lag_decay=spec.lag_decay,
            use_quality_gate=spec.use_quality_gate,
        )
    if spec.kind == "stale_intervention":
        return StaleInterventionCrossModalFusion(
            price_dim=price_dim,
            text_dim=text_dim,
            web_dim=web_dim,
            d_model=spec.d_model,
            n_heads=spec.n_heads,
            price_layers=spec.price_layers,
            fusion_layers=spec.fusion_layers,
            dropout=spec.dropout,
            init_lag_decay=spec.lag_decay,
            stale_threshold_hours=spec.stale_threshold_hours,
        )
    raise ValueError(f"Unknown deep model kind: {spec.kind}")


def contrastive_nce_loss(
    h_price: torch.Tensor,
    h_text: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between price and text representations (NT-Xent).

    Treats each (price_i, text_i) in-batch pair as a positive, all
    cross-sample pairs as negatives.  Both representations are L2-normalised
    before computing cosine similarity.
    """
    h_p = F.normalize(h_price, dim=-1)   # (B, d)
    h_t = F.normalize(h_text, dim=-1)    # (B, d)
    logits = h_p @ h_t.T / temperature  # (B, B)
    labels = torch.arange(len(h_p), device=h_p.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
    return loss


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Pairwise margin ranking loss for binary (or soft) labels.

    For every pair (i, j) where label_i > label_j by at least 0.2, penalises
    cases where logit_i is not at least ``margin`` larger than logit_j.
    Complements BCE by directly optimising the trading-relevant ranking.
    """
    if logits.size(0) < 2:
        return torch.tensor(0.0, device=logits.device)
    diff_logit = logits.unsqueeze(1) - logits.unsqueeze(0)   # (B, B)
    diff_label = labels.unsqueeze(1) - labels.unsqueeze(0)   # (B, B)
    # Only consider pairs where labels clearly differ
    significant = diff_label.abs() > 0.2
    direction = diff_label.sign()                             # +1 or -1
    pair_loss = torch.relu(margin - direction * diff_logit)
    pair_loss = pair_loss[significant]
    return pair_loss.mean() if pair_loss.numel() > 0 else torch.tensor(0.0, device=logits.device)


def return_weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    future_returns: torch.Tensor,
    pos_weight: torch.Tensor,
    alpha: float = 200.0,
    weight_clip: float = 5.0,
) -> torch.Tensor:
    per_item = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    weights = 1.0 + float(alpha) * future_returns.abs()
    weights = torch.clamp(weights, min=1.0, max=max(float(weight_clip), 1.0))
    return torch.mean(per_item * weights)


def soft_position_utility_loss(
    logits: torch.Tensor,
    future_returns: torch.Tensor,
    cost_bps: float,
    temperature: float = 1.0,
    mode: str = "mean",
    return_scale: float = 1000.0,
) -> torch.Tensor:
    temperature = max(float(temperature), 1e-3)
    positions = torch.tanh(logits / temperature)
    cost = max(float(cost_bps), 0.0) * 1e-4
    net_pnl = positions * future_returns - positions.abs() * cost
    mode = str(mode).strip().lower()
    if mode == "mean":
        return -net_pnl.mean() * float(return_scale)
    if mode == "sharpe":
        pnl_std = net_pnl.std(unbiased=False).clamp_min(1e-6)
        return -(net_pnl.mean() / pnl_std)
    raise ValueError(f"Unknown utility_mode: {mode}")


def logits_to_positions(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-3)
    return torch.tanh(logits / temperature)


def direct_position_target_loss(
    position_logits: torch.Tensor,
    future_returns: torch.Tensor,
    cost_bps: float,
    temperature: float = 1.0,
    target_threshold_bps: float = -1.0,
) -> torch.Tensor:
    threshold_bps = float(target_threshold_bps)
    if threshold_bps < 0.0:
        threshold_bps = max(float(cost_bps), 0.0)
    threshold = threshold_bps * 1e-4
    target = torch.zeros_like(future_returns)
    target = torch.where(future_returns > threshold, torch.ones_like(target), target)
    target = torch.where(future_returns < -threshold, -torch.ones_like(target), target)
    positions = logits_to_positions(position_logits, temperature)
    return F.smooth_l1_loss(positions, target)


def unpack_model_forward(
    fwd: torch.Tensor | tuple,
    use_direct_position: bool = False,
    use_contrastive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    position_logits = None
    price_repr = None
    text_repr = None
    if use_direct_position:
        if not isinstance(fwd, tuple) or len(fwd) < 2:
            raise ValueError("Direct-position models must return (direction_logits, position_logits, ...)")
        logits = fwd[0]
        position_logits = fwd[1]
        if use_contrastive and len(fwd) >= 4:
            price_repr = fwd[2]
            text_repr = fwd[3]
        return logits, position_logits, price_repr, text_repr

    if use_contrastive and isinstance(fwd, tuple):
        logits = fwd[0]
        if len(fwd) >= 3:
            price_repr = fwd[1]
            text_repr = fwd[2]
        return logits, None, price_repr, text_repr
    return fwd, None, None, None


def model_forward(
    model: nn.Module,
    kind: str,
    price_seq: torch.Tensor,
    text_emb: torch.Tensor,
    web_feat: torch.Tensor,
    lag_minutes: torch.Tensor | None = None,
    return_embeddings: bool = False,
    return_gate: bool = False,
    return_position: bool = False,
) -> torch.Tensor | tuple:
    if kind in (
        "price_sequence_transformer",
        "patchtst_price",
        "itransformer_price",
        "dlinear_price",
        "timesnet_lite_price",
    ):
        logits = model(price_seq)
        return (logits, None, None) if return_embeddings else logits
    if kind == "text_embedding_mlp":
        logits = model(text_emb, web_feat)
        return (logits, None, None) if return_embeddings else logits
    if kind == "price_web_late_fusion":
        return model(price_seq, text_emb, web_feat, lag_minutes, return_embeddings)
    if kind in ("cross_modal_attention", "gated_late_fusion"):
        logits = model(price_seq, text_emb, web_feat)
        return (logits, None, None) if return_embeddings else logits
    if kind in ("timesnet_gated_cross_modal", "timesnet_moe_fusion"):
        return model(
            price_seq,
            text_emb,
            web_feat,
            lag_minutes,
            return_embeddings=return_embeddings,
            return_gate=return_gate,
        )
    if kind == "timesnet_gated_cross_modal_dual_head":
        return model(
            price_seq,
            text_emb,
            web_feat,
            lag_minutes,
            return_embeddings=return_embeddings,
            return_gate=return_gate,
            return_position=return_position,
        )
    if kind in ("dual_branch_conditional_late_fusion", "conditionally_gated_cross_modal",
                "early_fusion", "bilstm_fusion", "multimodal_transformer", "tensor_fusion"):
        if kind == "conditionally_gated_cross_modal":
            return model(
                price_seq,
                text_emb,
                web_feat,
                lag_minutes,
                return_embeddings=return_embeddings,
                return_gate=return_gate,
            )
        return model(price_seq, text_emb, web_feat, lag_minutes, return_embeddings)
    if kind == "conditionally_gated_cross_modal_dual_head":
        return model(
            price_seq,
            text_emb,
            web_feat,
            lag_minutes,
            return_embeddings=return_embeddings,
            return_gate=return_gate,
            return_position=return_position,
        )
    if kind == "source_aware_quality_conditional_late_fusion":
        return model(price_seq, text_emb, web_feat, lag_minutes, return_embeddings)
    if kind in ("staleness_aware_cross_modal", "learned_gate_cross_modal", "stale_intervention"):
        return model(price_seq, text_emb, web_feat, lag_minutes, return_embeddings)
    raise ValueError(f"Unknown deep model kind: {kind}")


def compute_selection_score(metric_name: str, cls_metrics: Any, downstream_metrics: Any | None = None) -> float:
    metric = str(metric_name).strip().lower()
    if metric == "auc":
        return float(cls_metrics.auc)
    if metric == "macro_f1":
        return float(cls_metrics.macro_f1)
    if metric == "accuracy":
        return float(cls_metrics.accuracy)
    if metric == "neg_brier":
        return -float(cls_metrics.brier)
    if metric == "auc_macro_f1":
        return float(cls_metrics.auc) + 0.05 * float(cls_metrics.macro_f1)
    if metric == "legacy":
        return float(cls_metrics.auc) + 0.05 * float(cls_metrics.macro_f1)
    if metric in {
        "downstream_sharpe",
        "cost_sharpe",
        "val_sharpe",
        "position_sharpe",
        "direct_position_sharpe",
        "val_position_sharpe",
    }:
        if downstream_metrics is None:
            raise ValueError(f"{metric_name} requires downstream validation metrics")
        return float(downstream_metrics.sharpe_like)
    if metric in {
        "downstream_total_return",
        "val_total_return",
        "position_total_return",
        "direct_position_total_return",
        "val_position_total_return",
    }:
        if downstream_metrics is None:
            raise ValueError(f"{metric_name} requires downstream validation metrics")
        return float(downstream_metrics.total_return)
    raise ValueError(f"Unknown selection_metric: {metric_name}")


def mixup_batch(
    price: torch.Tensor,
    text: torch.Tensor,
    web: torch.Tensor,
    labels: torch.Tensor,
    lag: torch.Tensor,
    future_returns: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation to a training batch (Yun et al., 2019).

    Samples λ ~ Beta(α, α) and returns convex interpolation of each tensor
    with a randomly permuted copy.  Labels become soft targets, which
    are compatible with BCEWithLogitsLoss.  Lag is also interpolated so the
    staleness gate receives a consistent mixed input.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(price.size(0), device=price.device)
    return (
        lam * price + (1.0 - lam) * price[idx],
        lam * text + (1.0 - lam) * text[idx],
        lam * web + (1.0 - lam) * web[idx],
        lam * labels + (1.0 - lam) * labels[idx],
        lam * lag + (1.0 - lam) * lag[idx],
        lam * future_returns + (1.0 - lam) * future_returns[idx],
    )


def make_loader(
    price_seq: np.ndarray,
    text_emb: np.ndarray,
    web_feat: np.ndarray,
    labels: np.ndarray,
    lag_minutes: np.ndarray,
    batch_size: int,
    shuffle: bool,
    future_returns: np.ndarray | None = None,
) -> TorchDataLoader:
    if future_returns is None:
        future_returns = np.zeros(len(price_seq), dtype=np.float32)
    dataset = TensorDataset(
        torch.tensor(price_seq, dtype=torch.float32),
        torch.tensor(text_emb, dtype=torch.float32),
        torch.tensor(web_feat, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(lag_minutes.astype(np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(future_returns, dtype=np.float32), dtype=torch.float32),
    )
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def apply_split_shuffle(
    spec: DeepModelSpec,
    train_text: np.ndarray,
    val_text: np.ndarray,
    test_text: np.ndarray,
    train_web: np.ndarray,
    val_web: np.ndarray,
    test_web: np.ndarray,
    train_lag: np.ndarray,
    val_lag: np.ndarray,
    test_lag: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply within-split modality shuffles for causal sanity-check ablations."""
    if spec.shuffle_text:
        train_text = train_text[np.random.permutation(len(train_text))]
        val_text = val_text[np.random.permutation(len(val_text))]
        test_text = test_text[np.random.permutation(len(test_text))]
    if spec.shuffle_web:
        train_web = train_web[np.random.permutation(len(train_web))]
        val_web = val_web[np.random.permutation(len(val_web))]
        test_web = test_web[np.random.permutation(len(test_web))]
    if spec.shuffle_lag:
        train_lag = train_lag[np.random.permutation(len(train_lag))]
        val_lag = val_lag[np.random.permutation(len(val_lag))]
        test_lag = test_lag[np.random.permutation(len(test_lag))]
    return train_text, val_text, test_text, train_web, val_web, test_web, train_lag, val_lag, test_lag


def build_trade_prediction_rows(
    split_df: pd.DataFrame,
    y_prob: np.ndarray,
    split_name: str,
    model_name: str,
    fold_id: int,
    lag_minutes: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    group_cols = ["window_end"]
    if "symbol" in split_df.columns:
        group_cols = ["symbol", "window_end"]
    row_df = split_df[group_cols + ["future_return"]].copy()
    row_df["y_prob"] = np.asarray(y_prob, dtype=float)
    row_df["lag_minutes"] = np.asarray(lag_minutes, dtype=float)
    agg = (
        row_df.groupby(group_cols, as_index=False)
        .agg({"future_return": "last", "y_prob": "mean", "lag_minutes": "mean"})
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    if "symbol" not in agg.columns:
        agg["symbol"] = ""
    agg["split"] = split_name
    agg["model"] = model_name
    agg["fold_id"] = int(fold_id)
    agg["long_threshold"] = float(long_threshold)
    agg["short_threshold"] = float(short_threshold)
    agg["position"] = 0.0
    agg.loc[agg["y_prob"] >= long_threshold, "position"] = 1.0
    agg.loc[agg["y_prob"] <= short_threshold, "position"] = -1.0
    agg["cost_bps"] = float(cost_bps)
    agg["net_pnl"] = agg["position"] * agg["future_return"] - agg["position"].abs() * float(cost_bps) * 1e-4
    cols = [
        "split",
        "fold_id",
        "model",
        "symbol",
        "window_end",
        "future_return",
        "y_prob",
        "long_threshold",
        "short_threshold",
        "cost_bps",
        "position",
        "net_pnl",
        "lag_minutes",
    ]
    return agg[cols]


def aggregate_position_for_downstream(
    split_df: pd.DataFrame,
    y_prob: np.ndarray,
    positions: np.ndarray,
    lag_minutes: np.ndarray | None = None,
) -> pd.DataFrame:
    group_cols = ["window_end"]
    if "symbol" in split_df.columns:
        group_cols = ["symbol", "window_end"]
    downstream_df = split_df[group_cols + ["future_return"]].copy()
    downstream_df["y_prob"] = np.asarray(y_prob, dtype=float)
    downstream_df["position"] = np.clip(np.asarray(positions, dtype=float), -1.0, 1.0)
    agg_map = {"future_return": "last", "y_prob": "mean", "position": "mean"}
    if lag_minutes is not None:
        downstream_df["lag_minutes"] = np.asarray(lag_minutes, dtype=float)
        agg_map["lag_minutes"] = "mean"
    return (
        downstream_df.sort_values(group_cols)
        .groupby(group_cols, as_index=False)
        .agg(agg_map)
        .sort_values(group_cols)
        .reset_index(drop=True)
    )


def build_position_trade_prediction_rows(
    split_df: pd.DataFrame,
    y_prob: np.ndarray,
    positions: np.ndarray,
    split_name: str,
    model_name: str,
    fold_id: int,
    lag_minutes: np.ndarray,
    min_abs_position: float,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    agg = aggregate_position_for_downstream(
        split_df=split_df,
        y_prob=y_prob,
        positions=positions,
        lag_minutes=lag_minutes,
    )
    if "symbol" not in agg.columns:
        agg["symbol"] = ""
    min_abs_position = max(float(min_abs_position), 0.0)
    agg["split"] = split_name
    agg["model"] = model_name
    agg["fold_id"] = int(fold_id)
    agg["long_threshold"] = min_abs_position
    agg["short_threshold"] = -min_abs_position
    agg["raw_position"] = np.clip(agg["position"].to_numpy(dtype=float), -1.0, 1.0)
    agg["position"] = np.where(np.abs(agg["raw_position"]) >= min_abs_position, agg["raw_position"], 0.0)
    agg["cost_bps"] = float(cost_bps)
    agg["net_pnl"] = agg["position"] * agg["future_return"] - agg["position"].abs() * float(cost_bps) * 1e-4
    cols = [
        "split",
        "fold_id",
        "model",
        "symbol",
        "window_end",
        "future_return",
        "y_prob",
        "long_threshold",
        "short_threshold",
        "cost_bps",
        "raw_position",
        "position",
        "net_pnl",
        "lag_minutes",
    ]
    return agg[cols]


def predict_probs(
    model: nn.Module,
    kind: str,
    price_seq: np.ndarray,
    text_emb: np.ndarray,
    web_feat: np.ndarray,
    lag_minutes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    loader = make_loader(
        price_seq=price_seq,
        text_emb=text_emb,
        web_feat=web_feat,
        labels=np.zeros(len(price_seq), dtype=np.float32),
        lag_minutes=lag_minutes,
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_price, batch_text, batch_web, _, batch_lag, _ in loader:
            logits = model_forward(
                model,
                kind=kind,
                price_seq=batch_price.to(device),
                text_emb=batch_text.to(device),
                web_feat=batch_web.to(device),
                lag_minutes=batch_lag.to(device),
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def predict_probs_and_positions(
    model: nn.Module,
    kind: str,
    price_seq: np.ndarray,
    text_emb: np.ndarray,
    web_feat: np.ndarray,
    lag_minutes: np.ndarray,
    device: torch.device,
    batch_size: int,
    position_temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    loader = make_loader(
        price_seq=price_seq,
        text_emb=text_emb,
        web_feat=web_feat,
        labels=np.zeros(len(price_seq), dtype=np.float32),
        lag_minutes=lag_minutes,
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_price, batch_text, batch_web, _, batch_lag, _ in loader:
            fwd = model_forward(
                model,
                kind=kind,
                price_seq=batch_price.to(device),
                text_emb=batch_text.to(device),
                web_feat=batch_web.to(device),
                lag_minutes=batch_lag.to(device),
                return_position=True,
            )
            logits, position_logits, _, _ = unpack_model_forward(
                fwd,
                use_direct_position=True,
                use_contrastive=False,
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
            positions.append(logits_to_positions(position_logits, position_temperature).cpu().numpy())
    return np.concatenate(probs, axis=0), np.concatenate(positions, axis=0)


def collect_gate_analysis(
    model: nn.Module,
    kind: str,
    price_seq: np.ndarray,
    text_emb: np.ndarray,
    web_feat: np.ndarray,
    lag_minutes: np.ndarray,
    split_df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    if kind != "conditionally_gated_cross_modal":
        raise ValueError(f"Gate analysis is only supported for CGCMA, got {kind}")
    model.eval()
    rows: List[pd.DataFrame] = []
    loader = make_loader(
        price_seq=price_seq,
        text_emb=text_emb,
        web_feat=web_feat,
        labels=np.zeros(len(price_seq), dtype=np.float32),
        lag_minutes=lag_minutes,
        batch_size=batch_size,
        shuffle=False,
    )
    offset = 0
    with torch.no_grad():
        for batch_price, batch_text, batch_web, _, batch_lag, _ in loader:
            logits, h_price, h_text_ctx, gate = model_forward(
                model,
                kind=kind,
                price_seq=batch_price.to(device),
                text_emb=batch_text.to(device),
                web_feat=batch_web.to(device),
                lag_minutes=batch_lag.to(device),
                return_gate=True,
            )
            n = int(batch_price.size(0))
            part = split_df.iloc[offset : offset + n].copy().reset_index(drop=True)
            part["lag_minutes"] = batch_lag.cpu().numpy()
            part["y_prob"] = torch.sigmoid(logits).cpu().numpy()
            part["gate_mean"] = gate.mean(dim=-1).cpu().numpy()
            part["gate_std"] = gate.std(dim=-1).cpu().numpy()
            part["h_price_norm"] = torch.norm(h_price, dim=-1).cpu().numpy()
            part["h_context_norm"] = torch.norm(h_text_ctx, dim=-1).cpu().numpy()
            rows.append(part)
            offset += n
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def train_one_fold(
    spec: DeepModelSpec,
    split: Dict[str, pd.DataFrame],
    price_sequences: np.ndarray,
    text_embeddings: np.ndarray,
    web_features: np.ndarray,
    lag_features: np.ndarray,
    device: torch.device,
    text_blobs: np.ndarray | None = None,
    fold_id: int = 0,
    export_trade_predictions: bool = False,
    transaction_cost_bps: float = 0.0,
) -> Dict[str, Any]:
    train_idx = split["train"]["row_id"].to_numpy(dtype=int)
    val_idx = split["val"]["row_id"].to_numpy(dtype=int)
    test_idx = split["test"]["row_id"].to_numpy(dtype=int)

    train_seq, val_seq, test_seq = standardize_sequences(
        price_sequences[train_idx],
        price_sequences[val_idx],
        price_sequences[test_idx],
    )
    train_text, val_text, test_text = standardize_matrix(
        text_embeddings[train_idx],
        text_embeddings[val_idx],
        text_embeddings[test_idx],
    )
    train_web, val_web, test_web = standardize_matrix(
        web_features[train_idx],
        web_features[val_idx],
        web_features[test_idx],
    )
    train_lag = lag_features[train_idx]
    val_lag = lag_features[val_idx]
    test_lag = lag_features[test_idx]

    # Lag-stratified training: filter training samples to the specified lag window.
    # Validation and test sets are always evaluated on all lags (realistic deployment).
    if spec.train_lag_min > 0.0 or spec.train_lag_max < float("inf"):
        lag_mask = (train_lag >= spec.train_lag_min) & (train_lag < spec.train_lag_max)
        n_before = len(train_idx)
        if lag_mask.sum() == 0:
            # No samples in target window for this fold — fall back to full range
            logger.debug(
                "Lag filter [%.0f, %.0f): 0 / %d samples; falling back to full range",
                spec.train_lag_min, spec.train_lag_max, n_before,
            )
            lag_mask = np.ones(len(train_idx), dtype=bool)
            train_idx_filtered = train_idx
        else:
            train_seq = train_seq[lag_mask]
            train_text = train_text[lag_mask]
            train_web = train_web[lag_mask]
            train_lag = train_lag[lag_mask]
            train_idx_filtered = train_idx[lag_mask]
            logger.debug(
                "Lag filter [%.0f, %.0f): kept %d / %d training samples",
                spec.train_lag_min, spec.train_lag_max, lag_mask.sum(), n_before,
            )
    else:
        train_idx_filtered = train_idx

    y_train = split["train"]["label_up"].to_numpy(dtype=np.float32)
    train_future_returns = split["train"]["future_return"].to_numpy(dtype=np.float32)
    # Re-align y_train with potentially filtered training set
    if spec.train_lag_min > 0.0 or spec.train_lag_max < float("inf"):
        y_train = y_train[lag_mask]
        train_future_returns = train_future_returns[lag_mask]
    y_val = split["val"]["label_up"].to_numpy(dtype=np.int32)
    y_test = split["test"]["label_up"].to_numpy(dtype=np.int32)

    # --- Hybrid text: append fold-level TF-IDF features to sentence embeddings ---
    if spec.use_hybrid_text and text_blobs is not None:
        tfidf_vect = TfidfVectorizer(
            max_features=spec.hybrid_tfidf_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
        )
        train_texts = [str(text_blobs[i]) for i in train_idx_filtered]
        tfidf_vect.fit(train_texts)
        tfidf_tr = tfidf_vect.transform(train_texts).toarray().astype(np.float32)
        tfidf_va = tfidf_vect.transform(
            [str(text_blobs[i]) for i in val_idx]
        ).toarray().astype(np.float32)
        tfidf_te = tfidf_vect.transform(
            [str(text_blobs[i]) for i in test_idx]
        ).toarray().astype(np.float32)
        tfidf_tr, tfidf_va, tfidf_te = standardize_matrix(tfidf_tr, tfidf_va, tfidf_te)
        train_text = np.concatenate([train_text, tfidf_tr], axis=1)
        val_text = np.concatenate([val_text, tfidf_va], axis=1)
        test_text = np.concatenate([test_text, tfidf_te], axis=1)

    (
        train_text,
        val_text,
        test_text,
        train_web,
        val_web,
        test_web,
        train_lag,
        val_lag,
        test_lag,
    ) = apply_split_shuffle(
        spec,
        train_text,
        val_text,
        test_text,
        train_web,
        val_web,
        test_web,
        train_lag,
        val_lag,
        test_lag,
    )

    if spec.use_direct_position and spec.kind not in DIRECT_POSITION_KINDS:
        raise ValueError(f"use_direct_position=true requires one of {sorted(DIRECT_POSITION_KINDS)}, got {spec.kind}")

    model = build_model(
        spec,
        price_dim=int(train_seq.shape[-1]),
        text_dim=int(train_text.shape[-1]),
        web_dim=int(train_web.shape[-1]),
    ).to(device)

    train_loader = make_loader(
        train_seq,
        train_text,
        train_web,
        y_train,
        train_lag,
        batch_size=spec.batch_size,
        shuffle=True,
        future_returns=train_future_returns,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)

    # --- Cosine LR schedule with 2-epoch linear warmup ---
    if spec.lr_schedule == "cosine":
        _warmup = 2
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, spec.epochs - _warmup), eta_min=spec.lr * 0.05
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[_warmup]
        )
    else:
        lr_scheduler = None

    pos_count = max(1.0, float(np.sum(y_train)))
    neg_count = max(1.0, float(len(y_train) - np.sum(y_train)))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state: Dict[str, Any] | None = None
    best_score = -1e9
    patience_left = spec.patience
    # Names of text-branch parameters to freeze during P5 warm-up phase
    _text_param_keys = ("text_proj", "text_to_price")
    for _epoch in range(spec.epochs):
        # --- P5: two-phase text freeze ---
        if spec.freeze_text_epochs > 0:
            _freeze = _epoch < spec.freeze_text_epochs
            for _name, _param in model.named_parameters():
                if any(k in _name for k in _text_param_keys):
                    _param.requires_grad_(not _freeze)

        model.train()
        for batch_price, batch_text, batch_web, batch_labels, batch_lag, batch_returns in train_loader:
            optimizer.zero_grad(set_to_none=True)
            bp = batch_price.to(device)
            bt = batch_text.to(device)
            bw = batch_web.to(device)
            bl = batch_labels.to(device)
            blag = batch_lag.to(device)
            br = batch_returns.to(device)
            if spec.use_mixup and bp.size(0) > 1:
                bp, bt, bw, bl, blag, br = mixup_batch(bp, bt, bw, bl, blag, br, alpha=spec.mixup_alpha)
            fwd = model_forward(
                model,
                kind=spec.kind,
                price_seq=bp,
                text_emb=bt,
                web_feat=bw,
                lag_minutes=blag,
                return_embeddings=spec.use_contrastive,
                return_position=spec.use_direct_position,
            )
            logits, position_logits, price_repr, text_repr = unpack_model_forward(
                fwd,
                use_direct_position=spec.use_direct_position,
                use_contrastive=spec.use_contrastive,
            )
            if spec.use_return_weighted_loss:
                cls_loss = return_weighted_bce_loss(
                    logits,
                    bl,
                    br,
                    pos_weight=pos_weight,
                    alpha=spec.return_weight_alpha,
                    weight_clip=spec.return_weight_clip,
                )
            else:
                cls_loss = criterion(logits, bl)
            loss = cls_loss
            if spec.use_contrastive and price_repr is not None and text_repr is not None and bp.size(0) > 1:
                nce = contrastive_nce_loss(price_repr, text_repr, spec.contrastive_temperature)
                loss = loss + spec.contrastive_weight * nce
            # --- P4: pairwise ranking loss ---
            if spec.use_ranking_loss and bp.size(0) > 1:
                rank_loss = pairwise_ranking_loss(logits, bl, spec.ranking_margin)
                loss = loss + spec.ranking_weight * rank_loss
            if spec.use_utility_loss and _epoch >= spec.utility_warmup_epochs:
                utility_logits = position_logits if spec.use_direct_position else logits
                if utility_logits is None:
                    raise ValueError("Direct-position utility requested but model returned no position logits")
                utility_loss = soft_position_utility_loss(
                    utility_logits,
                    br,
                    cost_bps=transaction_cost_bps,
                    temperature=spec.position_temperature,
                    mode=spec.utility_mode,
                    return_scale=spec.utility_return_scale,
                )
                loss = loss + spec.utility_weight * utility_loss
            if spec.use_direct_position and spec.position_target_weight > 0.0 and position_logits is not None:
                target_loss = direct_position_target_loss(
                    position_logits,
                    br,
                    cost_bps=transaction_cost_bps,
                    temperature=spec.position_temperature,
                    target_threshold_bps=spec.position_target_threshold_bps,
                )
                loss = loss + spec.position_target_weight * target_loss
            if spec.use_direct_position and spec.position_l1_weight > 0.0 and position_logits is not None:
                positions = logits_to_positions(position_logits, spec.position_temperature)
                loss = loss + spec.position_l1_weight * positions.abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if spec.use_direct_position:
            val_prob, val_positions = predict_probs_and_positions(
                model,
                spec.kind,
                val_seq,
                val_text,
                val_web,
                val_lag,
                device,
                spec.batch_size,
                position_temperature=spec.position_temperature,
            )
        else:
            val_prob = predict_probs(
                model, spec.kind, val_seq, val_text, val_web, val_lag, device, spec.batch_size
            )
            val_positions = None
        cls = compute_classification_metrics(y_val, val_prob)
        downstream_for_selection = None
        if str(spec.selection_metric).strip().lower() in DOWNSTREAM_SELECTION_METRICS:
            if spec.use_direct_position:
                val_trade_for_selection = aggregate_position_for_downstream(
                    split["val"],
                    val_prob,
                    val_positions,
                )
                downstream_for_selection = compute_position_downstream_metrics(
                    future_returns=val_trade_for_selection["future_return"].to_numpy(dtype=float),
                    positions=val_trade_for_selection["position"].to_numpy(dtype=float),
                    min_abs_position=spec.direct_position_threshold,
                    cost_bps=transaction_cost_bps,
                )
            else:
                val_trade_for_selection = aggregate_for_downstream(split["val"], val_prob)
                sel_long_th, sel_short_th = select_trade_thresholds(
                    val_trade_for_selection["y_prob"].to_numpy(dtype=float),
                    val_trade_for_selection["future_return"].to_numpy(dtype=float),
                    cost_bps=transaction_cost_bps,
                )
                downstream_for_selection = compute_downstream_metrics(
                    future_returns=val_trade_for_selection["future_return"].to_numpy(dtype=float),
                    y_prob=val_trade_for_selection["y_prob"].to_numpy(dtype=float),
                    long_threshold=sel_long_th,
                    short_threshold=sel_short_th,
                    cost_bps=transaction_cost_bps,
                )
        score = compute_selection_score(spec.selection_metric, cls, downstream_for_selection)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = spec.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    model.load_state_dict(best_state)

    if spec.use_direct_position:
        val_prob, val_positions = predict_probs_and_positions(
            model,
            spec.kind,
            val_seq,
            val_text,
            val_web,
            val_lag,
            device,
            spec.batch_size,
            position_temperature=spec.position_temperature,
        )
        test_prob, test_positions = predict_probs_and_positions(
            model,
            spec.kind,
            test_seq,
            test_text,
            test_web,
            test_lag,
            device,
            spec.batch_size,
            position_temperature=spec.position_temperature,
        )
        val_trade_df = aggregate_position_for_downstream(split["val"], val_prob, val_positions)
        test_trade_df = aggregate_position_for_downstream(split["test"], test_prob, test_positions)
        long_th = float(spec.direct_position_threshold)
        short_th = -float(spec.direct_position_threshold)
        val_downstream_metrics = compute_position_downstream_metrics(
            future_returns=val_trade_df["future_return"].to_numpy(dtype=float),
            positions=val_trade_df["position"].to_numpy(dtype=float),
            min_abs_position=spec.direct_position_threshold,
            cost_bps=transaction_cost_bps,
        )
        test_downstream_metrics = compute_position_downstream_metrics(
            future_returns=test_trade_df["future_return"].to_numpy(dtype=float),
            positions=test_trade_df["position"].to_numpy(dtype=float),
            min_abs_position=spec.direct_position_threshold,
            cost_bps=transaction_cost_bps,
        )
    else:
        val_prob = predict_probs(
            model, spec.kind, val_seq, val_text, val_web, val_lag, device, spec.batch_size
        )
        test_prob = predict_probs(
            model, spec.kind, test_seq, test_text, test_web, test_lag, device, spec.batch_size
        )
        val_positions = None
        test_positions = None
        val_trade_df = aggregate_for_downstream(split["val"], val_prob)
        test_trade_df = aggregate_for_downstream(split["test"], test_prob)
        long_th, short_th = select_trade_thresholds(
            val_trade_df["y_prob"].to_numpy(dtype=float),
            val_trade_df["future_return"].to_numpy(dtype=float),
            cost_bps=transaction_cost_bps,
        )
        val_downstream_metrics = compute_downstream_metrics(
            future_returns=val_trade_df["future_return"].to_numpy(dtype=float),
            y_prob=val_trade_df["y_prob"].to_numpy(dtype=float),
            long_threshold=long_th,
            short_threshold=short_th,
            cost_bps=transaction_cost_bps,
        )
        test_downstream_metrics = compute_downstream_metrics(
            future_returns=test_trade_df["future_return"].to_numpy(dtype=float),
            y_prob=test_trade_df["y_prob"].to_numpy(dtype=float),
            long_threshold=long_th,
            short_threshold=short_th,
            cost_bps=transaction_cost_bps,
        )

    gate_analysis_df = None
    if spec.export_gate_analysis and spec.kind == "conditionally_gated_cross_modal":
        gate_analysis_df = collect_gate_analysis(
            model=model,
            kind=spec.kind,
            price_seq=test_seq,
            text_emb=test_text,
            web_feat=test_web,
            lag_minutes=test_lag,
            split_df=split["test"],
            device=device,
            batch_size=spec.batch_size,
        )
        if not gate_analysis_df.empty:
            gate_analysis_df["model"] = spec.name

    result = {
        "model": spec.name,
        "kind": spec.kind,
        "feature_group": "price_text_web",
        "n_features": int(train_seq.shape[-1] + train_text.shape[-1] + train_web.shape[-1]),
        "features": {
            "price_sequence_dim": int(train_seq.shape[-1]),
            "text_embedding_dim": int(train_text.shape[-1]),
            "web_dim": int(train_web.shape[-1]),
        },
        "val_classification": compute_classification_metrics(y_val, val_prob).to_dict(),
        "test_classification": compute_classification_metrics(y_test, test_prob).to_dict(),
        "val_downstream": val_downstream_metrics.to_dict(),
        "test_downstream": test_downstream_metrics.to_dict(),
        "val_trade_windows": int(len(val_trade_df)),
        "test_trade_windows": int(len(test_trade_df)),
        "learning_rate": spec.lr,
        "batch_size": spec.batch_size,
        "epochs": spec.epochs,
        "selection_metric": spec.selection_metric,
        "direct_position": bool(spec.use_direct_position),
    }
    if spec.shuffle_lag or spec.shuffle_text or spec.shuffle_web:
        result["sanity_shuffle"] = {
            "shuffle_lag": bool(spec.shuffle_lag),
            "shuffle_text": bool(spec.shuffle_text),
            "shuffle_web": bool(spec.shuffle_web),
        }
    if export_trade_predictions:
        if spec.use_direct_position:
            val_rows = build_position_trade_prediction_rows(
                split_df=split["val"],
                y_prob=val_prob,
                positions=val_positions,
                split_name="val",
                model_name=spec.name,
                fold_id=fold_id,
                lag_minutes=val_lag,
                min_abs_position=spec.direct_position_threshold,
                cost_bps=transaction_cost_bps,
            )
            test_rows = build_position_trade_prediction_rows(
                split_df=split["test"],
                y_prob=test_prob,
                positions=test_positions,
                split_name="test",
                model_name=spec.name,
                fold_id=fold_id,
                lag_minutes=test_lag,
                min_abs_position=spec.direct_position_threshold,
                cost_bps=transaction_cost_bps,
            )
        else:
            val_rows = build_trade_prediction_rows(
                split_df=split["val"],
                y_prob=val_prob,
                split_name="val",
                model_name=spec.name,
                fold_id=fold_id,
                lag_minutes=val_lag,
                long_threshold=long_th,
                short_threshold=short_th,
                cost_bps=transaction_cost_bps,
            )
            test_rows = build_trade_prediction_rows(
                split_df=split["test"],
                y_prob=test_prob,
                split_name="test",
                model_name=spec.name,
                fold_id=fold_id,
                lag_minutes=test_lag,
                long_threshold=long_th,
                short_threshold=short_th,
                cost_bps=transaction_cost_bps,
            )
        result["trade_prediction_rows"] = pd.concat([val_rows, test_rows], ignore_index=True).to_dict(orient="records")
    if gate_analysis_df is not None and not gate_analysis_df.empty:
        result["gate_analysis_rows"] = gate_analysis_df.to_dict(orient="records")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep multimodal ACM MM style experiments")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--build-only", action="store_true", help="Build dataset artifacts only")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed for reproducibility")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Torch device")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    setup_logging(verbose=not args.quiet)
    device = resolve_device(args.device)
    logger.info("Using torch device: requested=%s resolved=%s torch=%s", args.device, device, torch.__version__)
    config = load_config(args.config)
    dataset_config = MultimodalDatasetConfig.from_dict(config["dataset"])
    builder = DeepMultimodalDatasetBuilder(dataset_config)
    artifacts = builder.build()
    artifacts.dataset = artifacts.dataset.reset_index(drop=True).copy()
    artifacts.dataset["row_id"] = np.arange(len(artifacts.dataset), dtype=int)

    experiment_name = config.get("name", "mm_deep_experiment")
    output_dir = config.get("output_dir", "experiments/results_mm")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.seed is not None:
        timestamp = f"{timestamp}_seed{args.seed}"
    run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    saved_artifacts = builder.save(artifacts, output_dir=run_dir)

    text_cfg = config.get("text_encoder", {})
    text_model_name = text_cfg.get(
        "model_name",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    text_cache_dir = text_cfg.get("cache_dir", "experiments/mm_cache/text_embeddings")
    text_cache_key = text_cfg.get(
        "cache_dataset_key",
        (
            f"{experiment_name}_{dataset_symbol_tag(dataset_config)}_"
            f"{dataset_config.interval}_{len(artifacts.dataset)}"
        ),
    )
    text_embeddings = cache_text_embeddings(
        texts=artifacts.dataset["text_blob"].fillna("").astype(str).tolist(),
        model_name=text_model_name,
        cache_dir=text_cache_dir,
        dataset_key=text_cache_key,
    )

    if args.build_only:
        payload = {
            "saved_artifacts": saved_artifacts,
            "dataset_metadata": artifacts.metadata,
            "text_encoder": {"model_name": text_model_name, "embedding_dim": int(text_embeddings.shape[-1])},
        }
        print(json.dumps(payload, indent=2))
        return

    split_mode = config.get("split_mode", "rolling")
    transaction_cost_bps = float(config.get("transaction_cost_bps", 0.0))
    if split_mode == "rolling":
        rolling_cfg = config.get("rolling", {})
        split_sets = build_rolling_splits(
            artifacts.dataset,
            min_train_windows=int(rolling_cfg.get("min_train_windows", 40)),
            val_windows=int(rolling_cfg.get("val_windows", 16)),
            test_windows=int(rolling_cfg.get("test_windows", 12)),
            step_windows=int(rolling_cfg.get("step_windows", rolling_cfg.get("test_windows", 12))),
        )
    else:
        split_sets = [
            split_dataset(
                artifacts.dataset,
                train_ratio=float(config.get("train_ratio", 0.6)),
                val_ratio=float(config.get("val_ratio", 0.2)),
            )
        ]

    model_specs = [
        DeepModelSpec(
            name=model_cfg["name"],
            kind=model_cfg["kind"],
            d_model=int(model_cfg.get("d_model", 96)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            price_layers=int(model_cfg.get("price_layers", 2)),
            fusion_layers=int(model_cfg.get("fusion_layers", 1)),
            hidden_dim=int(model_cfg.get("hidden_dim", 192)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            lag_decay=float(model_cfg.get("lag_decay", 0.1)),
            lr=float(model_cfg.get("lr", 1e-3)),
            weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
            batch_size=int(model_cfg.get("batch_size", 32)),
            epochs=int(model_cfg.get("epochs", 12)),
            patience=int(model_cfg.get("patience", 4)),
            selection_metric=str(model_cfg.get("selection_metric", "auc")),
            use_mixup=bool(model_cfg.get("use_mixup", False)),
            mixup_alpha=float(model_cfg.get("mixup_alpha", 0.4)),
            train_lag_min=float(model_cfg.get("train_lag_min", 0.0)),
            train_lag_max=float(model_cfg.get("train_lag_max", float("inf"))),
            use_hybrid_text=bool(model_cfg.get("use_hybrid_text", False)),
            hybrid_tfidf_features=int(model_cfg.get("hybrid_tfidf_features", 50)),
            lr_schedule=str(model_cfg.get("lr_schedule", "none")),
            use_contrastive=bool(model_cfg.get("use_contrastive", False)),
            contrastive_weight=float(model_cfg.get("contrastive_weight", 0.1)),
            contrastive_temperature=float(model_cfg.get("contrastive_temperature", 0.07)),
            stale_threshold_hours=float(model_cfg.get("stale_threshold_hours", 1.5)),
            use_quality_gate=bool(model_cfg.get("use_quality_gate", False)),
            use_ranking_loss=bool(model_cfg.get("use_ranking_loss", False)),
            ranking_margin=float(model_cfg.get("ranking_margin", 0.1)),
            ranking_weight=float(model_cfg.get("ranking_weight", 0.3)),
            use_return_weighted_loss=bool(model_cfg.get("use_return_weighted_loss", False)),
            return_weight_alpha=float(model_cfg.get("return_weight_alpha", 200.0)),
            return_weight_clip=float(model_cfg.get("return_weight_clip", 5.0)),
            use_utility_loss=bool(model_cfg.get("use_utility_loss", False)),
            utility_weight=float(model_cfg.get("utility_weight", 0.1)),
            utility_mode=str(model_cfg.get("utility_mode", "mean")),
            utility_return_scale=float(model_cfg.get("utility_return_scale", 1000.0)),
            utility_warmup_epochs=int(model_cfg.get("utility_warmup_epochs", 0)),
            position_temperature=float(model_cfg.get("position_temperature", 1.0)),
            use_direct_position=bool(model_cfg.get("use_direct_position", False)),
            direct_position_threshold=float(model_cfg.get("direct_position_threshold", 0.05)),
            position_l1_weight=float(model_cfg.get("position_l1_weight", 0.0)),
            position_target_weight=float(model_cfg.get("position_target_weight", 0.0)),
            position_target_threshold_bps=float(model_cfg.get("position_target_threshold_bps", -1.0)),
            freeze_text_epochs=int(model_cfg.get("freeze_text_epochs", 0)),
            text_base_dim=int(model_cfg.get("text_base_dim", 384)),
            min_fold_text_diversity=float(model_cfg.get("min_fold_text_diversity", 0.0)),
            context_feature_set=str(model_cfg.get("context_feature_set", config.get("context_feature_set", "web"))),
            use_cross_attention=bool(model_cfg.get("use_cross_attention", True)),
            gate_mode=str(model_cfg.get("gate_mode", "vector")),
            use_web_gate_feature=bool(model_cfg.get("use_web_gate_feature", True)),
            use_lag_gate_feature=bool(model_cfg.get("use_lag_gate_feature", True)),
            export_gate_analysis=bool(model_cfg.get("export_gate_analysis", False)),
            patch_len=int(model_cfg.get("patch_len", 16)),
            stride=int(model_cfg.get("stride", 8)),
            max_len=int(model_cfg.get("max_len", 256)),
            shuffle_lag=bool(model_cfg.get("shuffle_lag", False)),
            shuffle_text=bool(model_cfg.get("shuffle_text", False)),
            shuffle_web=bool(model_cfg.get("shuffle_web", False)),
        )
        for model_cfg in config["models"]
    ]

    lag_features = artifacts.dataset["modality_age_minutes"].to_numpy(dtype=np.float32)
    text_blobs = artifacts.dataset["text_blob"].fillna("").astype(str).to_numpy()
    # news_direction_score is web_feat column 0 (WEB_FEATURES[0])
    direction_scores_all = artifacts.web_features[:, 0].astype(np.float32)

    export_trade_predictions = bool(config.get("export_trade_predictions", False))
    results = []
    for spec in model_specs:
        context_feature_set = str(spec.context_feature_set).strip().lower()
        if context_feature_set == "web":
            context_feature_names = WEB_FEATURES
        elif context_feature_set == "web_text_aux":
            context_feature_names = WEB_FEATURES + TEXT_AUX_FEATURES
        else:
            raise ValueError(f"Unknown context_feature_set: {spec.context_feature_set}")
        context_features = artifacts.dataset[context_feature_names].to_numpy(dtype=np.float32)
        fold_results = []
        skipped_diversity = 0
        for fold_id, split in enumerate(split_sets):
            # --- Diversity-aware fold filter ---
            if spec.min_fold_text_diversity > 0.0:
                train_idx = split["train"]["row_id"].to_numpy(dtype=int)
                direction_std = float(direction_scores_all[train_idx].std())
                logger.debug(
                    "%s fold diversity: news_direction_score std=%.4f (threshold=%.4f)",
                    spec.name, direction_std, spec.min_fold_text_diversity,
                )
                if direction_std < spec.min_fold_text_diversity:
                    skipped_diversity += 1
                    continue
            fold_results.append(train_one_fold(
                spec=spec,
                split=split,
                price_sequences=artifacts.price_sequences,
                text_embeddings=text_embeddings,
                web_features=context_features,
                lag_features=lag_features,
                device=device,
                text_blobs=text_blobs,
                fold_id=fold_id,
                export_trade_predictions=export_trade_predictions,
                transaction_cost_bps=transaction_cost_bps,
            ))
        if skipped_diversity > 0:
            logger.info(
                "%s: diversity filter skipped %d / %d folds (news_direction_score std < %.3f)",
                spec.name, skipped_diversity, len(split_sets), spec.min_fold_text_diversity,
            )
        if not fold_results:
            logger.warning("%s: all folds skipped by diversity filter — lowering threshold", spec.name)
            continue
        gate_frames = []
        trade_prediction_frames = []
        for fold_result in fold_results:
            gate_rows = fold_result.pop("gate_analysis_rows", None)
            if gate_rows:
                gate_frames.append(pd.DataFrame(gate_rows))
            trade_rows = fold_result.pop("trade_prediction_rows", None)
            if trade_rows:
                trade_prediction_frames.append(pd.DataFrame(trade_rows))
        result = aggregate_model_results(fold_results) if len(fold_results) > 1 else fold_results[0]
        if len(fold_results) == 1:
            result["n_folds"] = 1
            result["fold_results"] = [dict(fold_results[0])]
        results.append(result)
        if gate_frames:
            pd.concat(gate_frames, ignore_index=True).to_csv(
                os.path.join(run_dir, f"{sanitize_name(spec.name)}_gate_analysis.csv"),
                index=False,
            )
        if trade_prediction_frames:
            trade_path = os.path.join(run_dir, f"{sanitize_name(spec.name)}_trade_predictions.csv")
            pd.concat(trade_prediction_frames, ignore_index=True).to_csv(trade_path, index=False)
            result["trade_predictions_path"] = trade_path

    summary_df = summarize_results(results)
    result_payload = {
        "name": experiment_name,
        "config": config,
        "dataset_metadata": artifacts.metadata,
        "saved_artifacts": saved_artifacts,
        "text_encoder": {"model_name": text_model_name, "embedding_dim": int(text_embeddings.shape[-1])},
        "device_requested": args.device,
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "split_mode": split_mode,
        "transaction_cost_bps": transaction_cost_bps,
        "split_sizes": [
            {"fold": idx, **{name: int(len(split_df)) for name, split_df in split.items()}}
            for idx, split in enumerate(split_sets)
        ],
        "results": results,
    }

    summary_path = os.path.join(run_dir, "summary.csv")
    results_path = os.path.join(run_dir, "results.json")
    summary_df.to_csv(summary_path, index=False)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    logger.info("Saved deep multimodal outputs to %s", run_dir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
