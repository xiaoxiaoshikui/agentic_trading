"""
Deep multimodal models for ACM MM style event-driven experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PriceSequenceEncoder(nn.Module):
    def __init__(
        self,
        price_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.price_proj = nn.Linear(price_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = price_seq.shape
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, price_tokens], dim=1)
        tokens = self.encoder(tokens)
        return self.norm(tokens[:, 0])

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)


class PriceSequenceTransformer(nn.Module):
    def __init__(
        self,
        price_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = PriceSequenceEncoder(
            price_dim=price_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.encoder(price_seq)).squeeze(-1)
        return logits


class PatchTSTPriceClassifier(nn.Module):
    """PatchTST-style price-only baseline for rolling market sequences.

    The model splits the dense price history into temporal patches, projects each
    patch to a token, and classifies from a learned CLS token.  It is intentionally
    price-only so it can test whether CGCMA's gains are merely due to a weak price
    encoder.
    """

    def __init__(
        self,
        price_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        patch_len: int = 16,
        stride: int = 8,
        max_patches: int = 64,
    ) -> None:
        super().__init__()
        self.price_dim = int(price_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.max_patches = int(max_patches)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(self.patch_len * self.price_dim),
            nn.Linear(self.patch_len * self.price_dim, d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional = nn.Parameter(torch.zeros(1, self.max_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.positional, std=0.02)

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, price_dim = price_seq.shape
        if price_dim != self.price_dim:
            raise ValueError(f"Expected price_dim={self.price_dim}, got {price_dim}")
        if seq_len < self.patch_len:
            pad = self.patch_len - seq_len
            price_seq = F.pad(price_seq, (0, 0, pad, 0))
        patches = price_seq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold gives (B, n_patches, D, patch_len); make each patch contiguous.
        patches = patches.transpose(2, 3).contiguous()
        patches = patches.reshape(batch_size, patches.size(1), self.patch_len * self.price_dim)
        if patches.size(1) > self.max_patches:
            patches = patches[:, -self.max_patches :]
        patch_tokens = self.patch_proj(patches)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        tokens = tokens + self.positional[:, : tokens.size(1)]
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0]).squeeze(-1)


class ITransformerPriceClassifier(nn.Module):
    """iTransformer-style price-only baseline with variables as tokens."""

    def __init__(
        self,
        price_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.price_dim = int(price_dim)
        self.max_len = int(max_len)
        self.value_proj = nn.Sequential(
            nn.LayerNorm(self.max_len),
            nn.Linear(self.max_len, d_model),
            nn.GELU(),
        )
        self.variable_embedding = nn.Parameter(torch.zeros(1, self.price_dim, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.variable_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, price_dim = price_seq.shape
        if price_dim != self.price_dim:
            raise ValueError(f"Expected price_dim={self.price_dim}, got {price_dim}")
        x = price_seq.transpose(1, 2).contiguous()
        if seq_len < self.max_len:
            x = F.pad(x, (self.max_len - seq_len, 0))
        elif seq_len > self.max_len:
            x = x[:, :, -self.max_len :]
        tokens = self.value_proj(x) + self.variable_embedding
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


class DLinearPriceClassifier(nn.Module):
    """DLinear-style price-only baseline with trend and residual readouts."""

    def __init__(
        self,
        price_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        max_len: int = 64,
        trend_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.price_dim = int(price_dim)
        self.max_len = int(max_len)
        if trend_kernel % 2 == 0:
            trend_kernel += 1
        self.trend_pool = nn.AvgPool1d(
            kernel_size=trend_kernel,
            stride=1,
            padding=trend_kernel // 2,
            count_include_pad=False,
        )
        self.trend_linear = nn.Sequential(
            nn.LayerNorm(self.max_len),
            nn.Linear(self.max_len, 1),
        )
        self.residual_linear = nn.Sequential(
            nn.LayerNorm(self.max_len),
            nn.Linear(self.max_len, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.price_dim),
            nn.Linear(self.price_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, price_dim = price_seq.shape
        if price_dim != self.price_dim:
            raise ValueError(f"Expected price_dim={self.price_dim}, got {price_dim}")
        x = price_seq.transpose(1, 2).contiguous()
        if seq_len < self.max_len:
            x = F.pad(x, (self.max_len - seq_len, 0))
        elif seq_len > self.max_len:
            x = x[:, :, -self.max_len :]
        trend = self.trend_pool(x)
        residual = x - trend
        channel_scores = self.trend_linear(trend).squeeze(-1) + self.residual_linear(residual).squeeze(-1)
        if channel_scores.shape != (batch_size, self.price_dim):
            raise RuntimeError("Unexpected DLinear channel score shape")
        return self.head(channel_scores).squeeze(-1)


class TimesNetLiteEncoder(nn.Module):
    """Multi-scale convolutional price encoder inspired by TimesNet blocks."""

    def __init__(
        self,
        price_dim: int,
        d_model: int = 64,
        dropout: float = 0.2,
        kernel_sizes: tuple[int, ...] = (3, 5, 9, 15),
    ) -> None:
        super().__init__()
        self.price_dim = int(price_dim)
        self.d_model = int(d_model)
        self.input_proj = nn.Conv1d(self.price_dim, d_model, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.GELU(),
                )
                for k in kernel_sizes
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(d_model * len(kernel_sizes), d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def encode_tokens(self, price_seq: torch.Tensor) -> torch.Tensor:
        _, _, price_dim = price_seq.shape
        if price_dim != self.price_dim:
            raise ValueError(f"Expected price_dim={self.price_dim}, got {price_dim}")
        x = price_seq.transpose(1, 2).contiguous()
        x = self.input_proj(x)
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.mix(multi_scale).transpose(1, 2).contiguous()

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_tokens(price_seq)
        return self.pool(tokens.transpose(1, 2)).squeeze(-1)


class TimesNetLitePriceClassifier(nn.Module):
    """Multi-scale convolutional price-only baseline inspired by TimesNet blocks."""

    def __init__(
        self,
        price_dim: int,
        d_model: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        kernel_sizes: tuple[int, ...] = (3, 5, 9, 15),
    ) -> None:
        super().__init__()
        self.encoder = TimesNetLiteEncoder(
            price_dim=price_dim,
            d_model=d_model,
            dropout=dropout,
            kernel_sizes=kernel_sizes,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, price_seq: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(price_seq)
        return self.head(hidden).squeeze(-1)


class TimesNetGatedCrossModalFusion(nn.Module):
    """TimesNet price encoder with CGCMA-style gated multimodal residuals.

    The model keeps a strong price-only path and adds text/web information as a
    gated residual logit.  The residual head is zero-initialized, so training
    starts from the price-only decision and must earn any multimodal correction.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        gate_mode: str = "vector",
        use_web_gate_feature: bool = True,
        use_lag_gate_feature: bool = True,
    ) -> None:
        super().__init__()
        gate_mode = str(gate_mode).strip().lower()
        if gate_mode not in {"vector", "scalar", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.use_web_gate_feature = use_web_gate_feature
        self.use_lag_gate_feature = use_lag_gate_feature

        self.price_encoder = TimesNetLiteEncoder(price_dim=price_dim, d_model=d_model, dropout=dropout)
        self.price_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
        )
        self.text_to_price_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(d_model)
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_in = d_model * 3
        if self.use_web_gate_feature:
            gate_in += hidden_dim // 2
        if self.use_lag_gate_feature:
            gate_in += 1
        if self.gate_mode == "none":
            self.gate = None
        else:
            gate_out = d_model if self.gate_mode == "vector" else 1
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, gate_out),
                nn.Sigmoid(),
            )

        residual_in = d_model * 2 + hidden_dim // 2 + 1
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_in),
            nn.Linear(residual_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple:
        batch_size = price_seq.size(0)
        price_tokens = self.price_encoder.encode_tokens(price_seq)
        h_price = price_tokens.mean(dim=1)

        text_token = self.text_proj(text_emb).unsqueeze(1)
        h_text_ctx, _ = self.text_to_price_attn(text_token, price_tokens, price_tokens, need_weights=False)
        h_text_ctx = self.text_norm(h_text_ctx.squeeze(1))
        h_web = self.web_proj(web_feat)

        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
        else:
            lag_h = torch.zeros(batch_size, 1, device=price_seq.device)

        gate_parts = [h_price, h_text_ctx, h_price - h_text_ctx]
        if self.use_web_gate_feature:
            gate_parts.append(h_web)
        if self.use_lag_gate_feature:
            gate_parts.append(lag_h)

        if self.gate_mode == "none":
            gate = torch.ones_like(h_text_ctx)
            h_text_gated = h_text_ctx
        else:
            gate = self.gate(torch.cat(gate_parts, dim=-1))
            if self.gate_mode == "scalar":
                gate = gate.expand_as(h_text_ctx)
            h_text_gated = gate * h_text_ctx

        price_logit = self.price_head(h_price).squeeze(-1)
        residual_logit = self.residual_head(
            torch.cat([h_price, h_text_gated, h_web, lag_h], dim=-1)
        ).squeeze(-1)
        logits = price_logit + residual_logit

        if return_gate:
            return logits, h_price, h_text_ctx, gate
        if return_embeddings:
            return logits, h_price, h_text_ctx
        return logits


class DualHeadTimesNetGatedCrossModalFusion(nn.Module):
    """TimesNet-CGCMA with separate direction and direct-position heads."""

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        gate_mode: str = "vector",
        use_web_gate_feature: bool = True,
        use_lag_gate_feature: bool = True,
    ) -> None:
        super().__init__()
        gate_mode = str(gate_mode).strip().lower()
        if gate_mode not in {"vector", "scalar", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.use_web_gate_feature = use_web_gate_feature
        self.use_lag_gate_feature = use_lag_gate_feature

        self.price_encoder = TimesNetLiteEncoder(price_dim=price_dim, d_model=d_model, dropout=dropout)
        self.price_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
        )
        self.text_to_price_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(d_model)
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_in = d_model * 3
        if self.use_web_gate_feature:
            gate_in += hidden_dim // 2
        if self.use_lag_gate_feature:
            gate_in += 1
        if self.gate_mode == "none":
            self.gate = None
        else:
            gate_out = d_model if self.gate_mode == "vector" else 1
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, gate_out),
                nn.Sigmoid(),
            )

        head_in = d_model * 2 + hidden_dim // 2 + 1
        self.residual_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.position_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
        return_gate: bool = False,
        return_position: bool = False,
    ) -> torch.Tensor | tuple:
        batch_size = price_seq.size(0)
        price_tokens = self.price_encoder.encode_tokens(price_seq)
        h_price = price_tokens.mean(dim=1)

        text_token = self.text_proj(text_emb).unsqueeze(1)
        h_text_ctx, _ = self.text_to_price_attn(text_token, price_tokens, price_tokens, need_weights=False)
        h_text_ctx = self.text_norm(h_text_ctx.squeeze(1))
        h_web = self.web_proj(web_feat)

        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
        else:
            lag_h = torch.zeros(batch_size, 1, device=price_seq.device)

        gate_parts = [h_price, h_text_ctx, h_price - h_text_ctx]
        if self.use_web_gate_feature:
            gate_parts.append(h_web)
        if self.use_lag_gate_feature:
            gate_parts.append(lag_h)

        if self.gate_mode == "none":
            gate = torch.ones_like(h_text_ctx)
            h_text_gated = h_text_ctx
        else:
            gate = self.gate(torch.cat(gate_parts, dim=-1))
            if self.gate_mode == "scalar":
                gate = gate.expand_as(h_text_ctx)
            h_text_gated = gate * h_text_ctx

        shared = torch.cat([h_price, h_text_gated, h_web, lag_h], dim=-1)
        direction_logits = self.price_head(h_price).squeeze(-1) + self.residual_head(shared).squeeze(-1)
        position_logits = self.position_head(shared).squeeze(-1)

        if return_gate:
            if return_position:
                return direction_logits, position_logits, h_price, h_text_ctx, gate
            return direction_logits, h_price, h_text_ctx, gate
        if return_embeddings:
            if return_position:
                return direction_logits, position_logits, h_price, h_text_ctx
            return direction_logits, h_price, h_text_ctx
        if return_position:
            return direction_logits, position_logits
        return direction_logits


class TimesNetMixtureOfExpertsFusion(nn.Module):
    """TimesNet-backed mixture of price, text, web, and interaction experts."""

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.price_encoder = TimesNetLiteEncoder(price_dim=price_dim, d_model=d_model, dropout=dropout)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_to_price_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_norm = nn.LayerNorm(d_model)
        self.price_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.text_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.web_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 1),
            nn.Linear(d_model * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.expert_gate = nn.Sequential(
            nn.LayerNorm(d_model * 3 + 1),
            nn.Linear(d_model * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.expert_gate[-1].weight)
        with torch.no_grad():
            self.expert_gate[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple:
        batch_size = price_seq.size(0)
        price_tokens = self.price_encoder.encode_tokens(price_seq)
        h_price = price_tokens.mean(dim=1)
        h_text = self.text_proj(text_emb).unsqueeze(1)
        h_text_ctx, _ = self.text_to_price_attn(h_text, price_tokens, price_tokens, need_weights=False)
        h_text_ctx = self.text_norm(h_text_ctx.squeeze(1))
        h_web = self.web_proj(web_feat)
        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
        else:
            lag_h = torch.zeros(batch_size, 1, device=price_seq.device)

        gate_input = torch.cat([h_price, h_text_ctx, h_web, lag_h], dim=-1)
        expert_logits = torch.stack(
            [
                self.price_head(h_price).squeeze(-1),
                self.text_head(h_text_ctx).squeeze(-1),
                self.web_head(h_web).squeeze(-1),
                self.interaction_head(gate_input).squeeze(-1),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.expert_gate(gate_input), dim=-1)
        logits = torch.sum(weights * expert_logits, dim=-1)

        if return_gate:
            return logits, h_price, h_text_ctx, weights
        if return_embeddings:
            return logits, h_price, h_text_ctx
        return logits


class TextEmbeddingMLP(nn.Module):
    def __init__(self, text_dim: int, web_dim: int, hidden_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.text_norm = nn.LayerNorm(text_dim)
        self.web_norm = nn.LayerNorm(web_dim)
        self.head = nn.Sequential(
            nn.Linear(text_dim + web_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, text_emb: torch.Tensor, web_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.text_norm(text_emb), self.web_norm(web_feat)], dim=-1)
        return self.head(fused).squeeze(-1)


class PriceWebLateFusion(nn.Module):
    """Price-plus-web baseline without any text pathway."""

    def __init__(
        self,
        price_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.price_encoder = PriceSequenceEncoder(
            price_dim=price_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=price_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fused_dim = d_model + hidden_dim // 2 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes=None,
        return_embeddings: bool = False,
    ):
        h_price = self.price_encoder(price_seq)
        h_web = self.web_proj(web_feat)

        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
        else:
            lag_h = torch.zeros(price_seq.size(0), 1, device=price_seq.device)

        fused = torch.cat([h_price, h_web, lag_h], dim=-1)
        logits = self.head(fused).squeeze(-1)

        if return_embeddings:
            return logits, h_price, None
        return logits


class CrossModalAttentionFusion(nn.Module):
    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        price_layers: int = 2,
        fusion_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.price_proj = nn.Linear(price_dim, d_model)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.web_proj = nn.Sequential(nn.Linear(web_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))

        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.text_to_price = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.price_to_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )
        self._reset_parameters()

    def forward(self, price_seq: torch.Tensor, text_emb: torch.Tensor, web_feat: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = price_seq.shape
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        price_tokens = self.price_encoder(price_tokens)

        text_token = self.text_proj(text_emb).unsqueeze(1)
        web_token = self.web_proj(web_feat).unsqueeze(1)
        text_context, _ = self.text_to_price(text_token, price_tokens, price_tokens, need_weights=False)
        price_summary_query = self.cls_token.expand(batch_size, -1, -1)
        price_context, _ = self.price_to_text(price_summary_query, text_token, text_token, need_weights=False)

        fusion_tokens = torch.cat([price_summary_query, text_context, web_token], dim=1)
        fusion_tokens = self.fusion_encoder(fusion_tokens)
        pooled = torch.cat(
            [
                fusion_tokens[:, 0],
                text_context.squeeze(1),
                price_context.squeeze(1) + web_token.squeeze(1),
            ],
            dim=-1,
        )
        return self.head(pooled).squeeze(-1)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)


class GatedLateFusionModel(nn.Module):
    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.price_encoder = PriceSequenceEncoder(
            price_dim=price_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=price_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, price_seq: torch.Tensor, text_emb: torch.Tensor, web_feat: torch.Tensor) -> torch.Tensor:
        price_hidden = self.price_encoder(price_seq)
        text_hidden = self.text_proj(text_emb)
        web_hidden = self.web_proj(web_feat)
        context_hidden = self.context_proj(torch.cat([text_hidden, web_hidden], dim=-1))
        gate_input = torch.cat([price_hidden, context_hidden, price_hidden - context_hidden], dim=-1)
        gate = torch.sigmoid(self.gate_head(gate_input)).squeeze(-1)
        price_logit = self.price_head(price_hidden).squeeze(-1)
        context_logit = self.context_head(context_hidden).squeeze(-1)
        return price_logit + gate * context_logit


class DualBranchConditionalLateFusion(nn.Module):
    """Residual late fusion with separate dense and lexical text branches.

    The model keeps the strong price-only backbone and learns two conditional
    residual text branches:
    - a dense semantic branch over sentence embeddings
    - a sparse lexical branch over appended TF-IDF features

    Each branch predicts an additive correction to the price logit, gated by a
    small controller conditioned on price state, web context, and lag. This is
    intentionally more conservative than cross-modal attention and can fully
    ignore weak text by driving the residual gates towards zero.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        max_len: int = 256,
        dense_text_dim: int = 384,
    ) -> None:
        super().__init__()
        self.dense_text_dim = max(1, min(int(dense_text_dim), int(text_dim)))
        self.lexical_text_dim = max(0, int(text_dim) - self.dense_text_dim)

        self.price_encoder = PriceSequenceEncoder(
            price_dim=price_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=price_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.price_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dense_proj = nn.Sequential(
            nn.LayerNorm(self.dense_text_dim),
            nn.Linear(self.dense_text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.lexical_text_dim > 0:
            self.lexical_proj = nn.Sequential(
                nn.LayerNorm(self.lexical_text_dim),
                nn.Linear(self.lexical_text_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.lexical_proj = None

        gate_in_dim = d_model * 3 + hidden_dim // 2 + 1
        self.dense_gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.lexical_gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.dense_head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + hidden_dim // 2),
            nn.Linear(d_model * 2 + hidden_dim // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.lexical_head = nn.Sequential(
            nn.LayerNorm(d_model * 2 + hidden_dim // 2),
            nn.Linear(d_model * 2 + hidden_dim // 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple:
        price_hidden = self.price_encoder(price_seq)
        web_hidden = self.web_proj(web_feat)
        dense_hidden = self.dense_proj(text_emb[:, : self.dense_text_dim])

        if self.lexical_proj is not None:
            lexical_hidden = self.lexical_proj(text_emb[:, self.dense_text_dim :])
        else:
            lexical_hidden = torch.zeros_like(dense_hidden)

        lag_hours = (
            lag_minutes.float().unsqueeze(-1) / 60.0
            if lag_minutes is not None
            else torch.zeros(price_hidden.size(0), 1, device=price_hidden.device)
        )
        dense_gate_input = torch.cat(
            [price_hidden, dense_hidden, price_hidden - dense_hidden, web_hidden, lag_hours],
            dim=-1,
        )
        lexical_gate_input = torch.cat(
            [price_hidden, lexical_hidden, price_hidden - lexical_hidden, web_hidden, lag_hours],
            dim=-1,
        )

        gate_dense = torch.sigmoid(self.dense_gate(dense_gate_input)).squeeze(-1)
        gate_lexical = torch.sigmoid(self.lexical_gate(lexical_gate_input)).squeeze(-1)

        price_logit = self.price_head(price_hidden).squeeze(-1)
        dense_logit = self.dense_head(
            torch.cat([price_hidden, dense_hidden, web_hidden], dim=-1)
        ).squeeze(-1)
        lexical_logit = self.lexical_head(
            torch.cat([price_hidden, lexical_hidden, web_hidden], dim=-1)
        ).squeeze(-1)

        logits = price_logit + gate_dense * dense_logit + gate_lexical * lexical_logit
        if return_embeddings:
            return logits, price_hidden, dense_hidden
        return logits


class SourceAwareQualityConditionalLateFusion(nn.Module):
    """Late fusion with explicit source and quality-aware residual control."""

    BASE_WEB_DIM = 13

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        max_len: int = 256,
        dense_text_dim: int = 384,
    ) -> None:
        super().__init__()
        self.dense_text_dim = max(1, min(int(dense_text_dim), int(text_dim)))
        self.lexical_text_dim = max(0, int(text_dim) - self.dense_text_dim)
        self.base_web_dim = min(self.BASE_WEB_DIM, int(web_dim))
        self.quality_extra_dim = max(0, int(web_dim) - self.base_web_dim)

        self.price_encoder = PriceSequenceEncoder(
            price_dim=price_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=price_layers,
            dropout=dropout,
            max_len=max_len,
        )
        self.price_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.dense_proj = nn.Sequential(
            nn.LayerNorm(self.dense_text_dim),
            nn.Linear(self.dense_text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.lexical_text_dim > 0:
            self.lexical_proj = nn.Sequential(
                nn.LayerNorm(self.lexical_text_dim),
                nn.Linear(self.lexical_text_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.lexical_proj = None

        quality_dim = 2 + self.quality_extra_dim
        self.quality_proj = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.source_groups = {
            "news": [0, 1, 2, 9],
            "fear_greed": [3, 10],
            "social": [4, 11],
            "whale": [5, 6, 12],
        }
        source_head_in = d_model * 3
        source_gate_in = d_model * 3 + 1
        self.source_projs = nn.ModuleDict()
        self.source_heads = nn.ModuleDict()
        self.source_gates = nn.ModuleDict()
        for name, idxs in self.source_groups.items():
            self.source_projs[name] = nn.Sequential(
                nn.LayerNorm(len(idxs)),
                nn.Linear(len(idxs), hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.source_heads[name] = nn.Sequential(
                nn.LayerNorm(source_head_in),
                nn.Linear(source_head_in, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.source_gates[name] = nn.Sequential(
                nn.LayerNorm(source_gate_in),
                nn.Linear(source_gate_in, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        residual_in = d_model * 3
        gate_in = d_model * 3 + 1
        self.dense_head = nn.Sequential(
            nn.LayerNorm(residual_in),
            nn.Linear(residual_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.lexical_head = nn.Sequential(
            nn.LayerNorm(residual_in),
            nn.Linear(residual_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.dense_gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.lexical_gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _split_context(self, web_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = web_feat[:, : self.base_web_dim]
        quality_core = base[:, [7, 8]]
        if self.quality_extra_dim > 0:
            quality = torch.cat([quality_core, web_feat[:, self.base_web_dim :]], dim=-1)
        else:
            quality = quality_core
        return base, quality

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple:
        price_hidden = self.price_encoder(price_seq)
        dense_hidden = self.dense_proj(text_emb[:, : self.dense_text_dim])
        if self.lexical_proj is not None:
            lexical_hidden = self.lexical_proj(text_emb[:, self.dense_text_dim :])
        else:
            lexical_hidden = torch.zeros_like(dense_hidden)

        base_web, quality_raw = self._split_context(web_feat)
        quality_hidden = self.quality_proj(quality_raw)
        text_shared = self.text_fuse(torch.cat([dense_hidden, lexical_hidden, quality_hidden], dim=-1))
        lag_hours = (
            lag_minutes.float().unsqueeze(-1) / 60.0
            if lag_minutes is not None
            else torch.zeros(price_hidden.size(0), 1, device=price_hidden.device)
        )

        price_logit = self.price_head(price_hidden).squeeze(-1)
        dense_gate = torch.sigmoid(
            self.dense_gate(torch.cat([price_hidden, dense_hidden, quality_hidden, lag_hours], dim=-1))
        ).squeeze(-1)
        lexical_gate = torch.sigmoid(
            self.lexical_gate(torch.cat([price_hidden, lexical_hidden, quality_hidden, lag_hours], dim=-1))
        ).squeeze(-1)
        dense_logit = self.dense_head(torch.cat([price_hidden, dense_hidden, quality_hidden], dim=-1)).squeeze(-1)
        lexical_logit = self.lexical_head(
            torch.cat([price_hidden, lexical_hidden, quality_hidden], dim=-1)
        ).squeeze(-1)

        logits = price_logit + dense_gate * dense_logit + lexical_gate * lexical_logit
        for name, idxs in self.source_groups.items():
            source_hidden = self.source_projs[name](base_web[:, idxs])
            source_gate = torch.sigmoid(
                self.source_gates[name](torch.cat([price_hidden, source_hidden, quality_hidden, lag_hours], dim=-1))
            ).squeeze(-1)
            source_logit = self.source_heads[name](
                torch.cat([price_hidden, text_shared, source_hidden], dim=-1)
            ).squeeze(-1)
            logits = logits + source_gate * source_logit

        if return_embeddings:
            return logits, price_hidden, text_shared
        return logits


class StalenessAwareCrossModalFusion(nn.Module):
    """Staleness-aware cross-modal attention fusion with data-efficient design.

    Three architectural innovations over CrossModalAttentionFusion:

    1. **Staleness gating**: text context is explicitly damped by modality lag
       via ``text_context *= sigmoid(-lag_decay * lag_hours)``, so the model
       learns to discount stale web intelligence without requiring an external
       feature engineering step.

    2. **Mean pooling**: the fusion output is mean-pooled over the 3 fusion
       tokens (R^d) instead of concatenated (R^{3d}), reducing the final
       projection from 288 → 1 to 32 → 1 and cutting the dominant source of
       overfitting on small per-fold training sets.

    3. **Small d_model=32, dropout=0.4**: reduces total parameters from ~100K
       to ~12K, improving sample efficiency on the ~120-sample rolling folds
       present in the current CryptoMI dataset.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        fusion_layers: int = 1,
        dropout: float = 0.4,
        lag_decay: float = 0.1,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.lag_decay = lag_decay
        self.price_proj = nn.Linear(price_dim, d_model)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.web_proj = nn.Sequential(
            nn.Linear(web_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))

        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.text_to_price = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.price_to_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._reset_parameters()

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple:
        """Forward pass with optional staleness gating.

        Args:
            price_seq: (B, T, price_dim) price sequence window.
            text_emb: (B, text_dim) sentence embedding of web text blob.
            web_feat: (B, web_dim) scalar web intelligence features.
            lag_minutes: (B,) modality age in minutes; when provided, text
                context is damped by ``exp(-lag_decay * lag_hours)``.
            return_embeddings: if True, return ``(logits, price_repr, text_repr)``
                where ``price_repr`` is mean-pooled encoded price tokens and
                ``text_repr`` is the projected text token (pre-gate), both
                shape ``(B, d_model)``.  Used by contrastive auxiliary loss.
        """
        batch_size, seq_len, _ = price_seq.shape
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        price_tokens = self.price_encoder(price_tokens)

        text_token = self.text_proj(text_emb).unsqueeze(1)   # (B, 1, d)
        web_token = self.web_proj(web_feat).unsqueeze(1)     # (B, 1, d)

        # Capture pre-gate text repr for contrastive alignment
        text_repr = text_token.squeeze(1)                    # (B, d)
        price_repr = price_tokens.mean(dim=1)                # (B, d)

        # Text attends to encoded price sequence
        text_context, _ = self.text_to_price(
            text_token, price_tokens, price_tokens, need_weights=False
        )

        # Staleness gating: exponential decay — gate=1 when fresh, →0 as lag grows
        if lag_minutes is not None:
            lag_hours = lag_minutes.float() / 60.0               # (B,)
            staleness_gate = torch.exp(
                -self.lag_decay * lag_hours
            ).view(batch_size, 1, 1)                             # (B, 1, 1)
            text_context = text_context * staleness_gate

        # Price CLS token attends to (gated) text context
        price_query = self.cls_token.expand(batch_size, -1, -1)
        price_context, _ = self.price_to_text(
            price_query, text_context, text_context, need_weights=False
        )

        # Fuse and mean-pool over the 3 tokens → (B, d)
        fusion_tokens = torch.cat([price_context, text_context, web_token], dim=1)
        fusion_tokens = self.fusion_encoder(fusion_tokens)
        pooled = fusion_tokens.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)

        if return_embeddings:
            return logits, price_repr, text_repr
        return logits

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)


class LearnedGateCrossModalFusion(nn.Module):
    """SACMA variant with a learned exponential staleness decay rate.

    Identical to StalenessAwareCrossModalFusion except the decay constant λ is
    a learnable scalar parameter instead of a fixed hyperparameter.  The gate
    is still ``exp(-λ · lag_hours)`` so the functional form is preserved and
    the model is initialised to the same behaviour as SACMA (λ=init_lag_decay),
    but it can adapt during training to the actual predictive decay timescale.

    λ is parameterised as ``exp(log_lag_decay)`` (always positive) and is
    jointly optimised with the rest of the network via AdamW.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        fusion_layers: int = 1,
        dropout: float = 0.4,
        init_lag_decay: float = 0.1,
        use_quality_gate: bool = False,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        # Learned decay: always positive via exp; initialised to init_lag_decay
        self.log_lag_decay = nn.Parameter(torch.tensor(math.log(max(init_lag_decay, 1e-6))))
        # P2: quality gate — projects web_token repr to a [0,1] scalar gate on text_context
        self.use_quality_gate = use_quality_gate
        if use_quality_gate:
            self.quality_gate_linear = nn.Linear(d_model, 1)

        self.price_proj = nn.Linear(price_dim, d_model)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.web_proj = nn.Sequential(
            nn.Linear(web_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))

        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.text_to_price = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.price_to_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> "torch.Tensor | tuple":
        """Forward with optional intermediate embedding return for contrastive loss."""
        batch_size, seq_len, _ = price_seq.shape
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        price_tokens = self.price_encoder(price_tokens)

        text_token = self.text_proj(text_emb).unsqueeze(1)   # (B, 1, d)
        web_token = self.web_proj(web_feat).unsqueeze(1)     # (B, 1, d)

        # Capture pre-gate representations for contrastive alignment
        price_repr = price_tokens.mean(dim=1)                # (B, d)
        text_repr = text_token.squeeze(1)                    # (B, d)

        # Text attends to encoded price sequence
        text_context, _ = self.text_to_price(
            text_token, price_tokens, price_tokens, need_weights=False
        )

        # Learned staleness gate: λ = exp(log_lag_decay) > 0; gate = exp(-λ * lag_hours)
        if lag_minutes is not None:
            lag_hours = lag_minutes.float() / 60.0
            lag_decay = torch.exp(self.log_lag_decay)
            staleness_gate = torch.exp(-lag_decay * lag_hours).view(batch_size, 1, 1)
            text_context = text_context * staleness_gate

        # P2: quality gate — learned projection of web context richness onto text_context
        if self.use_quality_gate:
            quality_gate = torch.sigmoid(
                self.quality_gate_linear(web_token.squeeze(1))
            ).view(batch_size, 1, 1)
            text_context = text_context * quality_gate

        # Price CLS token attends to gated text context
        price_query = self.cls_token.expand(batch_size, -1, -1)
        price_context, _ = self.price_to_text(
            price_query, text_context, text_context, need_weights=False
        )

        # Fuse and mean-pool over 3 tokens → (B, d)
        fusion_tokens = torch.cat([price_context, text_context, web_token], dim=1)
        fusion_tokens = self.fusion_encoder(fusion_tokens)
        pooled = fusion_tokens.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)

        if return_embeddings:
            return logits, price_repr, text_repr
        return logits


class StaleInterventionCrossModalFusion(nn.Module):
    """SACMA-LG extended with a hard Stale Intervention mechanism.

    Motivation: bar-aligned lag analysis shows a non-monotonic Sharpe profile
    where the 90–120 min bin has Sharpe = −1.33 (news *hurts* at that age).
    Soft exponential decay still propagates some stale signal; this model uses
    a hard replacement: when lag > ``stale_threshold_hours``, the text context
    token is replaced by a learned ``null_text_emb`` parameter representing
    "no actionable information".  Fresh samples (lag ≤ threshold) receive
    the full soft-gated cross-modal attention path.

    Architecture: identical to LearnedGateCrossModalFusion plus
    ``null_text_emb: nn.Parameter(zeros(1, 1, d_model))``.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        fusion_layers: int = 1,
        dropout: float = 0.4,
        init_lag_decay: float = 0.1,
        stale_threshold_hours: float = 1.5,
        max_len: int = 256,
    ) -> None:
        super().__init__()
        self.stale_threshold_hours = stale_threshold_hours
        self.log_lag_decay = nn.Parameter(torch.tensor(math.log(max(init_lag_decay, 1e-6))))
        # Learned null embedding: what the model reads when text is too stale
        self.null_text_emb = nn.Parameter(torch.zeros(1, 1, d_model))

        self.price_proj = nn.Linear(price_dim, d_model)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.web_proj = nn.Sequential(
            nn.Linear(web_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))

        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.text_to_price = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.price_to_text = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: "torch.Tensor | None" = None,
        return_embeddings: bool = False,
        return_gate: bool = False,
    ) -> "torch.Tensor | tuple":
        """Forward with hard stale intervention when lag > stale_threshold_hours."""
        batch_size, seq_len, _ = price_seq.shape
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        price_tokens = self.price_encoder(price_tokens)

        text_token = self.text_proj(text_emb).unsqueeze(1)   # (B, 1, d)
        web_token = self.web_proj(web_feat).unsqueeze(1)     # (B, 1, d)

        price_repr = price_tokens.mean(dim=1)                # (B, d)
        text_repr = text_token.squeeze(1)                    # (B, d)

        # Text cross-attention to price sequence
        text_context, _ = self.text_to_price(
            text_token, price_tokens, price_tokens, need_weights=False
        )  # (B, 1, d)

        if lag_minutes is not None:
            lag_hours = lag_minutes.float() / 60.0           # (B,)

            # Soft gate: exp(-λ * lag_hours), λ learned; applied to fresh samples
            lag_decay = torch.exp(self.log_lag_decay)
            staleness_gate = torch.exp(-lag_decay * lag_hours).view(batch_size, 1, 1)
            soft_context = text_context * staleness_gate

            # Hard intervention: replace stale context with learned null embedding
            stale_mask = (lag_hours > self.stale_threshold_hours).view(batch_size, 1, 1)
            null_context = self.null_text_emb.expand(batch_size, 1, -1)
            text_context = torch.where(stale_mask, null_context, soft_context)

        # Price CLS token attends to (possibly null-replaced) text context
        price_query = self.cls_token.expand(batch_size, -1, -1)
        price_context, _ = self.price_to_text(
            price_query, text_context, text_context, need_weights=False
        )

        fusion_tokens = torch.cat([price_context, text_context, web_token], dim=1)
        fusion_tokens = self.fusion_encoder(fusion_tokens)
        pooled = fusion_tokens.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)

        if return_embeddings:
            return logits, price_repr, text_repr
        return logits


class ConditionallyGatedCrossModalFusion(nn.Module):
    """CGCMA: Cross-modal attention with content-aware conditional gating.

    Combines the deep text-price interaction from CrossMA with the per-sample
    adaptive trust control from DualBranch.  The text token first attends to
    the full price sequence (cross-modal attention) to form a context-aware
    representation h_text_ctx.  A small gate network then decides, conditioned
    on price state, text context, their difference, web scalars, and lag, how
    much of h_text_ctx to inject back into the price representation:

        h_fused = h_price + g ⊙ h_text_ctx
        logit   = MLP(h_fused, h_web)

    When the gate closes (g → 0) the model degrades exactly to the price-only
    Transformer, making it robust under low-quality or stale text.

    Parameters
    ----------
    price_dim:      raw OHLCV feature width
    text_dim:       sentence-embedding width (e.g. 384 for MiniLM-L12)
    web_dim:        scalar web-intelligence feature width
    d_model:        internal width (default 32 to match SACMA parameter budget)
    n_heads:        MHA heads
    price_layers:   Transformer layers for the price encoder
    hidden_dim:     gate MLP hidden width
    dropout:        dropout rate
    max_len:        maximum price-sequence length
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_len: int = 256,
        use_cross_attention: bool = True,
        gate_mode: str = "vector",
        use_web_gate_feature: bool = True,
        use_lag_gate_feature: bool = True,
    ) -> None:
        super().__init__()
        gate_mode = str(gate_mode).strip().lower()
        if gate_mode not in {"vector", "scalar", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.use_cross_attention = use_cross_attention
        self.gate_mode = gate_mode
        self.use_web_gate_feature = use_web_gate_feature
        self.use_lag_gate_feature = use_lag_gate_feature

        # ----- Price encoder (same budget as SACMA) -----
        self.price_proj = nn.Linear(price_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))
        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.price_norm = nn.LayerNorm(d_model)

        # ----- Text branch -----
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
        )
        # Text token attends to the full price sequence → context-enriched text repr
        if self.use_cross_attention:
            self.text_to_price_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.text_to_price_attn = None
        self.text_ctx_norm = nn.LayerNorm(d_model)

        # ----- Web branch -----
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----- Conditional gate -----
        gate_in = d_model * 3
        if self.use_web_gate_feature:
            gate_in += hidden_dim // 2
        if self.use_lag_gate_feature:
            gate_in += 1
        if self.gate_mode == "none":
            self.gate = None
        else:
            gate_out = d_model if self.gate_mode == "vector" else 1
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, gate_out),
                nn.Sigmoid(),
            )

        # ----- Prediction head -----
        head_in = d_model + hidden_dim // 2
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: "torch.Tensor | None" = None,
        return_embeddings: bool = False,
        return_gate: bool = False,
    ) -> "torch.Tensor | tuple":
        batch_size, seq_len, _ = price_seq.shape

        # --- Price encoding: CLS + positional tokens ---
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, price_tokens], dim=1)
        tokens = self.price_encoder(tokens)
        h_price = self.price_norm(tokens[:, 0])        # (B, d) — CLS repr
        price_seq_enc = tokens[:, 1:]                  # (B, T, d) — sequence for cross-attn

        # --- Text cross-attention to price sequence ---
        h_text_raw = self.text_proj(text_emb).unsqueeze(1)      # (B, 1, d)
        if self.use_cross_attention:
            h_text_ctx_raw, _ = self.text_to_price_attn(
                h_text_raw, price_seq_enc, price_seq_enc, need_weights=False
            )
            h_text_ctx = self.text_ctx_norm(h_text_ctx_raw.squeeze(1))  # (B, d)
        else:
            h_text_ctx = self.text_ctx_norm(h_text_raw.squeeze(1))

        # --- Web projection ---
        h_web = self.web_proj(web_feat)                # (B, hidden//2)

        gate_parts = [h_price, h_text_ctx, h_price - h_text_ctx]
        if self.use_web_gate_feature:
            gate_parts.append(h_web)
        if self.use_lag_gate_feature:
            if lag_minutes is not None:
                lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
            else:
                lag_h = torch.zeros(batch_size, 1, device=price_seq.device)
            gate_parts.append(lag_h)

        if self.gate_mode == "none":
            gate = torch.ones_like(h_text_ctx)
            h_fused = h_price + h_text_ctx
        else:
            gate_input = torch.cat(gate_parts, dim=-1)
            gate = self.gate(gate_input)
            if self.gate_mode == "scalar":
                gate = gate.expand_as(h_text_ctx)
            h_fused = h_price + gate * h_text_ctx

        # --- Head ---
        logits = self.head(torch.cat([h_fused, h_web], dim=-1)).squeeze(-1)

        if return_gate:
            return logits, h_price, h_text_ctx, gate
        if return_embeddings:
            return logits, h_price, h_text_ctx
        return logits


class DualHeadConditionallyGatedCrossModalFusion(nn.Module):
    """CGCMA with one head for direction and one head for direct position sizing."""

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_len: int = 256,
        use_cross_attention: bool = True,
        gate_mode: str = "vector",
        use_web_gate_feature: bool = True,
        use_lag_gate_feature: bool = True,
    ) -> None:
        super().__init__()
        gate_mode = str(gate_mode).strip().lower()
        if gate_mode not in {"vector", "scalar", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.use_cross_attention = use_cross_attention
        self.gate_mode = gate_mode
        self.use_web_gate_feature = use_web_gate_feature
        self.use_lag_gate_feature = use_lag_gate_feature

        self.price_proj = nn.Linear(price_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))
        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.price_norm = nn.LayerNorm(d_model)

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
        )
        if self.use_cross_attention:
            self.text_to_price_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.text_to_price_attn = None
        self.text_ctx_norm = nn.LayerNorm(d_model)

        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_in = d_model * 3
        if self.use_web_gate_feature:
            gate_in += hidden_dim // 2
        if self.use_lag_gate_feature:
            gate_in += 1
        if self.gate_mode == "none":
            self.gate = None
        else:
            gate_out = d_model if self.gate_mode == "vector" else 1
            self.gate = nn.Sequential(
                nn.LayerNorm(gate_in),
                nn.Linear(gate_in, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, gate_out),
                nn.Sigmoid(),
            )

        head_in = d_model + hidden_dim // 2
        self.direction_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.position_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: "torch.Tensor | None" = None,
        return_embeddings: bool = False,
        return_gate: bool = False,
        return_position: bool = False,
    ) -> "torch.Tensor | tuple":
        batch_size, seq_len, _ = price_seq.shape

        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, price_tokens], dim=1)
        tokens = self.price_encoder(tokens)
        h_price = self.price_norm(tokens[:, 0])
        price_seq_enc = tokens[:, 1:]

        h_text_raw = self.text_proj(text_emb).unsqueeze(1)
        if self.use_cross_attention:
            h_text_ctx_raw, _ = self.text_to_price_attn(
                h_text_raw, price_seq_enc, price_seq_enc, need_weights=False
            )
            h_text_ctx = self.text_ctx_norm(h_text_ctx_raw.squeeze(1))
        else:
            h_text_ctx = self.text_ctx_norm(h_text_raw.squeeze(1))

        h_web = self.web_proj(web_feat)

        gate_parts = [h_price, h_text_ctx, h_price - h_text_ctx]
        if self.use_web_gate_feature:
            gate_parts.append(h_web)
        if self.use_lag_gate_feature:
            if lag_minutes is not None:
                lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
            else:
                lag_h = torch.zeros(batch_size, 1, device=price_seq.device)
            gate_parts.append(lag_h)

        if self.gate_mode == "none":
            gate = torch.ones_like(h_text_ctx)
            h_fused = h_price + h_text_ctx
        else:
            gate = self.gate(torch.cat(gate_parts, dim=-1))
            if self.gate_mode == "scalar":
                gate = gate.expand_as(h_text_ctx)
            h_fused = h_price + gate * h_text_ctx

        shared = torch.cat([h_fused, h_web], dim=-1)
        direction_logits = self.direction_head(shared).squeeze(-1)
        position_logits = self.position_head(shared).squeeze(-1)

        if return_gate:
            if return_position:
                return direction_logits, position_logits, h_price, h_text_ctx, gate
            return direction_logits, h_price, h_text_ctx, gate
        if return_embeddings:
            if return_position:
                return direction_logits, position_logits, h_price, h_text_ctx
            return direction_logits, h_price, h_text_ctx
        if return_position:
            return direction_logits, position_logits
        return direction_logits


class EarlyFusionTransformer(nn.Module):
    """Early fusion baseline: project all modalities then concatenate before classification.

    Price sequence → Transformer CLS → h_price (d)
    Text embedding → Linear          → h_text  (d)
    Web scalars    → Linear          → h_web   (d)

    concat [h_price, h_text, h_web, τ_lag] → MLP → logit

    This is the simplest multimodal baseline: all information is merged at the
    representation level before any task-specific processing.  Unlike CGCMA,
    there is no gating, no cross-modal attention, and no graceful degradation
    to price-only.  The model must learn to ignore weak text purely through
    the MLP weights.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_len: int = 256,
    ) -> None:
        super().__init__()

        # Price encoder (same as CGCMA for fair comparison)
        self.price_proj = nn.Linear(price_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))
        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.price_norm = nn.LayerNorm(d_model)

        # Text and web projections
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # MLP head over concatenated representations + lag
        fused_dim = d_model * 3 + 1   # price + text + web + lag scalar
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes: "torch.Tensor | None" = None,
        return_embeddings: bool = False,
    ) -> "torch.Tensor | tuple":
        batch_size, seq_len, _ = price_seq.shape

        # Price encoding
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, price_tokens], dim=1)
        tokens = self.price_encoder(tokens)
        h_price = self.price_norm(tokens[:, 0])     # (B, d)

        # Text and web projections
        h_text = self.text_proj(text_emb)           # (B, d)
        h_web = self.web_proj(web_feat)             # (B, d)

        # Lag scalar
        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0   # (B, 1)
        else:
            lag_h = torch.zeros(batch_size, 1, device=price_seq.device)

        # Early fusion: concatenate all representations
        fused = torch.cat([h_price, h_text, h_web, lag_h], dim=-1)
        logits = self.head(fused).squeeze(-1)

        if return_embeddings:
            return logits, h_price, h_text
        return logits


class BiLSTMFusion(nn.Module):
    """BiLSTM price encoder with text late-fusion.

    Classic recipe used in cryptocurrency / stock prediction papers combining
    price history with text signals (news, tweets).

    Architecture
    ------------
    Price:  BiLSTM(num_layers) over OHLCV sequence -> mean-pool -> h_price (d)
    Text:   Linear projection of sentence embedding -> h_text (d)
    Web:    Linear projection of web scalars        -> h_web  (d/2)
    Fuse:   concat [h_price, h_text, h_web, tau_lag] -> MLP -> logit

    References
    ----------
    Hochreiter & Schmidhuber (1997) Long Short-Term Memory.
    Xu & Cohen (2018) Stock Movement Prediction from Tweets and Historical
    Prices. ACL 2018.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        num_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        lstm_hidden = d_model // 2 if bidirectional else d_model

        self.price_proj = nn.Linear(price_dim, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.price_norm = nn.LayerNorm(d_model)

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = d_model * 2 + hidden_dim // 2 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes=None,
        return_embeddings: bool = False,
    ):
        x = self.price_proj(price_seq)
        out, _ = self.lstm(x)
        h_price = self.price_norm(out.mean(dim=1))

        h_text = self.text_proj(text_emb)
        h_web = self.web_proj(web_feat)

        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
        else:
            lag_h = torch.zeros(price_seq.size(0), 1, device=price_seq.device)

        fused = torch.cat([h_price, h_text, h_web, lag_h], dim=-1)
        logits = self.head(fused).squeeze(-1)

        if return_embeddings:
            return logits, h_price, h_text
        return logits


class MultimodalTransformer(nn.Module):
    """Multimodal Transformer (MulT) -- Tsai et al., ACL 2019.

    Directional cross-modal attention: each modality updates its tokens by
    attending over the other modality sequence.

        text_enriched  = CrossAttn(Q=text,  K=V=price_tokens)
        price_enriched = CrossAttn(Q=price, K=V=text_token)

    Both are mean-pooled, concatenated with web features, then classified.
    Unlike CGCMA there is no conditional gating: cross-modal information is
    always injected regardless of text quality or modality lag.

    Reference
    ---------
    Tsai et al. (2019) Multimodal Transformer for Unaligned Multimodal
    Language Sequences. ACL 2019.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        num_cm_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_len: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()

        self.price_proj = nn.Linear(price_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))
        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
        )

        self.text_from_price = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(num_cm_layers)
        ])
        self.text_from_price_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cm_layers)
        ])
        self.price_from_text = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(num_cm_layers)
        ])
        self.price_from_text_norm = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cm_layers)
        ])

        self.web_proj = nn.Sequential(
            nn.LayerNorm(web_dim),
            nn.Linear(web_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        head_in = d_model * 2 + hidden_dim // 2 + 1
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.normal_(self.positional, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes=None,
        return_embeddings: bool = False,
    ):
        batch_size, seq_len, _ = price_seq.shape

        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        price_tokens = self.price_encoder(price_tokens)

        text_token = self.text_proj(text_emb).unsqueeze(1)

        text_enriched = text_token
        price_enriched = price_tokens
        for attn, norm in zip(self.text_from_price, self.text_from_price_norm):
            delta, _ = attn(text_enriched, price_tokens, price_tokens, need_weights=False)
            text_enriched = norm(text_enriched + delta)

        for attn, norm in zip(self.price_from_text, self.price_from_text_norm):
            delta, _ = attn(price_enriched, text_token, text_token, need_weights=False)
            price_enriched = norm(price_enriched + delta)

        h_text = text_enriched.mean(dim=1)
        h_price = price_enriched.mean(dim=1)

        h_web = self.web_proj(web_feat)

        if lag_minutes is not None:
            lag_h = lag_minutes.float().unsqueeze(-1) / 60.0
        else:
            lag_h = torch.zeros(batch_size, 1, device=price_seq.device)

        fused = torch.cat([h_price, h_text, h_web, lag_h], dim=-1)
        logits = self.head(fused).squeeze(-1)

        if return_embeddings:
            return logits, h_price, h_text
        return logits


class TensorFusionNetwork(nn.Module):
    """Tensor Fusion Network (TFN) -- Zadeh et al., EMNLP 2017.

    3-way outer product of unimodal representations:
        z = [h_price; 1] ⊗ [h_text; 1] ⊗ [h_web; 1]
        fusion_dim = (r+1)^3  where r = tfn_rank

    The constant-1 appended to each subspace ensures that uni- and
    bi-modal interactions are included alongside the tri-modal term.

    Reference
    ---------
    Zadeh et al. (2017) Tensor Fusion Network for Multimodal Sentiment
    Analysis. EMNLP 2017.
    """

    def __init__(
        self,
        price_dim: int,
        text_dim: int,
        web_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        price_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_len: int = 256,
        tfn_rank: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        r = tfn_rank

        # Price encoder
        self.price_proj = nn.Linear(price_dim, d_model)
        self.positional = nn.Parameter(torch.zeros(1, max_len, d_model))
        price_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.price_encoder = nn.TransformerEncoder(price_layer, num_layers=price_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Unimodal subspace projections (-> r dims each)
        self.price_sub = nn.Sequential(nn.Linear(d_model, r), nn.Tanh())
        self.text_sub = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, r), nn.Tanh()
        )
        self.web_sub = nn.Sequential(
            nn.LayerNorm(web_dim), nn.Linear(web_dim, r), nn.Tanh()
        )

        fusion_dim = (r + 1) ** 3
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        nn.init.normal_(self.positional, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        price_seq: torch.Tensor,
        text_emb: torch.Tensor,
        web_feat: torch.Tensor,
        lag_minutes=None,
        return_embeddings: bool = False,
    ):
        batch_size, seq_len, _ = price_seq.shape

        # Price encoder -- CLS token
        price_tokens = self.price_proj(price_seq) + self.positional[:, :seq_len]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, price_tokens], dim=1)
        tokens = self.price_encoder(tokens)
        h_price = tokens[:, 0]  # (B, d)

        # Unimodal subspaces
        h_p = self.price_sub(h_price)   # (B, r)
        h_t = self.text_sub(text_emb)   # (B, r)
        h_w = self.web_sub(web_feat)    # (B, r)

        # Append constant-1 for lower-order interactions
        ones = torch.ones(batch_size, 1, device=price_seq.device)
        h_p1 = torch.cat([h_p, ones], dim=-1)  # (B, r+1)
        h_t1 = torch.cat([h_t, ones], dim=-1)
        h_w1 = torch.cat([h_w, ones], dim=-1)

        # 3-way outer product and flatten
        fused = torch.einsum("bi,bj,bk->bijk", h_p1, h_t1, h_w1)
        fused = fused.reshape(batch_size, -1)  # (B, (r+1)^3)

        logits = self.head(fused).squeeze(-1)

        if return_embeddings:
            return logits, h_price, h_t
        return logits
