"""
Utilities for auditing and filtering multimodal web-intelligence history.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


ERROR_MARKERS = (
    "error code:",
    "insufficient_quota",
    "rate limit",
    "429",
    "server disconnected",
)

LOW_SIGNAL_EXACT = {
    "bearish",
    "bullish",
    "neutral",
    "fear",
    "greed",
    "extreme_fear",
    "extreme_greed",
    "high",
    "medium",
    "low",
    "exchange_inflow",
    "exchange_outflow",
    "exchange_inflows",
    "exchange_outflows",
}

TEMPLATE_PATTERNS = (
    re.compile(r"市场情绪.{0,16}(建议|投资者应|保持|关注|注意|避免|警惕)"),
    re.compile(r"近期市场.{0,16}(建议|投资者应|谨慎|关注|注意)"),
    re.compile(r"投资者应.{0,16}(谨慎|观望|注意|警惕|避免)"),
    re.compile(r"建议投资者.{0,20}(谨慎|观望|关注|注意|保持)"),
    re.compile(r"建议保持.{0,12}(观望|谨慎)"),
    re.compile(r"market sentiment.{0,24}(cautious|watch|observe|risk)"),
    re.compile(r"investors should.{0,24}(remain|stay|be)"),
    re.compile(r"monitor market developments"),
)


@dataclass
class QualityThresholds:
    min_text_char_len: int = 120
    min_text_source_count: int = 4
    max_text_error_count: int = 0
    max_template_ratio: float = 0.35
    min_quality_score: float = 0.45


def extract_quality_fields_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    web = record.get("web_intelligence") or {}
    return build_text_quality_fields(
        news_summary=web.get("news_summary") or {},
        fear_greed=web.get("fear_greed_index") or {},
        social=web.get("social_sentiment") or {},
        whale=web.get("whale_movements") or {},
        overall=web.get("overall_assessment") or {},
    )


def build_text_quality_fields(
    news_summary: Dict[str, Any],
    fear_greed: Dict[str, Any],
    social: Dict[str, Any],
    whale: Dict[str, Any],
    overall: Dict[str, Any],
) -> Dict[str, Any]:
    fragments: List[str] = []
    error_count = 0
    for blob in (news_summary, fear_greed, social, whale, overall):
        error_count += count_error_markers(blob)
        collect_text_fragments(blob, fragments)

    unique_fragments: List[str] = []
    seen: set[str] = set()
    for fragment in fragments:
        if fragment not in seen:
            unique_fragments.append(fragment)
            seen.add(fragment)

    template_fragments = [fragment for fragment in unique_fragments if is_template_like_fragment(fragment)]
    informative_fragments = [fragment for fragment in unique_fragments if fragment not in template_fragments]

    text_blob_raw = "\n".join(unique_fragments)[:4000]
    text_blob = "\n".join(informative_fragments or unique_fragments)[:4000]
    template_ratio = (
        float(len(template_fragments) / max(len(unique_fragments), 1)) if unique_fragments else 1.0
    )
    quality_score = compute_quality_score(
        text_char_len=len(text_blob),
        text_source_count=len(informative_fragments or unique_fragments),
        text_error_count=error_count,
        template_ratio=template_ratio,
    )

    return {
        "text_blob": text_blob,
        "text_blob_raw": text_blob_raw,
        "text_char_len": float(len(text_blob)),
        "text_source_count": float(len(informative_fragments or unique_fragments)),
        "text_error_count": float(error_count),
        "text_has_content": float(1 if text_blob else 0),
        "text_fragment_count": float(len(unique_fragments)),
        "text_template_count": float(len(template_fragments)),
        "text_template_ratio": float(template_ratio),
        "text_quality_score": float(quality_score),
        "text_fingerprint": text_fingerprint(text_blob),
        "text_raw_fingerprint": text_fingerprint(text_blob_raw),
    }


def collect_text_fragments(value: Any, fragments: List[str], depth: int = 0) -> None:
    if depth > 4 or value is None:
        return
    if isinstance(value, str):
        cleaned = clean_text_fragment(value)
        if cleaned:
            fragments.append(cleaned)
        return
    if isinstance(value, list):
        for item in value[:20]:
            collect_text_fragments(item, fragments, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).strip().lower() in {"error", "signal_breakdown"}:
                continue
            collect_text_fragments(item, fragments, depth + 1)


def clean_text_fragment(text: str) -> str:
    cleaned = re.sub(r"https?://\S+", " ", str(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 6:
        return ""
    lowered = cleaned.lower()
    if any(marker in lowered for marker in ERROR_MARKERS):
        return ""
    return cleaned


def count_error_markers(value: Any, depth: int = 0) -> int:
    if depth > 4 or value is None:
        return 0
    if isinstance(value, str):
        lowered = value.lower()
        return int(any(marker in lowered for marker in ERROR_MARKERS))
    if isinstance(value, list):
        return sum(count_error_markers(item, depth + 1) for item in value[:20])
    if isinstance(value, dict):
        count = 1 if "error" in {str(k).strip().lower() for k in value.keys()} else 0
        for item in value.values():
            count += count_error_markers(item, depth + 1)
        return count
    return 0


def is_template_like_fragment(text: str) -> bool:
    lowered = text.lower().strip()
    canonical = canonicalize_text(lowered)
    if lowered in LOW_SIGNAL_EXACT or canonical in LOW_SIGNAL_EXACT:
        return True
    if re.fullmatch(r"[a-z_]{3,24}", lowered) and "_" in lowered:
        return True
    return any(pattern.search(text) for pattern in TEMPLATE_PATTERNS)


def canonicalize_text(text: str) -> str:
    lowered = str(text).lower()
    lowered = re.sub(r"\b(btc|eth|sol)(usdt)?\b", "asset", lowered)
    lowered = re.sub(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", "date", lowered)
    lowered = re.sub(r"\d+[.,]?\d*", "#", lowered)
    lowered = re.sub(r"https?://\S+", " ", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def text_fingerprint(text: str) -> str:
    canonical = canonicalize_text(text)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16] if canonical else ""


def compute_quality_score(
    text_char_len: int,
    text_source_count: int,
    text_error_count: int,
    template_ratio: float,
) -> float:
    if text_char_len <= 0 or text_source_count <= 0:
        return 0.0
    len_score = min(float(text_char_len) / 400.0, 1.0)
    source_score = min(float(text_source_count) / 6.0, 1.0)
    template_score = max(0.0, 1.0 - float(template_ratio))
    error_penalty = min(float(text_error_count) / 2.0, 1.0)
    score = 0.45 * len_score + 0.35 * source_score + 0.20 * template_score - 0.30 * error_penalty
    return max(0.0, min(score, 1.0))


def evaluate_quality(
    fields: Dict[str, Any],
    thresholds: QualityThresholds,
    duplicate_count: Optional[int] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    if int(fields.get("text_char_len", 0)) < thresholds.min_text_char_len:
        reasons.append("short_text")
    if int(fields.get("text_source_count", 0)) < thresholds.min_text_source_count:
        reasons.append("few_sources")
    if int(fields.get("text_error_count", 0)) > thresholds.max_text_error_count:
        reasons.append("error_text")
    if float(fields.get("text_template_ratio", 1.0)) > thresholds.max_template_ratio:
        reasons.append("template_heavy")
    if float(fields.get("text_quality_score", 0.0)) < thresholds.min_quality_score:
        reasons.append("low_quality_score")
    if duplicate_count is not None and duplicate_count > 3:
        reasons.append("duplicate_text_group")
    return {
        "keep": int(len(reasons) == 0),
        "drop_reasons": reasons,
        "duplicate_count": int(duplicate_count or 1),
    }


def summarize_quality_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "rows": 0,
            "kept_rows": 0,
            "drop_rate": 0.0,
            "mean_text_char_len": 0.0,
            "mean_text_source_count": 0.0,
            "mean_template_ratio": 0.0,
            "mean_quality_score": 0.0,
            "duplicate_group_share": 0.0,
            "drop_reason_counts": {},
        }

    reason_counter: Counter[str] = Counter()
    duplicate_rows = 0
    kept_rows = 0
    for row in rows:
        if row.get("duplicate_count", 1) > 1:
            duplicate_rows += 1
        if row.get("keep", 0):
            kept_rows += 1
        for reason in row.get("drop_reasons", []):
            reason_counter[reason] += 1

    return {
        "rows": int(len(rows)),
        "kept_rows": int(kept_rows),
        "drop_rate": float(1.0 - (kept_rows / float(len(rows)))),
        "mean_text_char_len": float(sum(float(r.get("text_char_len", 0.0)) for r in rows) / len(rows)),
        "mean_text_source_count": float(sum(float(r.get("text_source_count", 0.0)) for r in rows) / len(rows)),
        "mean_template_ratio": float(sum(float(r.get("text_template_ratio", 0.0)) for r in rows) / len(rows)),
        "mean_quality_score": float(sum(float(r.get("text_quality_score", 0.0)) for r in rows) / len(rows)),
        "duplicate_group_share": float(duplicate_rows / float(len(rows))),
        "drop_reason_counts": dict(sorted(reason_counter.items())),
    }


def load_jsonl_rows(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows
