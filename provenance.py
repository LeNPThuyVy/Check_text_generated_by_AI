"""Provenance: so sánh bản AI draft với bản cuối để biết mỗi đoạn được bạn sửa bao nhiêu."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


def normalize_text(text: str) -> str:
    """Chuẩn hóa chữ thường và khoảng trắng để so khớp ổn định hơn."""
    return re.sub(r"\s+", " ", text.strip().lower())


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= 10]


@dataclass
class ParagraphProvenance:
    final_text: str
    best_draft_match: str | None
    similarity: float           # 0..1 kết hợp cả mức đoạn và mức câu
    sentence_coverage: float    # 0..1 tỉ lệ câu trong đoạn khớp mạnh với bất kỳ câu nào của draft
    category: str               # "ai_mostly_unchanged" | "ai_edited" | "mostly_own"


def categorize(sim: float) -> str:
    if sim >= 0.85:
        return "ai_mostly_unchanged"
    if sim >= 0.45:
        return "ai_edited"
    return "mostly_own"


def _sentence_coverage(final_para: str, draft_sentences_norm: list[str]) -> float:
    """Tính độ khớp trung bình của từng câu trong đoạn cuối so với tập câu draft (bắt trường hợp gộp/tách đoạn)."""
    f_sents = [normalize_text(s) for s in split_sentences(final_para)]
    if not f_sents or not draft_sentences_norm:
        return 0.0
    scores: list[float] = []
    for fs in f_sents:
        best_s = max((difflib.SequenceMatcher(None, ds, fs).ratio() for ds in draft_sentences_norm), default=0.0)
        scores.append(best_s)
    return sum(scores) / len(scores)


def compute_provenance(final_text: str, ai_draft_text: str) -> list[ParagraphProvenance]:
    finals = split_paragraphs(final_text)
    drafts = split_paragraphs(ai_draft_text)
    drafts_norm = [normalize_text(d) for d in drafts]
    draft_sentences_norm = [normalize_text(s) for d in drafts for s in split_sentences(d)]

    out: list[ParagraphProvenance] = []
    for f in finals:
        f_norm = normalize_text(f)
        best, best_para_sim = None, 0.0
        for d, d_norm in zip(drafts, drafts_norm):
            sim = difflib.SequenceMatcher(None, d_norm, f_norm).ratio()
            if sim > best_para_sim:
                best, best_para_sim = d, sim

        sent_cov = _sentence_coverage(f, draft_sentences_norm)
        effective_sim = max(best_para_sim, sent_cov)
        out.append(
            ParagraphProvenance(
                final_text=f,
                best_draft_match=best,
                similarity=effective_sim,
                sentence_coverage=sent_cov,
                category=categorize(effective_sim),
            )
        )
    return out
