"""Phân tích phân cấp Câu -> Đoạn -> Mục (Bài) -> Toàn tài liệu.

Thiết kế:
- Cấp Câu (Sentence): chấm từng câu kèm ngữ cảnh lân cận (±1 câu liền kề nếu câu ngắn)
  để đủ độ dài tối thiểu cho Stylometry. Chỉ dùng để tô màu gợi ý trực quan, không dùng
  độc lập để kết luận.
- Cấp Đoạn (Paragraph): kết hợp điểm bản thân đoạn (trọng số 0.7) và điểm trung bình
  cấp câu trong ngữ cảnh (trọng số 0.3).
- Cấp Mục / Bài (Section) & Toàn bài (Document): tự động tách mục theo các dòng tiêu đề ngắn
  (ví dụ: "Bài 1", "Đoạn 1") và tính điểm trung bình có trọng số theo số từ trên các đoạn
  đủ độ tin cậy (loại bỏ các đoạn quá ngắn hoặc confidence=low khi có đoạn đủ dài).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from detectors import (
    DetectionResult,
    Detector,
    combine_detections,
    detect_language,
)
from provenance import split_paragraphs, split_sentences


@dataclass
class SentenceResult:
    index: int
    text: str
    word_count: int
    score: float | None
    label: str
    note: str = ""


@dataclass
class ParagraphResult:
    index: int
    section_title: str
    text: str
    language: str
    word_count: int
    is_header: bool
    low_confidence: bool
    own_detection: DetectionResult
    detections: dict[str, DetectionResult]
    sentences: list[SentenceResult]
    sentence_mean_score: float | None
    paragraph_score: float | None
    label: str
    note: str = ""


@dataclass
class SectionResult:
    index: int
    title: str
    header_para_index: int | None
    paragraphs: list[ParagraphResult]
    total_words: int
    valid_paragraphs: int
    high_conf_paragraphs: int
    weighted_score: float | None
    high_ai_ratio: float
    label: str
    verdict_text: str


@dataclass
class DocumentHierarchyResult:
    sections: list[SectionResult]
    paragraphs: list[ParagraphResult]
    total_paragraphs: int
    valid_paragraphs: int
    high_conf_paragraphs: int
    weighted_score: float | None
    high_ai_ratio: float
    label: str
    verdict_text: str


def is_section_header(text: str) -> bool:
    """Nhận diện dòng tiêu đề ngắn tách mục (ví dụ: 'Bài 1', 'Đoạn 2', 'Phần I')."""
    stripped = text.strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) > 4:
        return False
    if stripped[-1] in ".!?…:;":
        return False
    return True


def split_sections(paragraphs: list[str]) -> list[dict]:
    """Tách danh sách đoạn văn thành các mục (Section) dựa trên tiêu đề ngắn.

    Nếu tài liệu không có tiêu đề ngắn nào, toàn bộ tài liệu được gom vào 1 mục 'Toàn bài'.
    """
    sections: list[dict] = []
    current: dict | None = None

    for idx, p in enumerate(paragraphs):
        if is_section_header(p):
            current = {
                "title": p.strip(),
                "header_para_index": idx,
                "items": [],
            }
            sections.append(current)
        else:
            if current is None:
                current = {
                    "title": "Toàn bài",
                    "header_para_index": None,
                    "items": [],
                }
                sections.append(current)
            current["items"].append((idx, p))

    # Loại bỏ các mục rỗng nếu có hai tiêu đề liên tiếp
    non_empty = [s for s in sections if s["items"] or s["header_para_index"] is not None]
    return non_empty


def _label_from_score(score: float | None, ai_thresh: float = 0.56, human_thresh: float = 0.36) -> str:
    if score is None:
        return "unavailable"
    if score >= ai_thresh:
        return "likely_ai"
    if score < human_thresh:
        return "likely_human"
    return "uncertain"


def score_sentences(paragraph: str, detectors: list[Detector]) -> list[SentenceResult]:
    """Chấm từng câu trong đoạn bằng cửa sổ ngữ cảnh lân cận (±1 câu liền kề).

    Vì các tín hiệu văn phong (burstiness, mật độ từ nối) không có ý nghĩa trên 1 câu ngắn
    đơn lẻ, mỗi câu dưới 35 từ được ghép với câu liền trước/sau để tạo cửa sổ ngữ cảnh >=25 từ.
    Kết quả cấp câu chỉ dùng để tô màu gợi ý trực quan.
    """
    sents = split_sentences(paragraph)
    if not sents:
        return []

    results: list[SentenceResult] = []
    for i, s_text in enumerate(sents):
        w_count = len(re.findall(r"\w+", s_text))
        ctx_parts = [s_text]
        used_context = False
        if w_count < 35 and len(sents) > 1:
            used_context = True
            if i > 0:
                ctx_parts.insert(0, sents[i - 1])
            if i + 1 < len(sents) and len(re.findall(r"\w+", " ".join(ctx_parts))) < 45:
                ctx_parts.append(sents[i + 1])

        eval_text = " ".join(ctx_parts)
        det_map: dict[str, DetectionResult] = {}
        for d in detectors:
            # Ở cấp câu, ưu tiên các detector nhanh (như stylometry) hoặc bộ detector đã truyền vào
            det_map[d.name] = d.detect(eval_text)

        ens = combine_detections(det_map)
        note_suffix = "ngữ cảnh ±1 câu (chỉ gợi ý trực quan)" if used_context else "câu độc lập (chỉ gợi ý trực quan)"
        note = f"{ens.note}; {note_suffix}" if ens.note else note_suffix
        results.append(
            SentenceResult(
                index=i,
                text=s_text,
                word_count=w_count,
                score=ens.score,
                label=ens.label,
                note=note,
            )
        )
    return results


def aggregate_paragraph(
    own_res: DetectionResult,
    sentence_results: list[SentenceResult],
) -> tuple[float | None, float | None, str, str]:
    """Gộp điểm cấp đoạn (trọng số 0.7) và trung bình cấp câu có ngữ cảnh (trọng số 0.3)."""
    if own_res.score is None:
        return None, None, "unavailable", own_res.note

    valid_sent_scores = [s.score for s in sentence_results if s.score is not None]
    if valid_sent_scores:
        sent_mean = round(statistics.mean(valid_sent_scores), 3)
        para_score = round(0.7 * own_res.score + 0.3 * sent_mean, 3)
    else:
        sent_mean = None
        para_score = round(own_res.score, 3)

    # Giữ phán quyết chính dựa trên đồng thuận của own_res kết hợp điểm phân cấp
    if own_res.label in ("likely_ai", "likely_human") and len(valid_sent_scores) < 2:
        label = own_res.label
    else:
        label = _label_from_score(para_score, ai_thresh=0.52, human_thresh=0.35)

    note = own_res.note
    if sent_mean is not None:
        note = f"{note}; own={own_res.score:.3f}, sent_ctx={sent_mean:.3f}, hier={para_score:.3f}"
    return sent_mean, para_score, label, note


def _verdict_for_collection(
    para_results: list[ParagraphResult],
) -> tuple[int, int, int, float | None, float, str, str]:
    """Tính điểm tổng hợp có trọng số theo số từ cho 1 Mục (Bài) hoặc Toàn tài liệu.

    Chỉ tính các đoạn đủ độ tin cậy (high-confidence: >=65 từ và >=3 câu/mệnh đề) nếu có;
    nếu mục chỉ toàn đoạn ngắn mới fallback sang toàn bộ các đoạn khả dụng.
    """
    content_paras = [p for p in para_results if not p.is_header]
    valid_paras = [p for p in content_paras if p.paragraph_score is not None]
    hc_paras = [p for p in valid_paras if not p.low_confidence]

    use_paras = hc_paras if hc_paras else valid_paras
    total_words = sum(p.word_count for p in content_paras)
    if not use_paras:
        return (
            total_words,
            0,
            0,
            None,
            0.0,
            "unavailable",
            "⚪ Không đủ văn bản để kết luận",
        )

    w_sum = sum(p.word_count for p in use_paras)
    # Sử dụng trung bình có trọng số của điểm cấp đoạn (kết hợp điểm đoạn gốc và điểm phân cấp)
    weighted_score = round(
        sum(((0.6 * (p.own_detection.score or 0.0) + 0.4 * (p.paragraph_score or 0.0)) * p.word_count) for p in use_paras)
        / max(1, w_sum),
        3,
    )
    high_ai_count = sum(
        1
        for p in use_paras
        if (p.own_detection.score is not None and p.own_detection.score >= 0.50)
        or p.label == "likely_ai"
    )
    high_ai_ratio = round(high_ai_count / len(use_paras), 3)

    if weighted_score >= 0.44 or (weighted_score >= 0.35 and high_ai_ratio >= 0.30):
        label = "likely_ai"
        verdict = (
            f"🚨 Khả năng cao do AI tạo ra "
            f"(điểm TB trọng số: {weighted_score:.3f}, {high_ai_count}/{len(use_paras)} đoạn điểm cao)"
        )
    elif weighted_score >= 0.33 or high_ai_ratio >= 0.25:
        label = "uncertain"
        verdict = (
            f"⚠️ Có dấu hiệu can thiệp AI / Văn phong hỗn hợp "
            f"(điểm TB trọng số: {weighted_score:.3f}, {high_ai_count}/{len(use_paras)} đoạn điểm cao)"
        )
    else:
        label = "likely_human"
        verdict = (
            f"✅ Phần lớn do con người viết "
            f"(điểm TB trọng số: {weighted_score:.3f}, {len(use_paras) - high_ai_count}/{len(use_paras)} đoạn tự nhiên)"
        )

    return (
        total_words,
        len(valid_paras),
        len(hc_paras),
        weighted_score,
        high_ai_ratio,
        label,
        verdict,
    )


def analyze_document_hierarchy(
    text: str,
    detectors: list[Detector],
    window_size: int = 1,
) -> DocumentHierarchyResult:
    """Chạy toàn bộ pipeline phân cấp Câu -> Đoạn -> Mục (Bài) -> Toàn tài liệu."""
    raw_paras = split_paragraphs(text)
    raw_sections = split_sections(raw_paras)
    win = max(1, int(window_size))

    # Chấm cấp câu bằng các detector nhẹ (Stylometry) để không làm chậm gấp nhiều lần khi bật Perplexity
    sentence_detectors = [d for d in detectors if d.name == "stylometry"] or detectors[:1]

    all_para_results: list[ParagraphResult] = []
    section_results: list[SectionResult] = []

    for sec_idx, sec in enumerate(raw_sections):
        sec_title = sec["title"]
        header_idx = sec["header_para_index"]
        sec_paras: list[ParagraphResult] = []

        if header_idx is not None:
            h_text = raw_paras[header_idx]
            h_res = ParagraphResult(
                index=header_idx,
                section_title=sec_title,
                text=h_text,
                language=detect_language(h_text),
                word_count=len(re.findall(r"\w+", h_text)),
                is_header=True,
                low_confidence=True,
                own_detection=DetectionResult(None, "unavailable", f"Tiêu đề mục ({sec_title})"),
                detections={},
                sentences=[],
                sentence_mean_score=None,
                paragraph_score=None,
                label="unavailable",
                note=f"Tiêu đề mục ({sec_title})",
            )
            all_para_results.append(h_res)

        items = sec["items"]
        for local_i, (p_idx, p_text) in enumerate(items):
            if win > 1:
                window_texts = [items[j][1] for j in range(local_i, min(len(items), local_i + win))]
                eval_text = "\n\n".join(window_texts)
            else:
                eval_text = p_text

            lang = detect_language(eval_text)
            w_count = len(re.findall(r"\w+", p_text))
            det_objs: dict[str, DetectionResult] = {}
            for d in detectors:
                det_objs[d.name] = d.detect(eval_text)

            own_ens = combine_detections(det_objs)
            # Xác định cờ confidence=low từ kết quả stylometry hoặc độ dài đoạn
            stylo_note = det_objs.get("stylometry", own_ens).note
            low_conf = ("confidence=low" in stylo_note) or (w_count < 65)

            sent_results = score_sentences(p_text, sentence_detectors) if w_count >= 25 else []
            sent_mean, para_score, para_label, para_note = aggregate_paragraph(own_ens, sent_results)

            p_res = ParagraphResult(
                index=p_idx,
                section_title=sec_title,
                text=p_text,
                language=lang,
                word_count=w_count,
                is_header=False,
                low_confidence=low_conf,
                own_detection=own_ens,
                detections=det_objs,
                sentences=sent_results,
                sentence_mean_score=sent_mean,
                paragraph_score=para_score,
                label=para_label,
                note=para_note,
            )
            sec_paras.append(p_res)
            all_para_results.append(p_res)

        (
            sec_words,
            sec_valid,
            sec_hc,
            sec_wscore,
            sec_hratio,
            sec_label,
            sec_verdict,
        ) = _verdict_for_collection(sec_paras)

        section_results.append(
            SectionResult(
                index=sec_idx,
                title=sec_title,
                header_para_index=header_idx,
                paragraphs=sec_paras,
                total_words=sec_words,
                valid_paragraphs=sec_valid,
                high_conf_paragraphs=sec_hc,
                weighted_score=sec_wscore,
                high_ai_ratio=sec_hratio,
                label=sec_label,
                verdict_text=sec_verdict,
            )
        )

    (
         _,
        doc_valid,
        doc_hc,
        doc_wscore,
        doc_hratio,
        doc_label,
        doc_verdict,
    ) = _verdict_for_collection(all_para_results)

    return DocumentHierarchyResult(
        sections=section_results,
        paragraphs=all_para_results,
        total_paragraphs=len(all_para_results),
        valid_paragraphs=doc_valid,
        high_conf_paragraphs=doc_hc,
        weighted_score=doc_wscore,
        high_ai_ratio=doc_hratio,
        label=doc_label,
        verdict_text=doc_verdict,
    )
