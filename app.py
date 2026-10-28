"""Gradio Web UI cho ai_provenance - Hỗ trợ upload file (.docx, .md, .txt),
phân tích văn phong AI theo đoạn & toàn bài, highlight trực quan và so sánh provenance.
Tương thích hoàn toàn để deploy lên Hugging Face Spaces.
"""
from __future__ import annotations

import html
import json
import os
import sys
from collections import Counter
from pathlib import Path

import gradio as gr

# Đảm bảo mã hóa UTF-8 cho Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cli import read_document
from detectors import (
    DetectionResult,
    Detector,
    LlamaCppPerplexityDetector,
    PerplexityDetector,
    StylometryDetector,
    combine_detections,
    detect_language,
)
from hierarchy import (
    DocumentHierarchyResult,
    ParagraphResult,
    analyze_document_hierarchy,
)
from provenance import compute_provenance, split_paragraphs

# Khởi tạo detector mặc định chạy CPU
STYLOMETRY_DETECTOR = StylometryDetector()
PERPLEXITY_CACHED_DETECTORS: dict[str, Detector] = {}


def get_perplexity_detector(model_type: str, model_name_or_path: str) -> Detector:
    cache_key = f"{model_type}:{model_name_or_path}"
    if cache_key not in PERPLEXITY_CACHED_DETECTORS:
        if model_type == "gguf":
            PERPLEXITY_CACHED_DETECTORS[cache_key] = LlamaCppPerplexityDetector(
                model_path=model_name_or_path or "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                low=10.0,
                high=28.0,
            )
        else:
            lower = model_name_or_path.lower()
            if lower in {"gpt2", "distilgpt2", "gpt2-medium", "gpt2-large"}:
                PERPLEXITY_CACHED_DETECTORS[cache_key] = PerplexityDetector(model_name=model_name_or_path)
            else:
                # Hiệu chỉnh dải perplexity phù hợp cho model đa ngôn ngữ Qwen2.5-0.5B
                PERPLEXITY_CACHED_DETECTORS[cache_key] = PerplexityDetector(
                    model_name=model_name_or_path or "Qwen/Qwen2.5-0.5B",
                    low=18.0,
                    high=42.0,
                )
    return PERPLEXITY_CACHED_DETECTORS[cache_key]


def _render_sentences_with_spans(p_res: ParagraphResult) -> str:
    """Tô màu trực quan từng câu bên trong đoạn văn (kèm tooltip điểm ngữ cảnh)."""
    if not p_res.sentences:
        return html.escape(p_res.text)

    spans: list[str] = []
    for s in p_res.sentences:
        s_esc = html.escape(s.text)
        sc_str = f"{s.score:.3f}" if s.score is not None else "-"
        status_icon = "🤖 Câu AI" if s.label == "likely_ai" else "⚠️ Nghi vấn" if s.label == "uncertain" else "👤 Tự nhiên"
        note_str = f" • {s.note}" if s.note else ""
        tip = html.escape(f"Câu #{s.index + 1} ({s.word_count} từ) • {status_icon} (Điểm: {sc_str}){note_str}")
        if s.label == "likely_ai":
            cls = "sent-ai"
        elif s.label == "uncertain":
            cls = "sent-uncertain"
        elif s.label == "likely_human":
            cls = "sent-human"
        else:
            cls = "sent-unavail"
        spans.append(f'<span class="{cls}" title="{tip}">{s_esc}</span>')
    return " ".join(spans)


def render_html_highlighted_doc(doc_res: DocumentHierarchyResult) -> str:
    """Tạo giao diện HTML tô màu phân cấp: Mục -> Đoạn -> Câu trong tài liệu."""
    style = """
    <style>
        .doc-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.65;
            max-height: 650px;
            overflow-y: auto;
            padding: 18px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
        }
        .dark .doc-container {
            background: #1e293b;
            border-color: #334155;
            color: #f1f5f9;
        }
        .sec-banner {
            margin: 18px 0 10px 0;
            padding: 10px 14px;
            border-radius: 8px;
            background: #f1f5f9;
            border-left: 5px solid #3b82f6;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .dark .sec-banner {
            background: #0f172a;
            color: #e2e8f0;
        }
        .para-block {
            margin-bottom: 14px;
            padding: 12px 16px;
            border-radius: 8px;
            border-left: 5px solid transparent;
            transition: all 0.15s ease-in-out;
        }
        .para-block:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .para-ai {
            background-color: #fef2f2;
            border-left-color: #ef4444;
            color: #991b1b;
        }
        .dark .para-ai {
            background-color: #450a0a;
            border-left-color: #f87171;
            color: #fecaca;
        }
        .para-human {
            background-color: #f0fdf4;
            border-left-color: #22c55e;
            color: #166534;
        }
        .dark .para-human {
            background-color: #052e16;
            border-left-color: #4ade80;
            color: #bbf7d0;
        }
        .para-uncertain {
            background-color: #fefce8;
            border-left-color: #eab308;
            color: #854d0e;
        }
        .dark .para-uncertain {
            background-color: #422006;
            border-left-color: #facc15;
            color: #fef08a;
        }
        .para-unavailable {
            background-color: #f8fafc;
            border-left-color: #94a3b8;
            color: #64748b;
        }
        .dark .para-unavailable {
            background-color: #0f172a;
            border-left-color: #64748b;
            color: #94a3b8;
        }
        .para-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 6px;
            opacity: 0.9;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        .badge-ai { background: #fee2e2; color: #b91c1c; }
        .badge-human { background: #dcfce7; color: #15803d; }
        .badge-uncertain { background: #fef9c3; color: #a16207; }
        .badge-unavailable { background: #f1f5f9; color: #475569; }
        .para-text {
            font-size: 0.96rem;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .sent-ai {
            background-color: rgba(239, 68, 68, 0.32);
            border-bottom: 2.5px solid #dc2626;
            border-radius: 3px;
            padding: 1px 3px;
            font-weight: 500;
            cursor: help;
            transition: background-color 0.15s ease;
        }
        .sent-ai:hover {
            background-color: rgba(239, 68, 68, 0.52);
        }
        .dark .sent-ai {
            background-color: rgba(239, 68, 68, 0.45);
            border-bottom-color: #f87171;
        }
        .sent-uncertain {
            background-color: rgba(234, 179, 8, 0.22);
            border-bottom: 2px dashed #ca8a04;
            border-radius: 3px;
            padding: 1px 3px;
            cursor: help;
        }
        .dark .sent-uncertain {
            background-color: rgba(234, 179, 8, 0.35);
            border-bottom-color: #facc15;
        }
        .sent-human {
            border-bottom: 1.5px dotted rgba(34, 197, 94, 0.65);
            cursor: help;
        }
        .para-note {
            margin-top: 6px;
            font-size: 0.78rem;
            opacity: 0.8;
            font-style: italic;
        }
    </style>
    """
    legend = """
    <div style="margin-bottom: 12px; padding: 10px 14px; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 0.85rem; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
        <span style="font-weight: 700; color: #334155;">💡 Chú thích highlight câu:</span>
        <span style="color: #b91c1c;"><span style="display:inline-block; width:12px; height:12px; background:#ef4444; border-radius:2px; vertical-align:middle; margin-right:5px;"></span><b>Đỏ đậm:</b> Khả năng cao AI (Hover để xem điểm)</span>
        <span style="color: #b45309;"><span style="display:inline-block; width:12px; height:12px; background:#eab308; border-radius:2px; vertical-align:middle; margin-right:5px;"></span><b>Vàng:</b> Nghi vấn / Hỗn hợp</span>
        <span style="color: #15803d;"><span style="display:inline-block; width:12px; height:12px; background:#22c55e; border-radius:2px; vertical-align:middle; margin-right:5px;"></span><b>Xanh:</b> Tự nhiên (Người viết)</span>
    </div>
    """
    html_parts = [style, legend, '<div class="doc-container">']
    sec_by_header = {s.header_para_index: s for s in doc_res.sections if s.header_para_index is not None}

    for p_res in doc_res.paragraphs:
        idx = p_res.index + 1
        if p_res.is_header and p_res.index in sec_by_header:
            sec = sec_by_header[p_res.index]
            if sec.label == "likely_ai":
                b_cls, b_txt = "badge-ai", f"🤖 Mục AI ({sec.weighted_score:.3f})" if sec.weighted_score is not None else "🤖 Mục AI"
            elif sec.label == "likely_human":
                b_cls, b_txt = "badge-human", f"👤 Mục Người viết ({sec.weighted_score:.3f})" if sec.weighted_score is not None else "👤 Mục Người viết"
            elif sec.label == "uncertain":
                b_cls, b_txt = "badge-uncertain", f"⚠️ Có can thiệp AI ({sec.weighted_score:.3f})" if sec.weighted_score is not None else "⚠️ Nghi vấn"
            else:
                b_cls, b_txt = "badge-unavailable", "⚪ Chưa đủ dữ liệu"
            html_parts.append(f"""
            <div class="sec-banner">
                <span>📑 {html.escape(sec.title)} ({sec.valid_paragraphs} đoạn nội dung &bull; {sec.total_words} từ)</span>
                <span class="badge {b_cls}">{b_txt}</span>
            </div>
            """)
            continue

        label = p_res.label
        score_val = p_res.paragraph_score
        score_text = f"{score_val:.3f}" if score_val is not None else "-"

        if label == "likely_ai":
            cls_name = "para-ai"
            badge_cls = "badge-ai"
            badge_text = f"🤖 AI ({score_text})"
        elif label == "likely_human":
            cls_name = "para-human"
            badge_cls = "badge-human"
            badge_text = f"👤 Người viết ({score_text})"
        elif label == "uncertain":
            cls_name = "para-uncertain"
            badge_cls = "badge-uncertain"
            badge_text = f"⚠️ Nghi vấn ({score_text})"
        else:
            cls_name = "para-unavailable"
            badge_cls = "badge-unavailable"
            badge_text = "⚪ Quá ngắn / Tiêu đề"

        conf_tag = " • ⚠️ Đoạn ngắn (confidence=low)" if (p_res.low_confidence and score_val is not None) else ""
        rendered_sentences = _render_sentences_with_spans(p_res)
        note_str = html.escape(p_res.note)

        html_parts.append(f"""
        <div class="para-block {cls_name}">
            <div class="para-header">
                <span>Đoạn #{idx} ({html.escape(p_res.section_title)}) &bull; {p_res.language.upper()} &bull; {p_res.word_count} từ &bull; {len(p_res.sentences)} câu{conf_tag}</span>
                <span class="badge {badge_cls}">{badge_text}</span>
            </div>
            <div class="para-text">{rendered_sentences}</div>
            {f'<div class="para-note">{note_str}</div>' if note_str else ''}
        </div>
        """)
    html_parts.append("</div>")
    return "".join(html_parts)


def run_stylometry_analysis(
    file_obj,
    raw_text: str,
    window_size: int,
    ppl_choice: str,
    ppl_model_custom: str,
):
    """Xử lý phân tích văn phong tài liệu theo phân cấp Câu -> Đoạn -> Mục -> Toàn bài."""
    if file_obj is not None:
        text = read_document(file_obj.name)
    elif raw_text and raw_text.strip():
        text = raw_text.strip()
    else:
        return (
            "⚠️ Vui lòng tải lên file (.docx, .md, .txt) hoặc dán văn bản để bắt đầu kiểm tra.",
            "",
            [],
            None,
        )

    paras = split_paragraphs(text)
    if not paras:
        return ("⚠️ Tài liệu không có nội dung chữ hoặc văn bản rỗng.", "", [], None)

    dets = [STYLOMETRY_DETECTOR]
    if ppl_choice and ppl_choice != "none":
        try:
            if ppl_choice == "gguf":
                path = ppl_model_custom.strip() or "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
                ppl_det = get_perplexity_detector("gguf", path)
                dets.append(ppl_det)
            elif ppl_choice == "transformers":
                m_name = ppl_model_custom.strip() or "Qwen/Qwen2.5-0.5B"
                ppl_det = get_perplexity_detector("transformers", m_name)
                dets.append(ppl_det)
        except Exception as e:
            gr.Warning(f"Không thể khởi tạo Perplexity: {e}. Hệ thống sẽ dùng Stylometry thuần túy.")

    win = max(1, int(window_size))
    doc_res = analyze_document_hierarchy(text, dets, window_size=win)

    total_paras = doc_res.total_paragraphs
    counts = Counter(p.label for p in doc_res.paragraphs if not p.is_header)
    n_ai = counts.get("likely_ai", 0)
    n_human = counts.get("likely_human", 0)
    n_uncertain = counts.get("uncertain", 0)
    n_unavail = sum(1 for p in doc_res.paragraphs if p.is_header or p.label == "unavailable")
    valid_paras = doc_res.valid_paragraphs
    ai_percent = (n_ai / valid_paras * 100.0) if valid_paras > 0 else 0.0

    if doc_res.label == "likely_ai":
        overall_color = "#ef4444"
    elif doc_res.label == "uncertain":
        overall_color = "#eab308"
    elif doc_res.label == "likely_human":
        overall_color = "#22c55e"
    else:
        overall_color = "#64748b"

    # Bảng tổng hợp theo từng Mục (Bài 1, Bài 2...) nếu có tách mục
    section_rows_html = ""
    if len(doc_res.sections) >= 1 and (len(doc_res.sections) > 1 or doc_res.sections[0].header_para_index is not None):
        sec_items_html = []
        for s in doc_res.sections:
            sc_txt = f"{s.weighted_score:.3f}" if s.weighted_score is not None else "-"
            sec_items_html.append(
                f"<tr style='border-bottom: 1px solid #e2e8f0;'>"
                f"<td style='padding: 6px 10px; font-weight: 600;'>{html.escape(s.title)}</td>"
                f"<td style='padding: 6px 10px;'>{s.valid_paragraphs} đoạn ({s.high_conf_paragraphs} đoạn đủ dài, {s.total_words} từ)</td>"
                f"<td style='padding: 6px 10px; font-weight: 700;'>{sc_txt}</td>"
                f"<td style='padding: 6px 10px;'>{html.escape(s.verdict_text)}</td>"
                f"</tr>"
            )
        section_rows_html = f"""
        <div style="margin-top: 12px; border-top: 1px solid #cbd5e1; padding-top: 10px;">
            <div style="font-weight: 700; margin-bottom: 6px; font-size: 0.92rem;">📑 Kết luận riêng theo từng Mục / Bài:</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                <thead>
                    <tr style="background: rgba(0,0,0,0.04); text-align: left;">
                        <th style="padding: 6px 10px;">Mục</th>
                        <th style="padding: 6px 10px;">Quy mô</th>
                        <th style="padding: 6px 10px;">Điểm TB</th>
                        <th style="padding: 6px 10px;">Phán quyết</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(sec_items_html)}
                </tbody>
            </table>
        </div>
        """

    summary_html = f"""
    <div style="background: linear-gradient(135deg, {overall_color}15, {overall_color}05);
                border: 2px solid {overall_color}; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h3 style="margin: 0 0 10px 0; color: {overall_color}; font-size: 1.15rem; font-weight: 700;">
            {html.escape(doc_res.verdict_text)}
        </h3>
        <div style="display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.95rem;">
            <span>📄 <b>Tổng số đoạn:</b> {total_paras} ({doc_res.high_conf_paragraphs} đoạn đủ tin cậy)</span>
            <span style="color: #ef4444;">🤖 <b>Đoạn AI:</b> {n_ai} ({ai_percent:.1f}%)</span>
            <span style="color: #22c55e;">👤 <b>Người viết:</b> {n_human}</span>
            <span style="color: #eab308;">⚠️ <b>Nghi vấn:</b> {n_uncertain}</span>
            <span style="color: #64748b;">⚪ <b>Tiêu đề/Ngắn:</b> {n_unavail}</span>
        </div>
        {section_rows_html}
    </div>
    """

    highlighted_html = render_html_highlighted_doc(doc_res)

    table_data = []
    json_rows = []
    for p_res in doc_res.paragraphs:
        sc = f"{p_res.paragraph_score:.3f}" if p_res.paragraph_score is not None else "-"
        preview = p_res.text[:90] + ("..." if len(p_res.text) > 90 else "")
        table_data.append([
            p_res.index + 1,
            f"[{p_res.section_title}] {p_res.label}",
            sc,
            p_res.language.upper(),
            preview,
            p_res.note,
        ])
        json_rows.append({
            "index": p_res.index,
            "section": p_res.section_title,
            "is_header": p_res.is_header,
            "language": p_res.language,
            "word_count": p_res.word_count,
            "low_confidence": p_res.low_confidence,
            "text": p_res.text,
            "own_score": p_res.own_detection.score,
            "sentence_mean_score": p_res.sentence_mean_score,
            "paragraph_score": p_res.paragraph_score,
            "label": p_res.label,
            "note": p_res.note,
            "sentences": [
                {
                    "index": s.index,
                    "word_count": s.word_count,
                    "score": s.score,
                    "label": s.label,
                    "text": s.text,
                }
                for s in p_res.sentences
            ],
        })

    # Xuất file tải về
    out_dir = Path("task")
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "web_report.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_paragraphs": total_paras,
                    "valid_paragraphs": valid_paras,
                    "high_conf_paragraphs": doc_res.high_conf_paragraphs,
                    "weighted_score": doc_res.weighted_score,
                    "high_ai_ratio": doc_res.high_ai_ratio,
                    "label": doc_res.label,
                    "verdict": doc_res.verdict_text,
                    "ai_percentage": round(ai_percent, 2),
                    "counts": dict(counts),
                    "sections": [
                        {
                            "title": s.title,
                            "total_words": s.total_words,
                            "valid_paragraphs": s.valid_paragraphs,
                            "high_conf_paragraphs": s.high_conf_paragraphs,
                            "weighted_score": s.weighted_score,
                            "high_ai_ratio": s.high_ai_ratio,
                            "label": s.label,
                            "verdict": s.verdict_text,
                        }
                        for s in doc_res.sections
                    ],
                },
                "paragraphs": json_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        summary_html,
        highlighted_html,
        table_data,
        str(json_path),
    )


def run_provenance_analysis(final_file, final_raw: str, draft_file, draft_raw: str):
    """So khớp độ chỉnh sửa giữa bản cuối và bản nháp AI."""
    f_text = read_document(final_file.name) if final_file is not None else (final_raw or "").strip()
    d_text = read_document(draft_file.name) if draft_file is not None else (draft_raw or "").strip()

    if not f_text or not d_text:
        return "⚠️ Vui lòng cung cấp cả Bản Báo Cáo Cuối và Bản Nháp AI để so sánh provenance.", []

    results = compute_provenance(f_text, d_text)
    counts = Counter(p.category for p in results)
    total = len(results)

    n_unchanged = counts.get("ai_mostly_unchanged", 0)
    n_edited = counts.get("ai_edited", 0)
    n_own = counts.get("mostly_own", 0)

    summary_html = f"""
    <div style="border: 2px solid #3b82f6; border-radius: 12px; padding: 18px; background: #eff6ff;">
        <h3 style="margin: 0 0 10px 0; color: #1d4ed8; font-size: 1.15rem;">
            📊 Kết quả Phân tích Nguồn gốc (Provenance Tracking)
        </h3>
        <div style="display: flex; gap: 20px; font-size: 0.95rem;">
            <span>📑 <b>Tổng số đoạn:</b> {total}</span>
            <span style="color: #dc2626;">🔴 <b>Giữ nguyên từ AI (&ge;85%):</b> {n_unchanged} ({(n_unchanged/total*100):.1f}%)</span>
            <span style="color: #d97706;">🟡 <b>Biên tập lại (45-84%):</b> {n_edited} ({(n_edited/total*100):.1f}%)</span>
            <span style="color: #16a34a;">🟢 <b>Tự viết mới (&lt;45%):</b> {n_own} ({(n_own/total*100):.1f}%)</span>
        </div>
    </div>
    """

    table_data = []
    for i, p in enumerate(results):
        preview = p.final_text[:90] + ("..." if len(p.final_text) > 90 else "")
        table_data.append([
            i + 1,
            p.category,
            f"{p.similarity:.3f}",
            f"{p.sentence_coverage:.3f}",
            preview,
        ])

    return summary_html, table_data


# Xây dựng giao diện Gradio
CUSTOM_CSS = """
footer { visibility: hidden !important; }
.gr-button-primary { background: #2563eb !important; border-color: #1d4ed8 !important; }
"""

with gr.Blocks(title="Check AI Tools - AI Stylometry & Provenance") as demo:
    gr.Markdown("""
    # 🔍 AI Provenance & Stylometry Toolkit
    ### Công cụ nhận diện văn phong AI và theo dõi mức độ chỉnh sửa tài liệu (.docx, .md, .txt)
    """)

    with gr.Tabs():
        # TAB 1: STYLOMETRY
        with gr.TabItem("🔍 Quét Văn Phong AI (Stylometry Scanner)"):
            with gr.Row():
                with gr.Column(scale=4):
                    file_input = gr.File(
                        label="1. Tải lên tài liệu (.docx, .md, .txt)",
                        file_types=[".docx", ".md", ".txt"],
                    )
                    text_input = gr.Textbox(
                        label="Hoặc dán văn bản trực tiếp vào đây",
                        placeholder="Dán nội dung văn bản tiếng Việt hoặc tiếng Anh vào đây...",
                        lines=7,
                    )
                    with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                        window_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1,
                            label="Kích thước cửa sổ gộp đoạn (--window N)",
                            info="Mặc định là 1 (từng đoạn). Tăng lên 2 hoặc 3 nếu tài liệu có nhiều câu ngắn rời rạc.",
                        )
                        ppl_radio = gr.Radio(
                            choices=[
                                ("Tắt Perplexity (Khuyên dùng - Chạy CPU ~1s, độ an toàn cao)", "none"),
                                ("Qwen2.5-0.5B (Transformers - Nhanh ~20s, hỗ trợ tiếng Việt)", "transformers"),
                                ("Qwen2.5-3B-Instruct (GGUF llama.cpp - Phân biệt AI tách bạch hơn, ~5 phút)", "gguf"),
                            ],
                            value="none",
                            label="Tùy chọn Perplexity Detector",
                        )
                        ppl_model_custom = gr.Textbox(
                            value="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
                            label="Model HuggingFace hoặc Đường dẫn file .gguf",
                            info="Mặc định 'Qwen2.5-3B-Instruct-Q4_K_M.gguf' cho GGUF hoặc 'Qwen/Qwen2.5-0.5B' cho Transformers.",
                        )
                    btn_run = gr.Button("🚀 Bắt đầu Phân tích", variant="primary", size="lg")

                with gr.Column(scale=6):
                    summary_output = gr.HTML(label="Kết luận tổng quan")
                    with gr.Tabs():
                        with gr.TabItem("📄 Văn bản tô màu trực quan (Highlighted View)"):
                            highlight_output = gr.HTML(label="Nội dung trực quan")
                        with gr.TabItem("📋 Bảng số liệu chi tiết"):
                            table_output = gr.Dataframe(
                                headers=["Đoạn #", "Phán quyết", "Điểm số", "Ngôn ngữ", "Trích đoạn", "Ghi chú kỹ thuật"],
                                datatype=["number", "str", "str", "str", "str", "str"],
                                wrap=True,
                            )
                    download_json = gr.File(label="Tải về báo cáo JSON đầy đủ", visible=True)

            btn_run.click(
                fn=run_stylometry_analysis,
                inputs=[file_input, text_input, window_slider, ppl_radio, ppl_model_custom],
                outputs=[summary_output, highlight_output, table_output, download_json],
            )

        # TAB 2: PROVENANCE
        with gr.TabItem("📝 So Sánh Nguồn Gốc (Provenance Tracker)"):
            gr.Markdown("So sánh văn bản cuối cùng với bản nháp AI để biết chính xác từng đoạn văn bạn đã tự viết bao nhiêu % và giữ nguyên bao nhiêu %.")
            with gr.Row():
                with gr.Column():
                    f_file = gr.File(label="Bản Báo Cáo Cuối (.docx, .md, .txt)", file_types=[".docx", ".md", ".txt"])
                    f_text = gr.Textbox(label="Hoặc dán bản cuối vào đây", lines=5)
                with gr.Column():
                    d_file = gr.File(label="Bản Nháp AI Gốc (.docx, .md, .txt)", file_types=[".docx", ".md", ".txt"])
                    d_text = gr.Textbox(label="Hoặc dán bản nháp AI vào đây", lines=5)

            btn_prov = gr.Button("⚖️ So Khớp Provenance", variant="primary", size="lg")
            prov_summary = gr.HTML()
            prov_table = gr.Dataframe(
                headers=["Đoạn #", "Phân loại", "Độ khớp đoạn (Sim)", "Bao phủ câu (Cov)", "Trích đoạn"],
                datatype=["number", "str", "str", "str", "str"],
                wrap=True,
            )
            btn_prov.click(
                fn=run_provenance_analysis,
                inputs=[f_file, f_text, d_file, d_text],
                outputs=[prov_summary, prov_table],
            )

        # TAB 3: DEPLOY HUGGING FACE
        with gr.TabItem("☁️ Hướng Dẫn Deploy Hugging Face Spaces"):
            gr.Markdown("""
            ### 🌐 Hướng dẫn 3 bước đưa App lên Hugging Face Spaces (Hoàn toàn Miễn phí)

            Hugging Face Spaces hỗ trợ native cho Gradio. Bạn chỉ cần đưa mã nguồn lên là có link web công khai vĩnh viễn:

            #### Bước 1: Tạo Space trên Hugging Face
            1. Đăng nhập vào [Hugging Face](https://huggingface.co/) (tạo tài khoản nếu chưa có).
            2. Truy cập [huggingface.co/new-space](https://huggingface.co/new-space).
            3. Đặt **Space name** (ví dụ: `ai-text-checker`).
            4. Chọn **Space SDK** là **`Gradio`**.
            5. Chọn **Space hardware** là **`CPU Basic (Free - 16GB RAM)`** &rarr; Bấm **Create Space**.

            #### Bước 2: Tải code lên Space
            Bạn có thể dùng Git hoặc kéo thả trực tiếp trên giao diện trình duyệt của Hugging Face:
            - **Cách 1: Kéo thả trên giao diện web (Đơn giản nhất)**:
              Vào tab **Files and versions** trên Space vừa tạo, bấm **Add file &rarr; Upload files** và tải các file sau lên:
              - `app.py`
              - `hierarchy.py`
              - `detectors.py`
              - `provenance.py`
              - `cli.py`
              - `requirements.txt`
              - `README.md`
              - File mẫu nếu muốn: `Test AI.docx`, `sample_draft.md`, `sample_final.md`.

            - **Cách 2: Đẩy bằng Git CLI**:
              ```bash
              git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
              cd YOUR_SPACE_NAME
              # Copy các file app.py, hierarchy.py, detectors.py, provenance.py, cli.py, requirements.txt vào đây
              git add .
              git commit -m "Deploy AI Provenance Web App"
              git push
              ```

            #### Bước 3: Xem thành quả
            Sau khi tải file lên, Hugging Face sẽ tự động cài đặt `requirements.txt` và khởi chạy `app.py`. Chỉ sau khoảng 30 giây, Space của bạn sẽ hiển thị chữ **Running** và bạn có ngay một đường link public (ví dụ: `https://huggingface.co/spaces/username/ai-text-checker`) để chia sẻ cho bất cứ ai sử dụng!
            """)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Khởi động Web UI tại http://127.0.0.1:{port} ...")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, theme=gr.themes.Soft(), css=CUSTOM_CSS)
