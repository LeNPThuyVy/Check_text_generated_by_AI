"""Chạy: py -3.12 cli.py --final bao_cao.md [--draft ai_draft.md] [--detectors stylometry,perplexity,anthropic_watermark]"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from detectors import (
    AnthropicWatermarkDetector,
    DetectionResult,
    Detector,
    DummyDetector,
    StylometryDetector,
    combine_detections,
    detect_language,
)
from provenance import compute_provenance, split_paragraphs


def read_document(path_str: str) -> str:
    """Đọc văn bản từ file .md/.txt hoặc .docx (hỗ trợ cả python-docx lẫn fallback đọc XML chuẩn)."""
    path = Path(path_str)
    if path.suffix.lower() != ".docx":
        return path.read_text(encoding="utf-8")

    try:
        import docx  # type: ignore

        doc = docx.Document(str(path))
        paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paras)
    except ImportError:
        # Fallback giải nén word/document.xml trực tiếp bằng thư viện chuẩn Python
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paras = []
        for p in root.findall(".//w:p", ns):
            texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
            joined = "".join(texts).strip()
            if joined:
                paras.append(joined)
        return "\n\n".join(paras)


def build_detectors(names: list[str], model_name: str = "gpt2", ppl_low: float = 20.0, ppl_high: float = 60.0) -> list[Detector]:
    dets: list[Detector] = []
    for n in [x.strip() for x in names if x.strip()]:
        if n == "perplexity":
            from detectors import PerplexityDetector  # import trễ để khỏi bắt buộc có torch

            dets.append(PerplexityDetector(model_name=model_name, low=ppl_low, high=ppl_high))
        elif n == "stylometry":
            dets.append(StylometryDetector())
        elif n == "perplexity_gguf":
            from detectors import LlamaCppPerplexityDetector

            dets.append(LlamaCppPerplexityDetector(model_path=model_name, low=ppl_low, high=ppl_high))
        elif n == "anthropic_watermark":
            dets.append(AnthropicWatermarkDetector())
        elif n == "dummy":
            dets.append(DummyDetector())
        else:
            raise SystemExit(f"Detector không hợp lệ: {n}")
    return dets


def render_markdown_report(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# Báo cáo kiểm tra AI Provenance & Stylometry",
        "",
        "> **Lưu ý**: Các chỉ số thống kê/văn phong dùng ngưỡng mặc định chưa hiệu chỉnh trên bộ dữ liệu riêng, chỉ mang tính tham khảo, không phải bằng chứng kết luận tác giả.",
        "",
        "## Tổng quan",
        f"- **Tổng số đoạn**: {summary.get('total_paragraphs', 0)}",
        f"- **Ngôn ngữ phát hiện**: `{json.dumps(summary.get('languages', {}), ensure_ascii=False)}`",
        f"- **Kết luận gộp (Ensemble)**: `{json.dumps(summary.get('ensemble', {}), ensure_ascii=False)}`",
    ]
    if "provenance" in summary:
        lines.append(f"- **Provenance (so với draft)**: `{json.dumps(summary['provenance'], ensure_ascii=False)}`")

    lines.extend([
        "",
        "## Chi tiết từng đoạn",
        "",
        "| # | Ngôn ngữ | Ensemble | Provenance | Preview | Ghi chú chính |",
        "|---|---|---|---|---|---|",
    ])
    for r in rows:
        ens = r["ensemble"]
        ens_str = f"{ens['label']} ({ens['score']})" if ens["score"] is not None else ens["label"]
        prov = r.get("provenance")
        prov_str = f"{prov['category']} ({prov['similarity']})" if prov else "-"
        preview = r["preview"].replace("\n", " ").replace("|", "\\|")
        notes = "; ".join(f"{k}: {v['note']}" for k, v in r["detections"].items() if v.get("note"))
        notes = notes.replace("|", "\\|")
        lines.append(f"| {r['index']} | {r['language']} | {ens_str} | {prov_str} | {preview} | {notes} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="Bản báo cáo cuối (.md, .txt, .docx)")
    ap.add_argument("--draft", help="Bản AI draft gốc (để tính provenance)")
    ap.add_argument("--detectors", default="stylometry,anthropic_watermark,perplexity")
    ap.add_argument(
        "--model",
        default="gpt2",
        help="Tên model HuggingFace cho PerplexityDetector (mặc định gpt2; gợi ý đa ngôn ngữ/tiếng Việt: Qwen/Qwen2.5-0.5B)",
    )
    ap.add_argument("--ppl-low", type=float, default=20.0, help="Ngưỡng perplexity thấp (mặc định 20.0)")
    ap.add_argument("--ppl-high", type=float, default=60.0, help="Ngưỡng perplexity cao (mặc định 60.0)")
    ap.add_argument("--window", type=int, default=1, help="Số đoạn liền kề gộp chung khi chấm detector (mặc định 1)")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--format", choices=["json", "md", "both"], default="json", help="Định dạng xuất báo cáo")
    args = ap.parse_args()

    final = read_document(args.final)
    dets = build_detectors(args.detectors.split(","), model_name=args.model, ppl_low=args.ppl_low, ppl_high=args.ppl_high)
    paras = split_paragraphs(final)
    prov = compute_provenance(final, read_document(args.draft)) if args.draft else None
    win = max(1, args.window)

    rows = []
    for i, p in enumerate(paras):
        eval_text = "\n\n".join(paras[i : i + win]) if win > 1 else p
        lang = detect_language(eval_text)
        det_objs: dict[str, DetectionResult] = {}
        row: dict = {"index": i, "language": lang, "preview": p[:80], "detections": {}}
        for d in dets:
            r = d.detect(eval_text)
            det_objs[d.name] = r
            row["detections"][d.name] = {"score": r.score, "label": r.label, "note": r.note}

        ens = combine_detections(det_objs)
        row["ensemble"] = {"score": ens.score, "label": ens.label, "note": ens.note}

        if prov:
            row["provenance"] = {
                "similarity": round(prov[i].similarity, 3),
                "sentence_coverage": round(prov[i].sentence_coverage, 3),
                "category": prov[i].category,
            }
        rows.append(row)

    summary: dict = {
        "total_paragraphs": len(paras),
        "languages": dict(Counter(r["language"] for r in rows)),
        "ensemble": dict(Counter(r["ensemble"]["label"] for r in rows)),
        "by_detector": {
            d.name: dict(Counter(r["detections"][d.name]["label"] for r in rows)) for d in dets
        },
    }
    if prov:
        summary["provenance"] = dict(Counter(x.category for x in prov))

    out_path = Path(args.out)
    if args.format in ("json", "both"):
        json_path = out_path if out_path.suffix.lower() == ".json" else out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps({"summary": summary, "paragraphs": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã xử lý {len(paras)} đoạn -> {json_path}")

    if args.format in ("md", "both"):
        md_path = out_path.with_suffix(".md")
        md_path.write_text(render_markdown_report(summary, rows), encoding="utf-8")
        print(f"Đã xuất báo cáo Markdown -> {md_path}")

    if "provenance" in summary:
        print("Provenance:", summary["provenance"])
    print("Ensemble:", summary["ensemble"])
    print("Lưu ý: kết quả detector chỉ mang tính tham khảo (ngưỡng mặc định chưa hiệu chỉnh), không phải bằng chứng ai là tác giả.")


if __name__ == "__main__":
    main()
