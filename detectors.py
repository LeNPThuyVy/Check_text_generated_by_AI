"""Detector adapters. Mọi detector đều trả về DetectionResult cho từng đoạn text.

Cắm detector mới = viết thêm 1 class kế thừa Detector.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

VIETNAMESE_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

TRANSITION_PREFIXES_VI = (
    "ngoài ra",
    "hơn nữa",
    "tuy nhiên",
    "mặt khác",
    "bên cạnh đó",
    "tóm lại",
    "nhìn chung",
    "đầu tiên",
    "thứ hai",
    "thứ ba",
    "cuối cùng",
    "do đó",
    "vì vậy",
    "đồng thời",
    "cụ thể là",
    "mặc dù vậy",
    "thay vì",
    "nhờ đó",
    "kết quả là",
    "tiếp nối",
    "về mặt ý nghĩa",
    "để giải đáp",
    "để trả lời",
    "nhờ những",
)

TRANSITION_PREFIXES_EN = (
    "furthermore",
    "moreover",
    "however",
    "nevertheless",
    "additionally",
    "in conclusion",
    "overall",
    "firstly",
    "secondly",
    "thirdly",
    "finally",
    "therefore",
    "consequently",
    "in addition",
    "specifically",
    "as a result",
    "subsequently",
)

INLINE_CONNECTORS_VI = (
    "đồng thời",
    "nhờ đó",
    "qua đó",
    "từ đó",
    "bên cạnh đó",
    "ngoài ra",
    "hơn nữa",
    "mặc dù vậy",
    "thay vì",
    "nhằm ",
    "thông qua việc",
    "dựa trên các",
    "từng bước",
    "về mặt ",
    "kết quả là",
    "cụ thể là",
    "tiêu biểu là",
    "nhờ những",
)

INLINE_CONNECTORS_EN = (
    "furthermore",
    "moreover",
    "additionally",
    "consequently",
    "subsequently",
    "simultaneously",
    "by leveraging",
    "in order to",
    "rather than",
    "specifically",
    "as a result",
)

CLICHE_PHRASES_VI = (
    "đóng vai trò quan trọng",
    "đóng một vai trò quan trọng",
    "đóng vai trò",
    "trong bối cảnh hiện nay",
    "trong thời đại công nghệ",
    "không thể phủ nhận",
    "mang lại nhiều lợi ích",
    "mở ra nhiều cơ hội",
    "đặt ra nhiều thách thức",
    "một cách toàn diện",
    "tối ưu hóa hiệu quả",
    "góp phần không nhỏ",
    "cần có sự phối hợp",
    "bước ngoặt",
    "ý nghĩa cốt lõi",
    "ý tưởng cốt lõi",
    "không chỉ dừng lại ở",
    "kế thừa trọn vẹn",
    "duy trì trọn vẹn",
    "bài học đáng chú ý",
    "đặt ra một câu hỏi",
    "như một lời khẳng định",
    "mang tính hệ thống",
    "ngang ngửa hoặc vượt trội",
    "độ chính xác vượt trội",
)

CLICHE_PHRASES_EN = (
    "plays a crucial role",
    "plays an important role",
    "plays a key role",
    "in today's fast-paced world",
    "in the realm of",
    "it is important to note",
    "it is worth noting",
    "delve into",
    "a testament to",
    "pave the way",
    "double-edged sword",
    "comprehensive overview",
    "unlock the potential",
    "paradigm shift",
    "core idea",
    "not only",
    "seamless integration",
    "unprecedented",
)


def detect_language(text: str) -> str:
    """Phát hiện ngôn ngữ: 'vi' | 'en' | 'mixed' dựa trên tỉ lệ từ có dấu tiếng Việt."""
    words = re.findall(r"\w+", text)
    if not words:
        return "en"
    vi_words = sum(1 for w in words if any(ch in VIETNAMESE_CHARS for ch in w))
    ratio = vi_words / len(words)
    if ratio >= 0.25:
        return "vi"
    if ratio >= 0.05:
        return "mixed"
    return "en"


@dataclass
class DetectionResult:
    score: float | None  # 0..1, càng cao càng "giống AI"; None = không đánh giá được
    label: str           # "likely_ai" | "uncertain" | "likely_human" | "unavailable"
    note: str = ""


class Detector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, text: str) -> DetectionResult: ...


class PerplexityDetector(Detector):
    """Detector thống kê: perplexity thấp => text 'dễ đoán' => khả năng cao do AI viết.

    Lưu ý: đây KHÔNG phải watermark detector. Chỉ là heuristic, sai được cả 2 chiều
    (văn phong trang trọng/ngắn của người vẫn có thể ra perplexity thấp).
    Cần: pip install torch transformers
    """

    name = "perplexity"

    def __init__(
        self,
        model_name: str = "gpt2",
        low: float = 20.0,
        high: float = 60.0,
        window_size: int = 512,
        stride: int = 256,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.low, self.high = low, high
        self.window_size = window_size
        self.stride = stride

    def _is_english_only_model(self) -> bool:
        lower = self.model_name.lower()
        return lower in {"gpt2", "distilgpt2", "gpt2-medium", "gpt2-large"}

    def detect(self, text: str) -> DetectionResult:
        lang = detect_language(text)
        ids = self.tok(text, return_tensors="pt", truncation=False).input_ids
        seq_len = ids.shape[1]
        if seq_len < 30:
            return DetectionResult(None, "unavailable", "Đoạn quá ngắn (<30 token) để đánh giá")

        losses: list[float] = []
        with self._torch.no_grad():
            for start in range(0, seq_len, self.stride):
                end = min(start + self.window_size, seq_len)
                chunk = ids[:, start:end]
                if chunk.shape[1] < 10:
                    break
                loss = self.model(chunk, labels=chunk).loss
                losses.append(float(loss))
                if end == seq_len:
                    break

        avg_loss = sum(losses) / len(losses)
        ppl = float(math.exp(min(avg_loss, 20.0)))
        # map perplexity -> score tuyến tính: ppl<=low => 1.0, ppl>=high => 0.0
        score = max(0.0, min(1.0, (self.high - ppl) / (self.high - self.low)))
        label = "likely_ai" if score > 0.7 else "likely_human" if score < 0.3 else "uncertain"

        notes = [f"perplexity={ppl:.1f}", f"lang={lang}", "ngưỡng mặc định chưa hiệu chỉnh"]
        if lang in ("vi", "mixed") and self._is_english_only_model():
            notes.append(f"CẢNH BÁO: model '{self.model_name}' chỉ tối ưu cho tiếng Anh, điểm tiếng Việt không đáng tin")
        return DetectionResult(round(score, 3), label, "; ".join(notes))


class LlamaCppPerplexityDetector(Detector):
    """Detector Perplexity sử dụng mô hình GGUF thông qua công cụ llama-perplexity.exe (llama.cpp).

    Ưu điểm: hỗ trợ nạp mô hình lượng tử hóa lớn (như Qwen2.5-3B-Instruct Q4_K_M) chạy trực tiếp
    trên CPU với RAM 16GB. Năng lực phân biệt văn phong AI vs Người viết (AUROC) cao hơn model 0.5B.
    """

    name = "perplexity_gguf"

    def __init__(
        self,
        model_path: str = "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        low: float = 10.0,
        high: float = 28.0,
        llama_bin: str | None = None,
    ):
        self.model_path = model_path
        self.low, self.high = low, high

        if llama_bin:
            self.llama_bin = llama_bin
        else:
            found = shutil.which("llama-perplexity")
            if not found:
                default_winget = (
                    r"C:\Users\MsVy\AppData\Local\Microsoft\WinGet\Packages"
                    r"\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-perplexity.exe"
                )
                if os.path.exists(default_winget):
                    found = default_winget
            self.llama_bin = found

    def detect(self, text: str) -> DetectionResult:
        if not self.llama_bin or not os.path.exists(self.llama_bin):
            return DetectionResult(None, "unavailable", "Không tìm thấy file thực thi llama-perplexity.exe")

        model_file = Path(self.model_path)
        if not model_file.is_absolute() and not model_file.exists():
            candidate = Path(__file__).resolve().parent / self.model_path
            if candidate.exists():
                model_file = candidate
        if not model_file.exists():
            return DetectionResult(None, "unavailable", f"Không tìm thấy model GGUF tại '{self.model_path}'")

        words = text.strip().split()
        if len(words) < 20:
            return DetectionResult(None, "unavailable", "Đoạn quá ngắn (<20 từ) để phân tích perplexity GGUF")

        # Lựa chọn danh sách context an toàn; llama-perplexity yêu cầu số token >= 2 * ctx.
        # Fallback thử từ context cao xuống context thấp hơn nếu đoạn văn không đủ token.
        if len(words) >= 90:
            ctx_candidates = [64, 32, 16]
        elif len(words) >= 45:
            ctx_candidates = [32, 16]
        else:
            ctx_candidates = [16]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tf:
            tf.write(text)
            temp_name = tf.name

        try:
            ppl = None
            used_ctx = ctx_candidates[-1]
            for ctx in ctx_candidates:
                cmd = [self.llama_bin, "-m", str(model_file), "-f", temp_name, "-c", str(ctx), "--ppl-stride", "0"]
                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
                combined_out = res.stdout + "\n" + res.stderr
                m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", combined_out)
                if not m:
                    m = re.search(r"PPL\s*=\s*([0-9.]+)", combined_out)
                if m:
                    ppl = float(m.group(1))
                    used_ctx = ctx
                    break

            if ppl is None:
                return DetectionResult(None, "unavailable", "Không trích xuất được PPL từ llama-perplexity")

            score = max(0.0, min(1.0, (self.high - ppl) / max(0.1, self.high - self.low)))
            label = "likely_ai" if score > 0.65 else "likely_human" if score < 0.35 else "uncertain"
            notes = [f"perplexity_3b={ppl:.1f}", f"ctx={used_ctx}", f"model={model_file.name}"]
            return DetectionResult(round(score, 3), label, "; ".join(notes))
        except Exception as e:
            return DetectionResult(None, "unavailable", f"Lỗi chạy llama-perplexity: {e}")
        finally:
            Path(temp_name).unlink(missing_ok=True)


class StylometryDetector(Detector):
    """Detector văn phong (stylometry) hỗ trợ cả tiếng Việt và tiếng Anh, chạy CPU không cần model.

    Đo 5 tín hiệu:
    1. Burstiness (hệ số biến thiên độ dài câu/mệnh đề): văn AI thường có độ dài câu đồng đều.
    2. Độ dài câu trung bình (mean_len): văn tổng hợp học thuật của AI thường có câu dài đều.
    3. Mật độ từ nối mở đầu câu (Tuy nhiên, Ngoài ra, Furthermore, Moreover...).
    4. Mật độ cụm liên kết học thuật trong câu (đồng thời, nhờ đó, thông qua việc, nhằm...).
    5. Cụm từ sáo rỗng hay gặp ở LLM + liệt kê đánh số nội dòng (1)... (2)... (3)...
    (Kèm chỉ số MATTR và cờ confidence=low trong phần ghi chú chẩn đoán).
    """

    name = "stylometry"

    @staticmethod
    def _split_sentences_and_clauses(text: str) -> tuple[list[str], list[str]]:
        raw_sents = [
            s.strip()
            for s in re.split(r"[.!?…]+(?:\s+|$)", text.strip())
            if len(re.findall(r"\w+", s)) >= 1
        ]
        if len(raw_sents) >= 3:
            # Nếu câu đầu tiên rất ngắn (<=10 từ) đóng vai trò tiêu đề phụ nội dòng
            # (ví dụ: "Masking.", "ConvNeXt V2.") và theo sau còn >=3 câu nội dung,
            # loại câu tiêu đề phụ khỏi phép tính độ biến thiên độ dài câu (cv_len).
            if len(raw_sents) >= 4 and len(re.findall(r"\w+", raw_sents[0])) <= 10:
                return raw_sents, raw_sents[1:]
            return raw_sents, raw_sents

        clause_sents: list[str] = []
        for s in raw_sents:
            w_count = len(re.findall(r"\w+", s))
            if w_count > 40 and (";" in s or re.search(r"\(\d+\)", s)):
                sub = [
                    p.strip()
                    for p in re.split(r";\s*|\s*(?=\(\d+\))", s)
                    if len(re.findall(r"\w+", p)) >= 6
                ]
                if len(sub) >= 3:
                    clause_sents.extend(sub)
                    continue
            clause_sents.append(s)
        return raw_sents, clause_sents

    def detect(self, text: str) -> DetectionResult:
        words = re.findall(r"\w+", text.lower())
        n_words = len(words)
        if n_words < 25:
            return DetectionResult(None, "unavailable", "Đoạn quá ngắn (<25 từ) để phân tích văn phong")

        lang = detect_language(text)
        raw_sents, clause_sents = self._split_sentences_and_clauses(text)
        n_sents = len(raw_sents)
        raw_lengths = [len(re.findall(r"\w+", s)) for s in raw_sents]
        mean_len = statistics.mean(raw_lengths) if raw_lengths else 0.0

        # 1. Burstiness (Coefficient of Variation = stdev / mean)
        if len(clause_sents) >= 3:
            c_lengths = [len(re.findall(r"\w+", s)) for s in clause_sents]
            c_mean = statistics.mean(c_lengths)
            cv = statistics.pstdev(c_lengths) / c_mean if c_mean > 0 else 0.0
            # CV <= 0.18 => rất đều (score ~1.0); CV >= 0.50 => rất đa dạng (score ~0.0)
            burst_score = max(0.0, min(1.0, (0.50 - cv) / (0.50 - 0.18)))
            burst_note = f"cv_len={cv:.2f}"
        else:
            burst_score = max(0.0, min(1.0, (mean_len - 25.0) / (55.0 - 25.0))) * 0.7
            burst_note = f"sentences={n_sents}(ít câu)"

        # 2. Độ dài câu trung bình (mean_len)
        len_score = max(0.0, min(1.0, (mean_len - 24.0) / (42.0 - 24.0)))

        # 3. Mật độ từ nối mở đầu câu
        prefixes = TRANSITION_PREFIXES_VI + TRANSITION_PREFIXES_EN
        trans_hits = 0
        for s in raw_sents:
            s_low = s.lower().lstrip(" -•*0123456789.)(“\"'")
            if any(s_low.startswith(p + " ") or s_low.startswith(p + ",") for p in prefixes):
                trans_hits += 1
        trans_ratio = trans_hits / max(1, n_sents)
        trans_score = min(1.0, trans_ratio / 0.40)

        # 4. Mật độ cụm liên kết học thuật trong câu (trên 100 từ)
        text_low = re.sub(r"\s+", " ", text.lower())
        conns = INLINE_CONNECTORS_VI + INLINE_CONNECTORS_EN
        conn_hits = sum(text_low.count(c) for c in conns)
        conn_per_100w = (conn_hits / max(1, n_words)) * 100.0
        conn_score = min(1.0, conn_per_100w / 2.2)

        # 5. Cụm từ sáo rỗng (clichés) + cấu trúc liệt kê đánh số nội dòng (1)... (2)... (3)...
        cliches = CLICHE_PHRASES_VI + CLICHE_PHRASES_EN
        matched_cliches = [c for c in cliches if c in text_low]
        has_numbered_enum = bool(re.search(r"\(1\).*\(2\).*\(3\)", text_low))
        cliche_count = len(matched_cliches) + (2 if has_numbered_enum else 0)
        cliche_score = min(1.0, cliche_count / 1.5)

        # Chỉ số MATTR (lưu trong note để tham khảo)
        window = 40
        if n_words >= window:
            ttrs = [len(set(words[i : i + window])) / window for i in range(0, n_words - window + 1, 10)]
            mattr = sum(ttrs) / len(ttrs)
        else:
            mattr = len(set(words)) / n_words

        low_confidence = len(clause_sents) < 3 or n_words < 65

        raw_score = (
            0.28 * burst_score
            + 0.22 * len_score
            + 0.18 * conn_score
            + 0.16 * trans_score
            + 0.16 * cliche_score
        )
        if low_confidence and n_words < 90 and cliche_count == 0 and trans_hits == 0:
            raw_score *= 0.75

        score = round(raw_score, 3)
        label = "likely_ai" if score > 0.58 else "likely_human" if score < 0.36 else "uncertain"

        note_parts = [
            f"lang={lang}",
            burst_note,
            f"mean_len={mean_len:.1f}",
            f"transitions={trans_hits}/{n_sents}",
            f"inline_connectors={conn_per_100w:.1f}/100w",
            f"cliches={cliche_count}",
            f"mattr={mattr:.2f}",
        ]
        if low_confidence:
            note_parts.append("confidence=low")
        return DetectionResult(score, label, "; ".join(note_parts))


class AnthropicWatermarkDetector(Detector):
    """Chỗ trống cho detector watermark chính thức của Anthropic.

    Khi Anthropic phát hành API, implement hàm detect() ở đây (gọi API, map kết quả
    sang DetectionResult). Cho tới lúc đó, trả về 'unavailable' để report nói rõ
    là chưa kiểm tra được, tránh gây hiểu nhầm là 'sạch'.
    """

    name = "anthropic_watermark"

    def detect(self, text: str) -> DetectionResult:
        return DetectionResult(None, "unavailable", "Chưa có detector công khai của Anthropic")


class DummyDetector(Detector):
    """Dùng để test pipeline không cần tải model."""

    name = "dummy"

    def detect(self, text: str) -> DetectionResult:
        return DetectionResult(0.5, "uncertain", "dummy")


def combine_detections(results: dict[str, DetectionResult]) -> DetectionResult:
    """Gộp kết quả nhiều detector: yêu cầu >=2 detector đồng thuận khi có >=2 detector khả dụng."""
    scored = [r for r in results.values() if r.score is not None]
    if not scored:
        return DetectionResult(None, "unavailable", "Không có detector nào khả dụng")

    avg_score = round(sum(r.score for r in scored if r.score is not None) / len(scored), 3)
    ai_votes = sum(1 for r in scored if r.label == "likely_ai")
    human_votes = sum(1 for r in scored if r.label == "likely_human")

    if len(scored) >= 2:
        if ai_votes >= 2:
            label = "likely_ai"
        elif human_votes >= 2:
            label = "likely_human"
        else:
            label = "uncertain"
    else:
        label = scored[0].label

    return DetectionResult(avg_score, label, f"active={len(scored)}, ai_votes={ai_votes}, human_votes={human_votes}")
