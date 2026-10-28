---
title: AI Provenance & Stylometry Toolkit
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# AI Provenance & Stylometry Toolkit (`ai_provenance`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Hardware: CPU Ready](https://img.shields.io/badge/hardware-CPU%20Standalone-orange.svg)]()

> **Công cụ phân tích mức độ can thiệp của AI (Provenance) và nhận diện văn phong học thuật do AI sinh ra (Stylometry & Perplexity) trên văn bản Tiếng Việt và Tiếng Anh.**  
> Hỗ trợ trực tiếp các định dạng `.docx` (Microsoft Word), `.md` (Markdown), và `.txt`.

---

## English Summary (Overview)

`ai_provenance` is a dual-engine analysis toolkit designed to evaluate AI-generated content in academic and technical reports:
1. **Provenance Engine**: Compares a final document against an initial AI draft at both the sentence and paragraph levels (handling paragraph merges/splits). Classifies each paragraph into `ai_mostly_unchanged`, `ai_edited`, or `mostly_own`.
2. **Stylometry & Linguistic Engine**: Analyzes 5 distinct statistical and structural signals calibrated for formal writing (Burstiness / CV of sentence length, Mean sentence length, Sentence-initial transitions, Inline connectors, and AI Clichés / Inline enumeration). Runs 100% on CPU without requiring GPU or heavy model downloads.
3. **Multilingual Perplexity Engine**: Optional sliding-window perplexity evaluation using causal language models (tested with `Qwen/Qwen2.5-0.5B` for Vietnamese/multilingual and `gpt2` for English).
4. **Ensemble Consensus**: Combines multiple detector signals requiring consensus among active detectors to minimize false accusations.

---

## 🌟 Tính năng nổi bật (Key Features)

- **Theo dõi nguồn gốc văn bản (Text Provenance)**:
  - So khớp đa tầng: Kết hợp độ tương đồng cấp đoạn (`difflib`) và độ bao phủ cấp câu (`_sentence_coverage`).
  - Xử lý tốt cả trường hợp người viết gộp nhiều đoạn AI thành một đoạn, đảo lộn thứ tự câu, hoặc chỉnh sửa từ ngữ cục bộ.
- **Phân tích văn phong chuyên sâu (Stylometry Detector)**:
  - **Hệ số biến thiên độ dài câu (`cv_len`)**: Đo lường tính đều đặn máy móc của LLM; tự động tách các mệnh đề ngăn cách bởi dấu chấm phẩy `;` hoặc đánh số `(1)... (2)...` trong các câu ghép siêu dài.
  - **Độ dài câu trung bình (`mean_len`)**: Nhận diện đặc trưng câu ghép phức, dài đều của văn phong AI học thuật.
  - **Từ nối mở đầu câu (`transitions`)**: Đo lường mật độ câu bắt đầu bằng từ nối khuôn mẫu song ngữ (*Tuy nhiên, Ngoài ra, Bên cạnh đó, Furthermore, Moreover...*).
  - **Cụm liên kết học thuật nội dòng (`inline_connectors`)**: Tần suất các cụm từ logic trong câu trên mỗi 100 từ (*đồng thời, nhờ đó, thông qua việc, nhằm, dựa trên...*).
  - **Cụm từ sáo rỗng & Cấu trúc liệt kê (`cliches`)**: Bắt các mẫu câu đặc trưng của LLM và cấu trúc liệt kê nội dòng `(1)... (2)... (3)...`.
  - **Cờ độ tin cậy (`confidence=low`)**: Tự động cảnh báo với các đoạn ngắn (<65 từ hoặc <3 câu) để hạn chế tối đa việc phán đoán nhầm văn bản ngắn của con người.
- **Hỗ trợ Perplexity đa ngôn ngữ**:
  - Tính toán Perplexity bằng cửa sổ trượt (`window_size=512`, `stride=256`), không bị cắt cụt văn bản dài.
  - Hỗ trợ model đa ngôn ngữ như `Qwen/Qwen2.5-0.5B` cho văn bản Tiếng Việt.
- **Đọc tài liệu Word `.docx` độc lập**:
  - Tự động ưu tiên thư viện `python-docx` nếu có, hoặc tự động fallback đọc file XML chuẩn bên trong `.docx` qua thư viện chuẩn Python (không cần cài Microsoft Word).
- **Xuất báo cáo trực quan**:
  - Hỗ trợ xuất dữ liệu cấu trúc `JSON` và bảng Markdown `MD` trực quan.

---

## 📦 Cài đặt (Installation)

Yêu cầu môi trường: **Python 3.12+**

```bash
# Clone repository
git clone https://github.com/your-username/Check_AI_tools.git
cd Check_AI_tools
```

### Tùy chọn 1: Chế độ Siêu nhẹ (Pure CPU - Khuyên dùng)
Chạy phân tích văn phong (Stylometry), so khớp nguồn gốc (Provenance) và đọc file Word hoàn toàn trên CPU:

```bash
pip install python-docx
```
*(Nếu không cài `python-docx`, công cụ vẫn tự động đọc được `.docx` bằng bộ giải nén XML có sẵn trong Python).*

### Tùy chọn 2: Chế độ Đầy đủ (Kèm Perplexity Detector)
Nếu muốn sử dụng thêm bộ đo Perplexity thống kê bằng Language Model:

```bash
pip install python-docx torch transformers
```

---

## 🚀 Hướng dẫn sử dụng nhanh (Quick Start)

### 1. Khởi chạy Web UI trực quan (Gradio App)
Trải nghiệm giao diện web có **tô màu trực quan từng đoạn văn**, kéo thả file Word, và xem báo cáo tương tác:

```bash
py -3.12 app.py
```
Sau đó mở trình duyệt truy cập: `http://127.0.0.1:7860`

---

### 2. Phân tích văn phong qua dòng lệnh (CLI)
Chạy quét văn phong siêu nhanh trên CPU (không cần mạng, không cần tải model):

```bash
py -3.12 cli.py --final "Test AI.docx" --detectors stylometry --format both --out report
```

### 3. So khớp mức độ chỉnh sửa so với bản AI Draft (Provenance)
So sánh bản báo cáo cuối với bản nháp ban đầu do AI tạo ra:

```bash
py -3.12 cli.py --final sample_final.md --draft sample_draft.md --detectors stylometry --format both --out report_provenance
```

### 4. Kết hợp Perplexity đa ngôn ngữ (Tiếng Việt & Tiếng Anh)
Sử dụng model `Qwen/Qwen2.5-0.5B` để tính toán độ bất ngờ (Perplexity) kết hợp cùng Stylometry:

```bash
py -3.12 cli.py --final "Test AI.docx" --detectors stylometry,perplexity --model Qwen/Qwen2.5-0.5B --format both --out report_full
```

### 5. Gộp cửa sổ đoạn văn (`--window N`)
Đối với văn bản có nhiều đoạn văn ngắn, bạn có thể gộp `N` đoạn liên tiếp để phân tích mượt hơn:

```bash
py -3.12 cli.py --final "Test AI.docx" --detectors stylometry --window 2 --format both --out report_window
```

---

## 📊 Bảng tham số dòng lệnh (CLI Reference)

| Tham số | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `--final` | `str` | *(Bắt buộc)* | Đường dẫn file báo cáo cần kiểm tra (`.docx`, `.md`, `.txt`). |
| `--draft` | `str` | `None` | Đường dẫn file nháp AI gốc để tính toán độ can thiệp (Provenance). |
| `--detectors` | `str` | `stylometry,anthropic_watermark,perplexity` | Danh sách detector (ngăn cách bằng dấu phẩy): `stylometry`, `perplexity`, `anthropic_watermark`, `dummy`. |
| `--model` | `str` | `gpt2` | Tên model HuggingFace cho `PerplexityDetector` (vd: `gpt2`, `Qwen/Qwen2.5-0.5B`). |
| `--ppl-low` | `float` | `20.0` | Ngưỡng Perplexity thấp (ppl <= low thì score tiệm cận 1.0). |
| `--ppl-high` | `float` | `60.0` | Ngưỡng Perplexity cao (ppl >= high thì score tiệm cận 0.0). |
| `--window` | `int` | `1` | Số đoạn văn liền kề gộp chung khi đánh giá detector. |
| `--format` | `str` | `json` | Định dạng xuất: `json`, `md`, hoặc `both`. |
| `--out` | `str` | `report.json` | Tên file hoặc đường dẫn file kết quả xuất ra. |

---

## 🔬 Cơ chế kỹ thuật (How It Works)

### 1. Phân loại Provenance (So khớp nháp)
Dựa trên chỉ số tương đồng kết hợp (`effective_similarity = max(para_sim, sentence_coverage)`):
- **`ai_mostly_unchanged`** (`sim >= 0.85`): Đoạn văn hầu như giữ nguyên từ bản nháp AI.
- **`ai_edited`** (`0.45 <= sim < 0.85`): Đoạn văn đã được biên tập, thay đổi một phần cấu trúc hoặc từ ngữ.
- **`mostly_own`** (`sim < 0.45`): Đoạn văn do người dùng tự viết mới hoặc viết lại hoàn toàn.

### 2. Thuật toán Stylometry (Phân tích văn phong)
Tổng điểm văn phong được chuẩn hóa về thang `[0, 1]` dựa trên 5 tín hiệu trọng số:
$$\text{Score} = 0.28 \times \text{Burstiness} + 0.22 \times \text{MeanLen} + 0.18 \times \text{InlineConnectors} + 0.16 \times \text{Transitions} + 0.16 \times \text{Clichés}$$

- **`likely_ai`** (`Score > 0.58`): Có dấu hiệu văn phong máy móc rõ rệt.
- **`likely_human`** (`Score < 0.36`): Độ biến thiên tự nhiên cao, cấu trúc câu đa dạng.
- **`uncertain`** (`0.36 <= Score <= 0.58`): Vùng trung gian hoặc đoạn văn kỹ thuật trang trọng.
- **`confidence=low`**: Gắn cờ khi đoạn quá ngắn (<65 từ hoặc <3 câu) để tránh kết luận vội vã.

---

## 📈 Kết quả kiểm chứng thực nghiệm (Validation)

Thuật toán đã được kiểm chứng thực tế và hiệu chỉnh qua phương pháp **Leave-One-Model-Out (LOMO)** trên 20 đoạn văn đối chứng từ các model AI hàng đầu và văn bản người viết thực tế:

| Nguồn văn bản kiểm thử | Số đoạn | Kết quả nhận diện | Điểm trung bình |
|---|---|---|---|
| **NotebookLM** (Đoạn 1) | 3 | **3/3** `likely_ai` (100%) | `0.792` |
| **Claude** (Đoạn 3) | 5 | **3/5** `likely_ai` + **2/5** `uncertain` | `0.669` |
| **ChatGPT** (Đoạn 4) | 1 | **1/1** `likely_ai` (100%) | `0.900` |
| **Gemini** (Đoạn 5) | 1 | **1/1** `likely_ai` (100%) | `0.904` |
| **Người viết (Human)** (Đoạn 2) | 10 | **9/10** `likely_human` + **1/10** `uncertain` | `0.170` |

- **AUROC trên văn bản thực nghiệm**: Đạt **`1.000`** (mọi đoạn văn AI đều có điểm số cao hơn mọi đoạn văn người viết).
- **Tỉ lệ báo nhầm người viết (False Positive Rate)**: **`0.0%`** (không có đoạn văn người viết nào bị gán nhãn `likely_ai`).

---

## 🧩 Mở rộng Detector mới (Extensibility)

Hệ thống được thiết kế theo dạng Adapter Module. Để thêm một detector mới:

1. Kế thừa class `Detector` trong `detectors.py`:
   ```python
   from detectors import Detector, DetectionResult

   class CustomDetector(Detector):
       name = "custom"

       def detect(self, text: str) -> DetectionResult:
           # Logic phân tích của bạn
           score = 0.85
           label = "likely_ai" if score > 0.6 else "likely_human"
           return DetectionResult(score=score, label=label, note="Phân tích custom")
   ```
2. Đăng ký tên detector trong hàm `build_detectors` tại `cli.py`.

---

## ⚠️ Lưu ý & Giới hạn (Disclaimer)

- **Công cụ mang tính chất tham khảo kỹ thuật**: Các chỉ số văn phong (Stylometry) và độ bất ngờ thống kê (Perplexity) là các ước lượng xác suất, **không phải bằng chứng tuyệt đối** để kết luận tác giả.
- Văn bản của người viết có tính trang trọng cao (luận văn, báo cáo kỹ thuật) đôi khi có thể đạt điểm ở mức `uncertain`.
- Khuyến nghị luôn kết hợp kiểm tra nguồn gốc (`--draft`) khi có thể để có đánh giá khách quan và chính xác nhất.

---

## 📄 Bản quyền (License)

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
