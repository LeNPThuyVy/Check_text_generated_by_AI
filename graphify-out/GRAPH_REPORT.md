# Graph Report - Check_AI_tools  (2026-10-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 88 nodes · 197 edges · 10 communities (5 shown, 5 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- hierarchy.py
- app.py
- cli.py
- provenance.py
- Detector
- PerplexityDetector
- StylometryDetector
- DetectionResult
- DummyDetector

## God Nodes (most connected - your core abstractions)
1. `analyze_document_hierarchy()` - 16 edges
2. `DetectionResult` - 15 edges
3. `Detector` - 12 edges
4. `compute_provenance()` - 11 edges
5. `PerplexityDetector` - 10 edges
6. `combine_detections()` - 9 edges
7. `main()` - 9 edges
8. `detect_language()` - 9 edges
9. `StylometryDetector` - 8 edges
10. `score_sentences()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `render_html_highlighted_doc()` --uses--> `DocumentHierarchyResult`  [INFERRED]
  app.py → hierarchy.py
- `_render_sentences_with_spans()` --uses--> `ParagraphResult`  [INFERRED]
  app.py → hierarchy.py
- `ParagraphResult` --uses--> `DetectionResult`  [INFERRED]
  hierarchy.py → detectors.py
- `analyze_document_hierarchy()` --uses--> `Detector`  [INFERRED]
  hierarchy.py → detectors.py
- `score_sentences()` --uses--> `Detector`  [INFERRED]
  hierarchy.py → detectors.py

## Import Cycles
- None detected.

## Communities (10 total, 5 thin omitted)

### Community 0 - "hierarchy.py"
Cohesion: 0.16
Nodes (12): combine_detections(), aggregate_paragraph(), analyze_document_hierarchy(), DocumentHierarchyResult, is_section_header(), _label_from_score(), ParagraphResult, score_sentences() (+4 more)

### Community 1 - "app.py"
Cohesion: 0.15
Nodes (6): get_perplexity_detector(), render_html_highlighted_doc(), _render_sentences_with_spans(), run_provenance_analysis(), run_stylometry_analysis(), read_document()

### Community 2 - "cli.py"
Cohesion: 0.19
Nodes (4): main(), render_markdown_report(), detect_language(), split_paragraphs()

### Community 3 - "provenance.py"
Cohesion: 0.24
Nodes (6): categorize(), compute_provenance(), normalize_text(), ParagraphProvenance, _sentence_coverage(), split_sentences()

### Community 4 - "Detector"
Cohesion: 0.50
Nodes (3): build_detectors(), AnthropicWatermarkDetector, Detector

## Knowledge Gaps
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DetectionResult` connect `DetectionResult` to `hierarchy.py`, `app.py`, `cli.py`, `PerplexityDetector`, `detectors.py`, `StylometryDetector`, `DummyDetector`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `analyze_document_hierarchy()` connect `hierarchy.py` to `DetectionResult`, `app.py`, `cli.py`, `Detector`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `PerplexityDetector` connect `PerplexityDetector` to `app.py`, `cli.py`, `Detector`, `detectors.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `DetectionResult` (e.g. with `main()` and `aggregate_paragraph()`) actually correct?**
  _`DetectionResult` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Detector` (e.g. with `build_detectors()` and `analyze_document_hierarchy()`) actually correct?**
  _`Detector` has 3 INFERRED edges - model-reasoned connections that need verification._