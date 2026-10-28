# Graph Report - Check_AI_tools  (2026-10-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 33 nodes · 57 edges · 6 communities (1 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- provenance.py
- DummyDetector
- detectors.py
- build_detectors
- PerplexityDetector

## God Nodes (most connected - your core abstractions)
1. `Detector` - 8 edges
2. `PerplexityDetector` - 7 edges
3. `DummyDetector` - 6 edges
4. `AnthropicWatermarkDetector` - 6 edges
5. `compute_provenance()` - 6 edges
6. `build_detectors()` - 6 edges
7. `DetectionResult` - 5 edges
8. `main()` - 4 edges
9. `split_paragraphs()` - 4 edges
10. `ParagraphProvenance` - 2 edges

## Surprising Connections (you probably didn't know these)
- `build_detectors()` --uses--> `Detector`  [INFERRED]
  cli.py → detectors.py
- `build_detectors()` --calls--> `DummyDetector`  [EXTRACTED]
  cli.py → detectors.py
- `build_detectors()` --calls--> `PerplexityDetector`  [EXTRACTED]
  cli.py → detectors.py
- `build_detectors()` --calls--> `AnthropicWatermarkDetector`  [EXTRACTED]
  cli.py → detectors.py
- `main()` --calls--> `compute_provenance()`  [EXTRACTED]
  cli.py → provenance.py

## Import Cycles
- None detected.

## Communities (6 total, 5 thin omitted)

### Community 0 - "provenance.py"
Cohesion: 0.33
Nodes (5): main(), categorize(), compute_provenance(), ParagraphProvenance, split_paragraphs()

## Knowledge Gaps
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PerplexityDetector` connect `PerplexityDetector` to `DummyDetector`, `cli.py`, `detectors.py`, `build_detectors`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Why does `DummyDetector` connect `DummyDetector` to `cli.py`, `detectors.py`, `build_detectors`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `AnthropicWatermarkDetector` connect `build_detectors` to `DummyDetector`, `cli.py`, `detectors.py`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._