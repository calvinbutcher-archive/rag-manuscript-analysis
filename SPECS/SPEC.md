# RAG Editorial Diagnosis — Minimal Spec (v0.1)

## Objective
Given a folder of per-scene `.txt` files (already split from Scrivener), generate editorial diagnosis reports that help the author spot:
- spikes (intensity / novelty / density)
- lulls (flatness / low movement / low novelty)
- deviations (style / POV / formatting / modality changes)

This system must preserve the author’s intent: spikes/lulls/deviations may be *good*.

## Hard Constraints
1) NEVER modify files inside the manuscript corpus directory.
2) NEVER rewrite prose. Diagnosis only.
3) All outputs MUST be written to `./reports/` (or user-specified output dir).
4) Every claim in a report MUST cite:
   - filename (e.g. `15_12_dream.txt`)
   - chunk index (if chunked)
   - a short excerpt (<= 300 chars)

## Inputs
- `--corpus-dir`: directory of scene files, e.g. `~/RAG/manuscript/KTIATL`
- Optional `--index-dir`: existing embeddings index directory
- Optional `--model`: Ollama model for summary/explanation (generation), NOT for rewriting

## Artifacts
### A) Scene Metrics (`reports/scene_metrics.jsonl`)
One JSON object per scene with:
- file, chapter, scene, title
- char_count, word_count, paragraph_count, dialogue_ratio
- lexical_diversity (simple type/token)
- punctuation_rate (per 1k chars)
- “mode flags”: looks_like_transcript / memo / list / epigraph

### B) Timeline Plots (text-first) (`reports/timeline.md`)
A markdown report with:
- top 10 spikes + why (citations)
- top 10 lulls + why (citations)
- top 10 deviations + why (citations)
- a quick “act/chapter” aggregation if chapter exists

### C) Diagnostic Q&A (`reports/qa_<query>.md`)
Given a user question, retrieve relevant scenes (embedding search) and answer with citations only.

## Detection Rules (v0.1)
### Spikes
A “spike” is a scene that is an outlier in one or more:
- sentence length variance
- punctuation density
- lexical diversity delta vs rolling average
- rare token burst (simple TF spike)
- dialogue_ratio delta
- sentiment/valence NOT required (optional later)

### Lulls
A “lull” is a scene that is an outlier low in:
- lexical novelty vs rolling window
- event markers proxy (verbs per 1k words, crude)
- paragraph transitions (very long blocks)

### Deviations
A “deviation” is a strong change in:
- formatting mode (transcript/memo/list suddenly appears)
- POV markers (I/we/he/she frequency swing)
- register markers (formal/technical vocabulary burst)

## CLI (v0.1)
We will implement one command first:

`python audit_signals.py --corpus-dir ... --out-dir ./reports`

It produces:
- `scene_metrics.jsonl`
- `timeline.md`

No model required for v0.1.

## Acceptance Criteria
- Running `audit_signals.py` on KTIATL completes in < 30 seconds on the current machine.
- Reports include useful ranked lists with citations.
- Manuscript directory unchanged (verify via `find ... -mtime` not required but recommended).
