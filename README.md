# Manuscript Signal Analysis

A structural analysis pipeline for long-form fiction manuscripts.

This tool extracts **structural telemetry** from a manuscript and produces a multi-layer editorial analysis including:

* signal metrics
* pacing graphs
* structural appraisal
* skeptical review
* editorial conclusion
* literary critical reading

The goal is **not rewriting or scoring a manuscript**, but providing **diagnostic insight into its structure, rhythm, and thematic behaviour**.

The system uses **local LLMs via Ollama** combined with deterministic signal analysis.

---

# Pipeline Overview

The pipeline processes a manuscript through several stages:

```
manuscript
↓
signal extraction
↓
timeline graphs
↓
structural appraisal
↓
skeptical review
↓
editorial conclusion
↓
critical reading
```

Each stage produces artifacts inside:

```
reports/<BOOK_NAME>/
```

---

# Directory Structure

Typical layout:

```
RAG/
├── manuscript/
│   └── AHOI/
│       ├── 01_01_scene.txt
│       ├── 01_02_scene.txt
│       └── BOOK_CONTEXT.yaml
│
├── project/
│   ├── analyze_manuscript.sh
│   ├── audit_signals.py
│   ├── timeline_plots.py
│   ├── mod_timeline_plots.py
│   ├── mod_timeline_plots_prose_only.py
│   ├── appraise_manuscript.py
│   ├── skeptic_review.py
│   ├── conclusion.py
│   ├── critical_reading.py
│   └── reports/
│       └── <BOOK_NAME>/
```

Generated outputs are stored under:

```
project/reports/<BOOK_NAME>/
```

---

# Requirements

Python 3.10+

Python packages used:

```
pandas
numpy
matplotlib
pyyaml
```

Install quickly with:

```
pip install pandas numpy matplotlib pyyaml
```

---

# LLM Requirements

The pipeline expects **Ollama** running locally.

Install:

```
https://ollama.ai
```

Pull a model (recommended):

```
ollama pull qwen2.5:14b
```

Other models may work.

---

# Quick Start

From the `project/` directory:

```
./analyze_manuscript.sh AHOI qwen2.5:14b
```

Arguments:

```
./analyze_manuscript.sh <BOOK_NAME> <OLLAMA_MODEL>
```

Example:

```
./analyze_manuscript.sh KTIATL qwen2.5:14b
```

The script will:

1. Extract scene metrics
2. Generate pacing graphs
3. Perform structural appraisal
4. Run a skeptical review
5. Produce an editorial conclusion
6. Produce a literary critical reading

---

# Output

Results appear in:

```
reports/<BOOK_NAME>/
```

Key outputs:

```
timeline.csv
timeline.md
trends.csv

APPRAISAL.md
SKEPTIC_REVIEW.md
CONCLUSION.md
CRITICAL_READING.md
```

Graphs:

```
out_timeline/
out_mod_timeline/
out_mod_timeline_prose_only/
```

View reports easily with:

```
glow reports/<BOOK_NAME>/APPRAISAL.md
glow reports/<BOOK_NAME>/CRITICAL_READING.md
```

---

# Manuscript Format

Scenes should be plain text files.

Example naming:

```
01_01_opening.txt
01_02_dialogue.txt
02_01_scene.txt
```

The pipeline extracts structural metrics from each scene.

---

# Optional: BOOK_CONTEXT.yaml

A `BOOK_CONTEXT.yaml` file can be placed inside the manuscript directory.

This provides canonical information to prevent the model from inventing characters or plot elements.

Example:

```
manuscript/AHOI/BOOK_CONTEXT.yaml
```

This file is optional but recommended.

---

# Philosophy

Most AI writing tools attempt to:

* rewrite text
* critique prose
* offer stylistic advice

This project instead performs **structural telemetry**.

It measures:

* pacing
* energy
* deviation
* dialogue density
* structural shifts

Then combines those signals with LLM interpretation to produce a layered editorial analysis.

The goal is to answer questions such as:

* What kind of narrative object is this manuscript?
* Where does its structural pressure lie?
* What patterns emerge across the book?

---

# Notes

The tool is designed for **completed manuscripts** rather than early drafts.

Generated reports are reproducible and therefore excluded from version control.
