#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./analyze_manuscript.sh BOOK_NAME [MODEL]
#
# Examples:
#   ./analyze_manuscript.sh AHOI
#   ./analyze_manuscript.sh AHOI qwen2.5:7b
#   ./analyze_manuscript.sh KTIATL qwen2.5:14b

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 BOOK_NAME [MODEL]"
  exit 1
fi

BOOK_NAME="$1"
MODEL="${2:-qwen2.5:14b}"

PROJECT_DIR="${HOME}/RAG/project"
MANUSCRIPT_DIR="${HOME}/RAG/manuscript/${BOOK_NAME}"
REPORT_DIR="${PROJECT_DIR}/reports/${BOOK_NAME}"

TIMELINE_CSV="${REPORT_DIR}/timeline.csv"
TIMELINE_OUT="${REPORT_DIR}/out_timeline"
MOD_OUT="${REPORT_DIR}/out_mod_timeline"
PROSE_OUT="${REPORT_DIR}/out_mod_timeline_prose_only"

APPRAISAL_MD="${REPORT_DIR}/APPRAISAL.md"
CONTEXT_FILE="${MANUSCRIPT_DIR}/BOOK_CONTEXT.yaml"

if [[ ! -d "${MANUSCRIPT_DIR}" ]]; then
  echo "Error: manuscript directory not found:"
  echo "  ${MANUSCRIPT_DIR}"
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found in ${PROJECT_DIR}"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo
echo "=== Manuscript Analysis ==="
echo "Book:        ${BOOK_NAME}"
echo "Model:       ${MODEL}"
echo "Project dir: ${PROJECT_DIR}"
echo "Manuscript:  ${MANUSCRIPT_DIR}"
echo "Reports:     ${REPORT_DIR}"
echo

mkdir -p "${REPORT_DIR}"

echo "-> Running audit_signals.py"
python audit_signals.py \
  --corpus-dir "${MANUSCRIPT_DIR}" \
  --out-dir "${REPORT_DIR}"

echo
echo "-> Running timeline_plots.py"
python timeline_plots.py \
  "${TIMELINE_CSV}" \
  --outdir "${TIMELINE_OUT}"

echo
echo "-> Running mod_timeline_plots.py"
python mod_timeline_plots.py \
  "${TIMELINE_CSV}" \
  --outdir "${MOD_OUT}"

echo
echo "-> Running mod_timeline_plots_prose_only.py"
python mod_timeline_plots_prose_only.py \
  "${TIMELINE_CSV}" \
  --outdir "${PROSE_OUT}"

echo
echo "-> Running appraise_manuscript.py"
if [[ -f "${CONTEXT_FILE}" ]]; then
  echo "   Using context: ${CONTEXT_FILE}"
  python appraise_manuscript.py "${BOOK_NAME}" \
    --model "${MODEL}" \
    --context "${CONTEXT_FILE}"
else
  echo "   No BOOK_CONTEXT.yaml found, running without context"
  python appraise_manuscript.py "${BOOK_NAME}" \
    --model "${MODEL}"
fi

echo "-> Generating conclusion"

python conclusion.py \
  --book "$BOOK_NAME" \
  --model "$MODEL"

echo "-> Generating critical reading"

python critical_reading.py \
  --book "$BOOK_NAME" \
  --model "$MODEL"

echo
echo "=== Done ==="
echo "Generated:"
echo "  ${REPORT_DIR}/timeline.csv"
echo "  ${REPORT_DIR}/timeline.md"
echo "  ${TIMELINE_OUT}"
echo "  ${MOD_OUT}"
echo "  ${PROSE_OUT}"
echo "  ${APPRAISAL_MD}"
echo

echo "Open graphs with:"
echo "  open ${TIMELINE_OUT}/*.png"
echo "  open ${MOD_OUT}/*.png"
echo "  open ${PROSE_OUT}/*.png"
echo

if command -v glow >/dev/null 2>&1; then
  echo "-> Opening appraisal in glow"
  glow "${REPORT_DIR}/APPRAISAL.md"

  echo "-> Opening skeptical review in glow"
  glow "${REPORT_DIR}/SKEPTIC_REVIEW.md"

  echo "-> Opening conclusion in glow"
  glow "${REPORT_DIR}/CONCLUSION.md"

  echo "-> Opening critical reading in glow"
  glow "${REPORT_DIR}/CRITICAL_READING.md"
else
  echo "glow not found. Install with:"
  echo "  brew install glow"
  echo
  echo "Then view manually with:"
  echo "  glow ${APPRAISAL_MD}"
fi
