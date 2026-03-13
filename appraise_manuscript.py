#!/usr/bin/env python3
from __future__ import annotations

import yaml
import argparse
import json
import os
import re
import textwrap
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# --------------------------------------------------
# Ollama helpers
# --------------------------------------------------

def ollama_post(url: str, payload: dict, timeout_s: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_generate(prompt: str, *, url: str, model: str, timeout_s: int = 300) -> str:

    out = ollama_post(
        url,
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout_s=timeout_s,
    )

    if "response" in out:
        return out["response"]
    raise RuntimeError(f"Unexpected Ollama generate response keys: {list(out.keys())}")


# --------------------------------------------------
# Text helpers
# --------------------------------------------------

HEADER_RE = re.compile(r"^(CHAPTER|SCENE|TITLE|SOURCE|SYNOPSIS)\s*:\s*(.*)\s*$", re.IGNORECASE)


def load_book_context(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"Context file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_header_body(text: str) -> Tuple[Dict[str, str], str]:
    lines = text.splitlines()
    header: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            break
        m = HEADER_RE.match(line)
        if m:
            header[m.group(1).upper()] = m.group(2).strip()
        i += 1

    while i < len(lines) and lines[i].strip() == "":
        i += 1

    body = "\n".join(lines[i:]).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return header, body


def compact_excerpt(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# --------------------------------------------------
# Flag selection
# --------------------------------------------------

def load_timeline(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Remove rows that are not real manuscript scenes
    df = df[df["chapter"].notna() & df["scene"].notna()].copy()

    required = [
        "filename", "chapter", "scene", "type", "title",
        "words", "z_energy_global", "z_energy_type", "z_deviation", "path"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"timeline.csv missing required columns: {missing}")

    df = df.copy()
    df["chapter"] = pd.to_numeric(df["chapter"], errors="coerce")
    df["scene"] = pd.to_numeric(df["scene"], errors="coerce")
    df["words"] = pd.to_numeric(df["words"], errors="coerce")
    df["z_energy_global"] = pd.to_numeric(df["z_energy_global"], errors="coerce")
    df["z_energy_type"] = pd.to_numeric(df["z_energy_type"], errors="coerce")
    df["z_deviation"] = pd.to_numeric(df["z_deviation"], errors="coerce")

    df = df.sort_values(["chapter", "scene", "filename"], kind="mergesort").reset_index(drop=True)
    df["idx"] = range(1, len(df) + 1)
    df["delta_z_energy"] = df["z_energy_global"].diff()
    df["abs_delta_z_energy"] = df["delta_z_energy"].abs()

    return df


def chapter_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("chapter", dropna=True, sort=True).agg(
        scenes=("idx", "count"),
        mean_z_energy=("z_energy_global", "mean"),
        std_z_energy=("z_energy_global", "std"),
        mean_abs_delta=("abs_delta_z_energy", "mean"),
    ).reset_index()
    grp["std_z_energy"] = grp["std_z_energy"].fillna(0.0)
    grp["mean_abs_delta"] = grp["mean_abs_delta"].fillna(0.0)
    return grp


def select_flagged_scenes(
    df: pd.DataFrame,
    *,
    top_energy: int = 8,
    top_deviation: int = 8,
    top_transition: int = 8,
    top_lull: int = 6,
    include_ending_scenes: int = 8,
) -> List[Dict]:
    selected: List[Dict] = []
    seen = set()

    def add_row(row: pd.Series, reason: str) -> None:
        key = str(row["filename"])
        if key in seen:
            return
        seen.add(key)

        selected.append({
            "idx": int(row["idx"]),
            "filename": str(row["filename"]),
            "chapter": int(row["chapter"]) if pd.notna(row["chapter"]) else None,
            "scene": int(row["scene"]) if pd.notna(row["scene"]) else None,
            "type": str(row["type"]),
            "title": str(row["title"]) if pd.notna(row["title"]) else "",
            "words": int(row["words"]) if pd.notna(row["words"]) else None,

            "z_energy_global": float(row["z_energy_global"]),
            "z_energy_type": float(row["z_energy_type"]),
            "z_deviation": float(row["z_deviation"]),
            "delta_z_energy": None if pd.isna(row["delta_z_energy"]) else float(row["delta_z_energy"]),
            "abs_delta_z_energy": None if pd.isna(row["abs_delta_z_energy"]) else float(row["abs_delta_z_energy"]),

            "dialogue_line_ratio": float(row["dialogue_line_ratio"]) if pd.notna(row["dialogue_line_ratio"]) else None,

            "path": str(row["path"]),
            "reason": reason,
        })

    # highest energy
    for _, row in df.nlargest(top_energy, "z_energy_global").iterrows():
        add_row(row, "top_energy_spike")

    # highest deviation
    for _, row in df.nlargest(top_deviation, "z_deviation").iterrows():
        add_row(row, "top_deviation")

    # biggest transition shocks (current scene after the cut)
    transitions = df[df["abs_delta_z_energy"].notna()].nlargest(top_transition, "abs_delta_z_energy")
    for _, row in transitions.iterrows():
        add_row(row, "top_transition_shock")

    # deepest lulls
    for _, row in df.nsmallest(top_lull, "z_energy_global").iterrows():
        add_row(row, "deep_lull")

    # opening cluster
    for _, row in df.head(6).iterrows():
        add_row(row, "opening_cluster")

    # ending cluster
    for _, row in df.tail(include_ending_scenes).iterrows():
        add_row(row, "ending_cluster")

    selected.sort(key=lambda x: x["idx"])
    return selected


def attach_excerpts(flags: List[Dict], excerpt_chars: int) -> List[Dict]:
    out = []
    for item in flags:
        path = Path(item["path"])
        excerpt = ""
        if path.exists():
            raw = safe_read_text(path)
            _, body = split_header_body(raw)
            excerpt = compact_excerpt(body, max_chars=excerpt_chars)
        x = dict(item)
        x["excerpt"] = excerpt
        out.append(x)
    return out


# --------------------------------------------------
# Prompt building
# --------------------------------------------------

def build_prompt(
    *,
    book_name: str,
    book_context=None,
    df: pd.DataFrame,
    ch_summary: pd.DataFrame,
    flags: List[Dict],
    include_excerpts: bool,
) -> str:
    total_scenes = len(df)
    prose_scenes = int((df["type"].astype(str).str.upper() == "PROSE").sum())
    types = sorted(df["type"].dropna().astype(str).str.upper().unique().tolist())

    overall = {
        "book_name": book_name,
        "total_scenes": total_scenes,
        "prose_scenes": prose_scenes,
        "scene_types": types,
        "max_z_energy_global": float(df["z_energy_global"].max()),
        "min_z_energy_global": float(df["z_energy_global"].min()),
        "max_z_deviation": float(df["z_deviation"].max()),
        "mean_z_energy_global": float(df["z_energy_global"].mean()),
        "ending_scene_ids": df.tail(8)[["idx", "filename", "chapter", "scene", "title"]].to_dict(orient="records"),
    }

    chapter_rows = ch_summary.to_dict(orient="records")

    book_context_json = json.dumps(book_context or {}, ensure_ascii=False, indent=2)

    # Trim flag payload to essentials
    flag_rows = []
    for f in flags:
        row = {
            "idx": f["idx"],
            "filename": f["filename"],
            "chapter": f["chapter"],
            "scene": f["scene"],
            "type": f["type"],
            "title": f["title"],
            "words": f["words"],
            "z_energy_global": round(f["z_energy_global"], 3),
            "z_energy_type": round(f["z_energy_type"], 3),
            "z_deviation": round(f["z_deviation"], 3),
            "delta_z_energy": None if f["delta_z_energy"] is None else round(f["delta_z_energy"], 3),
            "abs_delta_z_energy": None if f["abs_delta_z_energy"] is None else round(f["abs_delta_z_energy"], 3),
            "reason": f["reason"],
        }
        if include_excerpts:
            row["excerpt"] = f["excerpt"]
        flag_rows.append(row)

    prompt = f"""
    You are a structural appraisal assistant for a novel manuscript.

    Your task is to analyse structural signals extracted from the manuscript and
    produce a careful editorial appraisal based strictly on the supplied evidence.

    You are not a rewrite assistant, not a developmental editor prescribing fixes,
    and not a story-template enforcer.

    Your role is forensic: describe patterns, identify structural behaviour,
    and raise intelligent questions for the author.

    ---------------------------------------------------------------------

    CORE ORIENTATION

    • Evaluate the manuscript against its own internal behaviour.
    • Do NOT compare the manuscript to external story templates.
    • Do NOT invoke Freytag, three-act structure, Save the Cat, Hero’s Journey,
      or any narrative formula.
    • Differences, asymmetries, spikes, lulls, and irregularities may be intentional.
    • Document-like sections may serve structural or thematic purposes.

    ---------------------------------------------------------------------

    CRITICAL RULES

    1. Do NOT rewrite any scene.
    2. Do NOT prescribe repairs.
    3. Do NOT suggest smoothing spikes or balancing energy.
    4. Do NOT suggest increasing or decreasing intensity.
    5. Do NOT recommend adding or removing scenes.
    6. Treat spikes, lulls, asymmetries, and deviations as potentially intentional.
    7. Describe structural behaviour rather than judging it.
    8. If evidence is insufficient, say so explicitly.
    9. Only refer to characters that appear in the canonical book context
       or the flagged scene data.
    10. Do not invent names or merge characters.
    11. Do not infer chapters, acts, or scene ranges that are not present in the data.
    12. Prefer precise citations over general descriptions.

    ---------------------------------------------------------------------

    SCENE CITATION RULE

    Scene references MUST use this exact format:

    [idx=12 | ch=3 sc=4 | file=03_04_example.txt]

    Where:

    idx = scene index in manuscript order  
    ch = chapter number  
    sc = scene number  
    file = scene file name

    Do NOT invent identifiers such as:

    scene_002  
    scene_013  

    If chapter or scene data is missing, omit the citation rather than inventing one.

    ---------------------------------------------------------------------

    EVIDENCE DISCIPLINE

    • Interpret signals cautiously.
    • Avoid attributing narrative intention unless strongly supported.
    • Prefer neutral structural language such as:

      escalation  
      plateau  
      compression  
      variance shift  
      drop in intensity  
      sustained low signal  

    • Do NOT assert intention unless clearly evidenced.

    Use cautious language when appropriate:

    • the signal suggests  
    • this may indicate  
    • this appears to  
    • this could function as  

    ---------------------------------------------------------------------

    ENDING INTERPRETATION RULE

    Low energy near the ending is not inherently negative.

    A late drop in signal may represent:

    • decompression  
    • aftermath  
    • silence  
    • emotional exhaustion  
    • anti-climax  
    • deliberate quieting  

    Describe what the signal does rather than judging whether it is correct.

    ---------------------------------------------------------------------

    DOCUMENT-LIKE MATERIAL RULE

    Do not assume document-like material is a weakness.

    Document-style sections may serve purposes such as:

    • bureaucratic rhythm
    • exposition compression
    • tonal estrangement
    • structural contrast
    • narrative framing

    Describe how these sections behave structurally without assuming defect.

    ---------------------------------------------------------------------

    FORBIDDEN PHRASES

    Do NOT use phrases such as:

    • need for better pacing  
    • need for payoff  
    • need for resolution  
    • disrupts the flow  
    • maintain reader engagement  
    • careful editing may be needed  
    • should be fixed  
    • should be improved  
    • lacks payoff  
    • underwritten  
    • under-supported  
    - hooks the reader
    - maintain reader engagement
    - emotional impact
    - effective engagement
    - narrative impact
    - enhances impact

    Replace them with neutral observations.

    Do not describe a section as conclusive, reflective, cathartic, climactic, or resolved unless the claim is explicitly hedged and scene-cited.

    Prefer:
    - may read as
    - may function as
    - could suggest

    ---------------------------------------------------------------------

    CANONICAL BOOK CONTEXT

    Use the following canonical book context as ground truth:

    {book_context_json}

    ---------------------------------------------------------------------

    MANUSCRIPT DATA

    You are analysing this manuscript:

    {json.dumps(overall, ensure_ascii=False, indent=2)}

    Per-chapter summary:

    {json.dumps(chapter_rows, ensure_ascii=False, indent=2)}

    Field meanings:

    idx = scene index in manuscript order  
    chapter = chapter number  
    scene = scene number  
    z_energy_global = relative narrative energy compared with the manuscript baseline  
    z_deviation = local structural shift compared with neighbouring scenes  
    dialogue_line_ratio = proportion of dialogue lines in the scene  
    reason = why the scene was flagged for review  

    Flagged scenes selected for review:

    {json.dumps(flag_rows, ensure_ascii=False, indent=2)}

    ---------------------------------------------------------------------

    PERFORM THE TASK IN TWO STAGES

    ---------------------------------------------------------------------

    STAGE 1 — SIGNAL READING

    Write a short diagnostic reading of the evidence only.

    Describe:

    • overall structural shape  
    • where energy rises, falls, compresses, or sustains  
    • where deviations cluster  
    • what the ending signal does relative to earlier sections  
    • how prose and document-like material behave structurally  
    • where the manuscript departs from its own baseline  

    Rules for Stage 1:

    • Do NOT praise the manuscript.
    • Do NOT advise changes.
    • Do NOT speculate beyond the evidence.
    • Do NOT call anything a flaw.
    • If multiple interpretations exist, briefly mention them.

    ---------------------------------------------------------------------

    STAGE 2 — EDITORIAL APPRAISAL

    Using Stage 1 only, write a markdown report with exactly these sections.

    # Appraisal: {book_name}

    ## Executive Summary

    Write 5-10 sentences summarising the structural behaviour of the manuscript
    based strictly on the signal evidence.

    ## Structural Shape

    Describe the pacing architecture or waveform suggested by the signals.

    Avoid comparison with external narrative templates.

    ## Phase Behaviour

    Discuss only:

    • early manuscript  
    • middle manuscript  
    • late manuscript  

    Use actual chapter numbers and scene citations where possible.

    Do not infer acts unless explicitly stated in the book context.

    ## Ending Appraisal

    Describe how the ending behaves structurally relative to earlier sections.

    Consider multiple interpretations if evidence allows.

    ## Signal-Supported Strengths

    Provide 3-6 bullet points describing structural behaviours that appear effective.

    Each bullet must include at least one scene citation.

    ## Signal-Supported Attention Zones

    Provide 3-6 bullet points identifying zones that differ from the manuscript’s
    own baseline.

    Explain why they merit human attention.

    Do NOT propose fixes.

    ## Questions for Human Review

    Provide 5-10 precise editorial questions.

    Questions should investigate:

    • authorial intent  
    • structural clarity  
    • narrative preparation  
    • reader interpretation  

    Do NOT propose solutions.

    ## Evidence Appendix

    List the most significant flagged scenes and explain why they matter
    structurally.

    All scenes must use the canonical citation format.

    ---------------------------------------------------------------------

    STYLE RULES

    • Be precise, restrained, and evidence-led.
    • Prefer structural language over narrative judgement.
    • Stay close to the supplied data.
    • Avoid generic workshop advice.
    • Do not moralise the manuscript's structure.
    """
    return textwrap.dedent(prompt).strip()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a structural editorial appraisal from timeline metrics.")
    ap.add_argument("--context", help="Path to BOOK_CONTEXT.yaml")
    ap.add_argument("book_name", help="Book identifier, e.g. AHOI or KTIATL")
    ap.add_argument("--reports-dir", default="./reports", help="Root reports directory")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_GEN_MODEL", "qwen2.5:7b"),
                    help="Ollama generation model")
    ap.add_argument("--url", default=os.environ.get("OLLAMA_GEN_URL", "http://localhost:11434/api/generate"),
                    help="Ollama generate endpoint")
    ap.add_argument("--excerpt-chars", type=int, default=1200, help="Excerpt cap per flagged scene")
    ap.add_argument("--no-excerpts", action="store_true", help="Do metrics-only appraisal")
    args = ap.parse_args()

    book_context = load_book_context(args.context)
    reports_root = Path(args.reports_dir).expanduser().resolve()
    book_dir = reports_root / args.book_name
    timeline_csv = book_dir / "timeline.csv"
    out_md = book_dir / "APPRAISAL.md"
    out_json = book_dir / "APPRAISAL_FLAGS.json"

    if not timeline_csv.exists():
        raise SystemExit(f"timeline.csv not found: {timeline_csv}")

    df = load_timeline(timeline_csv)
    ch_summary = chapter_summary(df)
    flags = select_flagged_scenes(df)
    flags = attach_excerpts(flags, excerpt_chars=args.excerpt_chars) if not args.no_excerpts else flags

    prompt = build_prompt(
        book_name=args.book_name,
        df=df,
        ch_summary=ch_summary,
        flags=flags,
        include_excerpts=not args.no_excerpts,
        book_context=book_context,
    )

    appraisal = ollama_generate(prompt, url=args.url, model=args.model)

    out_json.write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(appraisal.strip() + "\n", encoding="utf-8")

    skeptic_prompt = f"""
    You are reviewing an editorial appraisal of a novel manuscript.

    Your job is to audit the appraisal's reasoning, not the manuscript itself.

    Be adversarial but fair.

    Tasks:
    1. Identify claims that are well supported.
    2. Identify claims that may exceed the evidence.
    3. Identify places where alternative interpretations are plausible.
    4. Give a short overall reliability assessment.

    Rules:
    - Do not rewrite the appraisal.
    - Do not suggest edits to the manuscript.
    - Do not invent manuscript facts.
    - Stay close to the appraisal text and its cited evidence.
    - Be concise and precise.
    - Pay special attention to claims about reader experience, emotional effect, closure, payoff, and authorial intention.
      These usually require stronger evidence than structural observations.

    Appraisal to review:

    {appraisal.strip()}

    Write markdown with exactly these sections:

    # Skeptical Review

    ## Claims Well Supported

    ## Claims That May Exceed Evidence

    ## Alternative Interpretations

    ## Overall Reliability Assessment
    """

    skeptic_review = ollama_generate(skeptic_prompt, url=args.url, model=args.model)
    (book_dir / "SKEPTIC_REVIEW.md").write_text(skeptic_review.strip() + "\n", encoding="utf-8")

    print(f"Book:      {args.book_name}")
    print(f"Timeline:  {timeline_csv}")
    print(f"Flags:     {out_json}")
    print(f"Appraisal: {out_md}")
    print(f"Skeptic:  {book_dir / 'SKEPTIC_REVIEW.md'}")

if __name__ == "__main__":
    main()
