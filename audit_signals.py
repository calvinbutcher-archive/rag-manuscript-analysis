#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------
# Prep for REDACTIONS, preventing it creating cartoon z score outlier
# --------------------------------------------------------------------

RE_REDACTION_BLOCK = re.compile(r"[█■▮▯]{6,}")  # common redaction glyphs
RE_RULE_LINE = re.compile(r"^[\-\–\—_]{6,}\s*$", re.M)  # long separator lines
RE_MULTISPACE = re.compile(r"[ \t]{2,}")

# ----------------------------
# Parsing / classification
# ----------------------------

HEADER_RE = re.compile(r"^(TITLE|SYNOPSIS)\s*:\s*(.*)\s*$", re.IGNORECASE)
FILENAME_RE = re.compile(r"^(?P<ch>\d+)[_\-](?P<sc>\d+)_", re.IGNORECASE)

WORD_RE = re.compile(r"[A-Za-z0-9']+")
DIALOGUE_LINE_RE = re.compile(r'^\s*["“].*')  # line starts with a quote
DOCISH_RE = re.compile(r"^\s*(TO|FROM|SUBJECT|DATE|REF|ATTN)\s*:", re.IGNORECASE)

TYPE_KEYWORDS = [
    ("MEMO", "MEMO"),
    ("NON-MEMO", "MEMO"),
    ("TRANSCRIPT", "TRANSCRIPT"),
    ("INTERVIEW", "INTERVIEW"),
    ("REPORT", "REPORT"),
    ("BRIEF", "REPORT"),
    ("SESSION", "TRANSCRIPT"),
    ("LOG", "LOG"),
    ("MINUTES", "TRANSCRIPT"),
    ("Q&A", "TRANSCRIPT"),
]


def normalise_for_metrics(text: str) -> str:
    """
    Remove/neutralise formatting artifacts that blow up density metrics
    (e.g., redaction bars), without changing the *content*.
    """
    # Collapse redaction bars into a single stable token
    text = RE_REDACTION_BLOCK.sub(" [REDACTED] ", text)

    # Collapse long rule lines (-----) into a token
    text = RE_RULE_LINE.sub(" [RULE] ", text)

    # Clean excessive spacing
    text = RE_MULTISPACE.sub(" ", text)

    return text


def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def parse_filename_ids(filename: str) -> Tuple[Optional[int], Optional[int]]:
    m = FILENAME_RE.match(filename)
    if not m:
        return None, None
    return int(m.group("ch")), int(m.group("sc"))


def split_header_body(text: str) -> Tuple[Dict[str, str], str]:
    """
    Scene files:

    TITLE: ...
    SYNOPSIS: ...
    <blank>
    body

    We'll treat leading TITLE/SYNOPSIS lines as header until first blank line.
    """
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


def classify_scene(title: str, body: str) -> str:
    t = (title or "").upper()

    for needle, label in TYPE_KEYWORDS:
        if needle in t:
            return label

    first = "\n".join(body.splitlines()[:20])
    docish_hits = len(DOCISH_RE.findall(first))
    if docish_hits >= 2:
        return "DOC"

    bullet_hits = sum(1 for ln in body.splitlines()[:30] if ln.strip().startswith(("-", "•", "*")))
    colon_hits = sum(1 for ln in body.splitlines()[:30] if ":" in ln[:30])
    if bullet_hits >= 4 and colon_hits >= 3:
        return "DOC"

    return "PROSE"


# ----------------------------
# Metrics
# ----------------------------

@dataclass
class Scene:
    path: Path
    filename: str
    chapter: Optional[int]
    scene: Optional[int]
    title: str
    synopsis: str
    type: str
    body: str


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def punctuation_count(text: str) -> int:
    # count "intensity" punctuation (not apostrophes)
    return sum(1 for ch in text if ch in "!?;:—-…,")


def dialogue_line_ratio(lines: List[str]) -> float:
    if not lines:
        return 0.0
    d = sum(1 for ln in lines if DIALOGUE_LINE_RE.match(ln))
    return d / max(1, len(lines))


def unique_token_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / max(1, len(tokens))


def bigram_novelty_ratio(tokens: List[str]) -> float:
    if len(tokens) < 3:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / max(1, len(bigrams))


def safe_zscores(values: List[float]) -> List[float]:
    if not values:
        return []
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, (len(values) - 1))
    sd = math.sqrt(var)
    if sd < 1e-12:
        return [0.0 for _ in values]
    return [(v - mu) / sd for v in values]


def rolling_mean(xs: List[float], window: int) -> List[float]:
    if window <= 1:
        return xs[:]
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in xs:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


# ----------------------------
# Upgrade B helpers
# ----------------------------

def percentile(values: List[float], p: float) -> float:
    """
    Simple deterministic percentile. p in [0,1].
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def winsorize(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def short_scene_scale(words: int, floor_words: int = 250, exponent: float = 0.35) -> float:
    """
    Gently down-weights energy scores for very short scenes which otherwise
    inflate uniq ratio and punct/1k.

    scale = min(1, words/floor) ** exponent
    """
    if words <= 0:
        return 0.0
    x = min(1.0, words / float(floor_words))
    return x ** exponent


# ----------------------------
# Analysis
# ----------------------------

def load_scenes(corpus_dir: Path) -> List[Scene]:

    files = sorted(
        [
            p for p in corpus_dir.rglob("*.txt")
            if p.is_file()
            and p.name != "manifest.txt"
            and FILENAME_RE.match(p.name)
        ],
        key=lambda p: p.name
    )

    scenes: List[Scene] = []
    for p in files:
        raw = safe_read_text(p)
        header, body = split_header_body(raw)
        title = header.get("TITLE", "").strip()
        synopsis = header.get("SYNOPSIS", "").strip()
        ch, sc = parse_filename_ids(p.name)
        stype = classify_scene(title, body)
        scenes.append(Scene(
            path=p,
            filename=p.name,
            chapter=ch,
            scene=sc,
            title=title,
            synopsis=synopsis,
            type=stype,
            body=body,
        ))
    return scenes


def compute_base_rows(scenes: List[Scene]) -> List[Dict]:
    """
    Compute raw metrics WITHOUT energy_density, so we can apply winsorization
    (Upgrade B2) deterministically across the corpus before energy is computed.
    """
    rows: List[Dict] = []
    for s in scenes:
        body = normalise_for_metrics(s.body)
        lines = body.splitlines()
        tokens = tokenize(body)

        words = count_words(body)
        n_lines = len(lines)
        avg_line_len = (sum(len(ln) for ln in lines) / max(1, n_lines)) if n_lines else 0.0

        punct = punctuation_count(body)
        punct_per_1k = (punct / max(1, words)) * 1000.0

        dlg_ratio = dialogue_line_ratio(lines)
        uniq_ratio = unique_token_ratio(tokens)
        bigram_nov = bigram_novelty_ratio(tokens)

        rows.append({
            "path": str(s.path),
            "filename": s.filename,
            "chapter": s.chapter,
            "scene": s.scene,
            "title": s.title,
            "synopsis": s.synopsis,
            "type": s.type,
            "words": words,
            "lines": n_lines,
            "avg_line_len": avg_line_len,
            "punct": punct,
            "punct_per_1k_raw": punct_per_1k,  # keep raw for debugging
            "dialogue_line_ratio": dlg_ratio,
            "unique_token_ratio": uniq_ratio,
            "bigram_novelty_ratio": bigram_nov,
        })
    return rows


def compute_energy_density(
    rows: List[Dict],
    punct_winsor_lo: float,
    punct_winsor_hi: float,
    short_floor_words: int = 250,
    short_exponent: float = 0.35,
) -> None:
    """
    Upgrade B:
      B2) Winsorize punct/1k before using it in energy_density
      B1) Apply a gentle short-scene scaling to energy_density
    """
    for r in rows:
        words = int(r["words"])
        p1k_raw = float(r["punct_per_1k_raw"])
        p1k = winsorize(p1k_raw, punct_winsor_lo, punct_winsor_hi)

        dlg_ratio = float(r["dialogue_line_ratio"])
        uniq_ratio = float(r["unique_token_ratio"])
        bigram_nov = float(r["bigram_novelty_ratio"])

        # Base energy proxy
        energy_density = (
            0.50 * p1k +
            25.0 * dlg_ratio +
            40.0 * uniq_ratio +
            10.0 * bigram_nov
        )

        # Short scene damping (word-floor scaling)
        scale = short_scene_scale(words, floor_words=short_floor_words, exponent=short_exponent)
        energy_density *= scale

        # Persist
        r["punct_per_1k"] = p1k
        r["energy_density"] = energy_density
        r["short_scene_scale"] = scale


def compute_metrics(
    scenes: List[Scene],
    punct_winsor_p_lo: float = 0.01,
    punct_winsor_p_hi: float = 0.95,
    short_floor_words: int = 250,
    short_exponent: float = 0.35,
) -> Tuple[List[Dict], Dict]:
    """
    Returns (rows, config_used).
    """
    rows = compute_base_rows(scenes)

    p1k_all = [float(r["punct_per_1k_raw"]) for r in rows]
    lo = percentile(p1k_all, punct_winsor_p_lo)
    hi = percentile(p1k_all, punct_winsor_p_hi)

    compute_energy_density(
        rows,
        punct_winsor_lo=lo,
        punct_winsor_hi=hi,
        short_floor_words=short_floor_words,
        short_exponent=short_exponent,
    )

    cfg = {
        "punct_winsor_p_lo": punct_winsor_p_lo,
        "punct_winsor_p_hi": punct_winsor_p_hi,
        "punct_winsor_lo_value": lo,
        "punct_winsor_hi_value": hi,
        "short_floor_words": short_floor_words,
        "short_exponent": short_exponent,
    }
    return rows, cfg


def add_zscores(rows: List[Dict]) -> None:
    # Global z-scores
    energy = [float(r["energy_density"]) for r in rows]
    words = [float(r["words"]) for r in rows]
    avgll = [float(r["avg_line_len"]) for r in rows]
    p1k = [float(r["punct_per_1k"]) for r in rows]
    dlg = [float(r["dialogue_line_ratio"]) for r in rows]
    uniq = [float(r["unique_token_ratio"]) for r in rows]

    z_energy = safe_zscores(energy)
    z_words = safe_zscores(words)
    z_avgll = safe_zscores(avgll)
    z_p1k = safe_zscores(p1k)
    z_dlg = safe_zscores(dlg)
    z_uniq = safe_zscores(uniq)

    for i, r in enumerate(rows):
        r["z_energy_global"] = z_energy[i]
        r["z_words_global"] = z_words[i]
        r["z_avg_line_len_global"] = z_avgll[i]
        r["z_punct_per_1k_global"] = z_p1k[i]
        r["z_dialogue_ratio_global"] = z_dlg[i]
        r["z_unique_ratio_global"] = z_uniq[i]

    # Within-type z-scores
    by_type: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by_type.setdefault(r["type"], []).append(i)

    for t, idxs in by_type.items():
        e = [float(rows[i]["energy_density"]) for i in idxs]
        zw = safe_zscores(e)
        for j, i in enumerate(idxs):
            rows[i]["z_energy_type"] = zw[j]


def add_deviation(rows: List[Dict]) -> None:
    """
    Deviation = abs change in energy density relative to previous scene (in global z space).
    """
    devs: List[float] = []
    prev = None
    for r in rows:
        cur = float(r.get("z_energy_global", 0.0))
        if prev is None:
            dev = 0.0
        else:
            dev = abs(cur - prev)
        devs.append(dev)
        prev = cur

    z_devs = safe_zscores(devs)
    for i, r in enumerate(rows):
        r["deviation"] = devs[i]
        r["z_deviation"] = z_devs[i]


def detect_runs(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Runs / clusters: consecutive same type, consecutive low-energy, consecutive high-deviation.
    Thresholds are deliberately mild and non-forcing.
    """
    low_flags = [float(r["z_energy_global"]) < -0.9 for r in rows]
    hi_dev_flags = [float(r["z_deviation"]) > 1.2 for r in rows]

    def runs_for(predicate: List[bool], label: str) -> List[Dict]:
        out: List[Dict] = []
        start = None
        for i, ok in enumerate(predicate + [False]):  # sentinel
            if ok and start is None:
                start = i
            elif (not ok) and start is not None:
                end = i - 1
                length = end - start + 1
                if length >= 3:
                    out.append({"label": label, "start": start, "end": end, "length": length})
                start = None
        return out

    type_runs: List[Dict] = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["type"] != rows[start]["type"]:
            length = i - start
            if length >= 3:
                type_runs.append({
                    "label": f"type:{rows[start]['type']}",
                    "start": start,
                    "end": i - 1,
                    "length": length
                })
            start = i

    low_runs = runs_for(low_flags, "low_energy")
    dev_runs = runs_for(hi_dev_flags, "high_deviation")

    type_runs.sort(key=lambda d: d["length"], reverse=True)
    low_runs.sort(key=lambda d: d["length"], reverse=True)
    dev_runs.sort(key=lambda d: d["length"], reverse=True)

    return {"type_runs": type_runs, "low_runs": low_runs, "dev_runs": dev_runs}


def top_n(
    rows: List[Dict],
    key: str,
    n: int,
    reverse: bool = True,
    where: Optional[Callable[[Dict], bool]] = None
) -> List[Dict]:
    xs = rows
    if where is not None:
        xs = [r for r in rows if where(r)]
    return sorted(xs, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[:n]


# ----------------------------
# Upgrade A: Transition hotspots
# ----------------------------

def component_deltas(prev: Dict, cur: Dict) -> Dict[str, float]:
    return {
        "punct_per_1k": float(cur["punct_per_1k"]) - float(prev["punct_per_1k"]),
        "dialogue_line_ratio": float(cur["dialogue_line_ratio"]) - float(prev["dialogue_line_ratio"]),
        "unique_token_ratio": float(cur["unique_token_ratio"]) - float(prev["unique_token_ratio"]),
        "bigram_novelty_ratio": float(cur["bigram_novelty_ratio"]) - float(prev["bigram_novelty_ratio"]),
        "words": float(cur["words"]) - float(prev["words"]),
    }


def top_changed_components(prev: Dict, cur: Dict, k: int = 2) -> List[Tuple[str, float]]:
    ds = component_deltas(prev, cur)
    ranked = sorted(ds.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return ranked[:k]


def write_outputs(rows: List[Dict], out_dir: Path, top_n_count: int, cfg: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "scene_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_dir / "timeline.csv"
    cols = [
        "filename", "chapter", "scene", "type", "title",
        "words", "lines", "avg_line_len",
        "punct", "punct_per_1k_raw", "punct_per_1k",
        "dialogue_line_ratio", "unique_token_ratio", "bigram_novelty_ratio",
        "short_scene_scale", "energy_density",
        "z_energy_global", "z_energy_type",
        "deviation", "z_deviation",
        "path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    energy_g = [float(r["z_energy_global"]) for r in rows]
    dev = [float(r["z_deviation"]) for r in rows]
    words = [float(r["words"]) for r in rows]

    e_roll = rolling_mean(energy_g, window=5)
    d_roll = rolling_mean(dev, window=5)
    w_roll = rolling_mean(words, window=5)

    trends_csv = out_dir / "trends.csv"
    with trends_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "filename", "chapter", "scene", "type", "z_energy_global", "z_deviation", "words", "roll5_energy", "roll5_dev", "roll5_words"])
        for i, r in enumerate(rows):
            w.writerow([
                i,
                r["filename"],
                r.get("chapter", ""),
                r.get("scene", ""),
                r.get("type", ""),
                f"{energy_g[i]:.4f}",
                f"{dev[i]:.4f}",
                int(words[i]),
                f"{e_roll[i]:.4f}",
                f"{d_roll[i]:.4f}",
                f"{w_roll[i]:.1f}",
            ])

    trends_md = out_dir / "trends.md"
    with trends_md.open("w", encoding="utf-8") as f:
        f.write("# Trends (rolling window = 5 scenes)\n\n")
        f.write("- `roll5_energy`: rolling mean of global energy z-score\n")
        f.write("- `roll5_dev`: rolling mean of deviation z-score\n")
        f.write("- `roll5_words`: rolling mean of scene length\n\n")
        f.write("See `trends.csv` for plotting.\n")

    md_path = out_dir / "timeline.md"
    runs = detect_runs(rows)

    spikes_all = top_n(rows, "z_energy_global", top_n_count, reverse=True)
    spikes_prose = top_n(rows, "z_energy_type", top_n_count, reverse=True, where=lambda r: r["type"] == "PROSE")

    lulls_raw = top_n(rows, "z_energy_global", top_n_count, reverse=False)
    lulls_norm = top_n(rows, "energy_density", top_n_count, reverse=False)

    devs_all = top_n(rows, "z_deviation", top_n_count, reverse=True)

    types = sorted(set(r["type"] for r in rows))
    spikes_by_type: Dict[str, List[Dict]] = {
        t: top_n(rows, "z_energy_type", min(top_n_count, 10), reverse=True, where=lambda r, tt=t: r["type"] == tt)
        for t in types
    }

    def fmt_row(r: Dict) -> str:
        ch = r.get("chapter", "")
        sc = r.get("scene", "")
        t = r.get("type", "")
        return (
            f"- **{r['filename']}**  (CH={ch} SC={sc} TYPE={t})\n"
            f"  - TITLE: {r.get('title','')}\n"
            f"  - words={r.get('words')}  punct/1k={float(r.get('punct_per_1k',0.0)):.2f}  dlg={float(r.get('dialogue_line_ratio',0.0)):.3f}  uniq={float(r.get('unique_token_ratio',0.0)):.3f}\n"
            f"  - z_energy_global={float(r.get('z_energy_global',0.0)):.3f}  z_energy_type={float(r.get('z_energy_type',0.0)):.3f}  z_dev={float(r.get('z_deviation',0.0)):.3f}\n"
        )

    def fmt_transition(prev: Dict, cur: Dict, idx: int) -> str:
        # idx is current index in rows
        dz = float(cur.get("z_energy_global", 0.0)) - float(prev.get("z_energy_global", 0.0))
        devz = float(cur.get("z_deviation", 0.0))
        top2 = top_changed_components(prev, cur, k=2)

        def short_type(r: Dict) -> str:
            return f"{r.get('type','')}"

        def fmt_comp(name: str, delta: float) -> str:
            # small, readable scaling
            if name in ("dialogue_line_ratio", "unique_token_ratio", "bigram_novelty_ratio"):
                return f"{name} {delta:+.3f}"
            if name == "punct_per_1k":
                return f"{name} {delta:+.2f}"
            if name == "words":
                return f"{name} {int(delta):+d}"
            return f"{name} {delta:+.3f}"

        changed = ", ".join(fmt_comp(n, d) for n, d in top2)

        return (
            f"- idx={idx}\n"
            f"  - prev: **{prev['filename']}** (TYPE={short_type(prev)}) zE={float(prev.get('z_energy_global',0.0)):.3f}\n"
            f"  - cur : **{cur['filename']}** (TYPE={short_type(cur)}) zE={float(cur.get('z_energy_global',0.0)):.3f}  ΔzE={dz:+.3f}  zDev={devz:.3f}\n"
            f"  - biggest changes: {changed}\n"
        )

    # Transition hotspots: take top deviations, then emit (prev -> cur)
    transition_hotspots: List[Tuple[int, Dict]] = []
    # Sort indices by z_deviation desc, skip idx=0 since no prev
    idxs = list(range(1, len(rows)))
    idxs.sort(key=lambda i: float(rows[i].get("z_deviation", 0.0)), reverse=True)
    for i in idxs[:top_n_count]:
        transition_hotspots.append((i, rows[i]))

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Timeline Diagnostics\n\n")
        f.write(f"Scenes: {len(rows)}\n\n")

        # config block
        f.write("## Normalisation / dampening config\n\n")
        f.write(f"- punct/1k winsorize: p{cfg['punct_winsor_p_lo']:.2f}..p{cfg['punct_winsor_p_hi']:.2f} "
                f"(lo={cfg['punct_winsor_lo_value']:.2f}, hi={cfg['punct_winsor_hi_value']:.2f})\n")
        f.write(f"- short-scene scaling: floor_words={cfg['short_floor_words']} exponent={cfg['short_exponent']}\n\n")

        f.write("## Top spikes (global)\n\n")
        for r in spikes_all:
            f.write(fmt_row(r) + "\n")

        f.write("## Top spikes (PROSE only, within-type)\n\n")
        if spikes_prose:
            for r in spikes_prose:
                f.write(fmt_row(r) + "\n")
        else:
            f.write("- (No PROSE scenes classified)\n\n")

        f.write("## Top deviations (global)\n\n")
        for r in devs_all:
            f.write(fmt_row(r) + "\n")

        # Upgrade A
        f.write("## Transition hotspots (prev → cur)\n\n")
        f.write("These are the highest deviation points, printed as the *actual cut* between scenes.\n\n")
        for idx, cur in transition_hotspots:
            prev = rows[idx - 1]
            f.write(fmt_transition(prev, cur, idx=idx) + "\n")

        f.write("## Lulls (global z, raw)\n\n")
        for r in lulls_raw:
            f.write(fmt_row(r) + "\n")

        f.write("## Lulls (length-normalised by density)\n\n")
        for r in lulls_norm:
            f.write(fmt_row(r) + "\n")

        f.write("## Spikes by type (within-type z)\n\n")
        for t in types:
            f.write(f"### {t}\n\n")
            for r in spikes_by_type[t]:
                f.write(fmt_row(r) + "\n")

        f.write("## Runs / clusters\n\n")
        f.write("### Longest type runs\n\n")
        for run in runs["type_runs"][:10]:
            s = rows[run["start"]]
            e = rows[run["end"]]
            f.write(f"- {run['label']} length={run['length']}  from {s['filename']} to {e['filename']}\n")

        f.write("\n### Longest low-energy runs\n\n")
        for run in runs["low_runs"][:10]:
            s = rows[run["start"]]
            e = rows[run["end"]]
            f.write(f"- {run['label']} length={run['length']}  from {s['filename']} to {e['filename']}\n")

        f.write("\n### Longest high-deviation runs\n\n")
        for run in runs["dev_runs"][:10]:
            s = rows[run["start"]]
            e = rows[run["end"]]
            f.write(f"- {run['label']} length={run['length']}  from {s['filename']} to {e['filename']}\n")
        f.write("\n")

    print(f"Wrote:   {metrics_path}")
    print(f"Wrote:   {md_path}")
    print(f"Wrote:   {csv_path}")
    print(f"Wrote:   {trends_csv}")
    print(f"Wrote:   {trends_md}")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic editorial signal audit (spikes/lulls/deviations/runs).")
    ap.add_argument("--corpus-dir", required=True, help="Directory containing scene .txt files")
    ap.add_argument("--out-dir", default="./reports", help="Output directory")
    ap.add_argument("--top-n", type=int, default=20, help="How many items per top list to show")

    # Upgrade B knobs (optional overrides)
    ap.add_argument("--punct-winsor-lo", type=float, default=0.01, help="Lower percentile for punct/1k winsorize (0..1)")
    ap.add_argument("--punct-winsor-hi", type=float, default=0.95, help="Upper percentile for punct/1k winsorize (0..1)")
    ap.add_argument("--short-floor-words", type=int, default=250, help="Word floor for short-scene energy damping")
    ap.add_argument("--short-exponent", type=float, default=0.35, help="Exponent for short-scene damping (smaller=gentler)")

    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    scenes = load_scenes(corpus_dir)
    rows, cfg = compute_metrics(
        scenes,
        punct_winsor_p_lo=args.punct_winsor_lo,
        punct_winsor_p_hi=args.punct_winsor_hi,
        short_floor_words=args.short_floor_words,
        short_exponent=args.short_exponent,
    )

    add_zscores(rows)
    add_deviation(rows)

    print(f"Corpus:  {corpus_dir}")
    print(f"Out:     {out_dir}")
    print(f"Scenes:  {len(rows)}")

    write_outputs(rows, out_dir, top_n_count=args.top_n, cfg=cfg)


if __name__ == "__main__":
    main()
