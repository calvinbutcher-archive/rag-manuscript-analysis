#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request

import numpy as np


HDR_RE = re.compile(r"^(CHAPTER|SCENE|TITLE|SOURCE):\s*(.*)\s*$", re.IGNORECASE)


def ollama_embed(texts: List[str], *, url: str, model: str, timeout_s: int = 120) -> List[List[float]]:
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]
    if "embedding" in data and isinstance(data["embedding"], list):
        return [data["embedding"]]
    raise RuntimeError(f"Unexpected Ollama response keys: {list(data.keys())}")


def load_meta(meta_path: Path) -> List[Dict]:
    rows = []
    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cosine_topk(matrix: np.ndarray, q: np.ndarray, k: int) -> List[Tuple[int, float]]:
    # matrix: (N, D), q: (D,)
    # cosine similarity = (A·B)/(|A||B|)
    # Pre-normalize
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn
    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]

def read_snippet(path: Path, *, max_chars: int = 900) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Could not read file: {e}]"

    lines = text.splitlines()

    # Skip structured header block until first blank line
    i = 0
    while i < len(lines) and lines[i].strip() != "":
        i += 1
    if i < len(lines) and lines[i].strip() == "":
        i += 1

    # Now skip SYNOPSIS block if present
    if i < len(lines) and lines[i].startswith("SYNOPSIS:"):
        i += 1
        # skip bullet lines and blanks immediately following synopsis
        while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("-")):
            i += 1

    # skip extra blank lines before prose
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    body = "\n".join(lines[i:]).strip()
    if not body:
        return "[No body text found]"
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body[:max_chars] + ("…" if len(body) > max_chars else "")

def main() -> None:
    ap = argparse.ArgumentParser(description="Query a local Ollama-embedded index (cosine similarity).")
    ap.add_argument("--index-dir", required=True, help="Directory containing vectors.npy and meta.jsonl")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma"),
                    help="Ollama embedding model name (default: embeddinggemma)")
    ap.add_argument("--ollama-url", default=os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
                    help="Ollama embeddings endpoint (default: http://localhost:11434/api/embed)")
    ap.add_argument("--topk", type=int, default=8, help="Number of hits to return (default: 8)")
    ap.add_argument("--snippet-chars", type=int, default=900, help="Snippet length (default: 900)")
    ap.add_argument("--open", action="store_true",
                    help="Open a hit in vim (default: top hit).")
    ap.add_argument("--open-rank", type=int, default=1,
                    help="Which ranked hit to open when --open is set (default: 1).")
    ap.add_argument("query", nargs="+", help="Query text")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    vec_path = index_dir / "vectors.npy"
    meta_path = index_dir / "meta.jsonl"

    if not vec_path.exists():
        raise SystemExit(f"Missing: {vec_path}")
    if not meta_path.exists():
        raise SystemExit(f"Missing: {meta_path}")

    X = np.load(vec_path).astype(np.float32)
    meta = load_meta(meta_path)
    if X.shape[0] != len(meta):
        raise SystemExit(f"Mismatch: vectors {X.shape[0]} rows, meta {len(meta)} rows")

    qtext = " ".join(args.query).strip()
    qvec = np.array(ollama_embed([qtext], url=args.ollama_url, model=args.model)[0], dtype=np.float32)

    hits = cosine_topk(X, qvec, args.topk)

    if args.open:
        if not hits:
            raise SystemExit("No hits to open.")
        r = args.open_rank
        if r < 1 or r > len(hits):
            raise SystemExit(f"--open-rank must be between 1 and {len(hits)} (got {r})")
        idx, score = hits[r - 1]
        m = meta[idx]
        path = Path(m.get("path", m.get("source", ""))).expanduser()

        if not path.exists():
            raise SystemExit(f"File does not exist: {path}")

        # Open in vim. If you want to use $EDITOR instead, tell me.
        subprocess.run(["vim", str(path)])
        return

    print()
    print(f"Query: {qtext}")
    print(f"Index: {index_dir}")
    print(f"Hits:  {len(hits)}")
    print()

    for rank, (i, score) in enumerate(hits, start=1):
        m = meta[i]
        path = Path(m.get("path", m.get("source", "")))
        chap = m.get("chapter", "")
        scene = m.get("scene", "")
        title = m.get("title", "")
        chunk_id = m.get("chunk_id", "")
        chunk_count = m.get("chunk_count", "")

        print(f"{rank:>2}.  score={score:.4f}   CH={chap} SC={scene}   chunk={chunk_id}/{chunk_count}")
        print(f"     TITLE: {title}")
        print(f"     FILE:  {path}")
        print("     ---")
        print(read_snippet(path, max_chars=args.snippet_chars).replace("\n", "\n     "))
        print()

if __name__ == "__main__":
    main()
