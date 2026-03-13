#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request

import numpy as np


def ollama_post(url: str, payload: dict, timeout_s: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_embed(texts: List[str], *, url: str, model: str, timeout_s: int = 120) -> List[List[float]]:
    out = ollama_post(url, {"model": model, "input": texts}, timeout_s=timeout_s)
    if "embeddings" in out and isinstance(out["embeddings"], list):
        return out["embeddings"]
    if "embedding" in out and isinstance(out["embedding"], list):
        return [out["embedding"]]
    raise RuntimeError(f"Unexpected embed response keys: {list(out.keys())}")


def ollama_generate(prompt: str, *, url: str, model: str, timeout_s: int = 300) -> str:
    # Use /api/generate (non-streaming)
    out = ollama_post(url, {"model": model, "prompt": prompt, "stream": False}, timeout_s=timeout_s)
    if "response" in out:
        return out["response"]
    raise RuntimeError(f"Unexpected generate response keys: {list(out.keys())}")


def load_meta(meta_path: Path) -> List[Dict]:
    rows = []
    with meta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cosine_topk(matrix: np.ndarray, q: np.ndarray, k: int) -> List[Tuple[int, float]]:
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn
    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def read_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Skip structured header until first blank line
    i = 0
    while i < len(lines) and lines[i].strip() != "":
        i += 1
    if i < len(lines) and lines[i].strip() == "":
        i += 1

    body = "\n".join(lines[i:]).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def snippet_from_body(body: str, max_chars: int = 900) -> str:
    body = body.strip()
    if len(body) <= max_chars:
        return body
    return body[:max_chars] + "…"

def build_rerank_prompt(query: str, cands: List[dict]) -> str:
    items = []
    for c in cands:
        items.append({
            "key": c["key"],
            "chapter": c.get("chapter",""),
            "scene": c.get("scene",""),
            "title": c.get("title",""),
            "source": c.get("source",""),
            "text": c["text"][:1800],
        })

    return (
        "You are selecting the most relevant candidates for a query.\n"
        "Return JSON ONLY.\n\n"
        "OUTPUT FORMAT (exact):\n"
        "{\"keys\":[\"A01\",\"A02\",\"A03\"]}\n\n"
        "RULES:\n"
        "1) Output must be valid JSON and must start with '{' and end with '}'.\n"
        "2) Only output keys from the provided candidates (A01..A30). Do not invent new strings.\n"
        "3) Return at least 10 keys (unless fewer than 10 candidates were provided).\n"
        "4) Order keys from most relevant to least relevant.\n\n"
        f"QUERY:\n{query}\n\n"
        f"CANDIDATES:\n{json.dumps(items, ensure_ascii=False)}\n"
    )

def parse_rerank_json(s: str) -> dict:
    """
    Extract the first JSON object found in the model output and parse it.
    This makes reranking robust even if the model adds commentary.
    """
    s = s.strip()

    # Fast path: already valid JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract a JSON object {...} from the text
    start = s.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output")

    # Scan for a balanced JSON object
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    return json.loads(candidate)

    raise ValueError("Found '{' but could not extract a balanced JSON object")

def main() -> None:
    ap = argparse.ArgumentParser(description="Query index + rerank using an Ollama generation model.")
    ap.add_argument("--index-dir", required=True, help="Directory containing vectors.npy and meta.jsonl")
    ap.add_argument("--embed-model", default=os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma"),
                    help="Embedding model used for index/query (default: embeddinggemma)")
    ap.add_argument("--gen-model", default=os.environ.get("OLLAMA_GEN_MODEL", "qwen2.5:7b"),
                    help="Generation model for reranking (default: qwen2.5:7b)")
    ap.add_argument("--embed-url", default=os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
                    help="Ollama embed endpoint (default: http://localhost:11434/api/embed)")
    ap.add_argument("--gen-url", default=os.environ.get("OLLAMA_GEN_URL", "http://localhost:11434/api/generate"),
                    help="Ollama generate endpoint (default: http://localhost:11434/api/generate)")
    ap.add_argument("--retrieve-k", type=int, default=30, help="How many candidates to retrieve before rerank (default: 30)")
    ap.add_argument("--topk", type=int, default=8, help="How many hits to display after rerank (default: 8)")
    ap.add_argument("--snippet-chars", type=int, default=900, help="Snippet length (default: 900)")
    ap.add_argument("query", nargs="+", help="Query text")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    vec_path = index_dir / "vectors.npy"
    meta_path = index_dir / "meta.jsonl"

    X = np.load(vec_path).astype(np.float32)
    meta = load_meta(meta_path)
    if X.shape[0] != len(meta):
        raise SystemExit(f"Mismatch: vectors {X.shape[0]} rows, meta {len(meta)} rows")

    qtext = " ".join(args.query).strip()
    qvec = np.array(ollama_embed([qtext], url=args.embed_url, model=args.embed_model)[0], dtype=np.float32)

    retrieved = cosine_topk(X, qvec, args.retrieve_k)

    # Build candidates with text
    cands = []
    for j, (cand_id, sim) in enumerate(retrieved, start=1):
        m = meta[cand_id]
        path = Path(m.get("path", m.get("source", "")))
        body = read_body(path)
        key = f"A{j:02d}"  # A01..A30

        cands.append({
            "key": key,
            "id": cand_id,   # keep internal numeric id
            "sim": sim,
            "chapter": m.get("chapter",""),
            "scene": m.get("scene",""),
            "title": m.get("title",""),
            "source": m.get("source",""),
            "path": str(path),
            "text": body,
        })

    prompt = build_rerank_prompt(qtext, cands)
    resp = ollama_generate(prompt, url=args.gen_url, model=args.gen_model)

    try:
        rr = parse_rerank_json(resp)
        keys = rr.get("keys", [])
        if not keys:
            print("\n[DEBUG] Reranker returned no keys.\nRaw model output:\n")
            print(resp)
            raise SystemExit(2)

    except Exception:
        # Fallback: ask the model to convert its own previous answer into strict JSON
        fix_prompt = (
            "Convert the following text into STRICT JSON with schema:\n"
            "{\"keys\":[\"A01\",\"A02\",\"A03\"]}\n"
            "Output JSON ONLY (start with '{' end with '}').\n\n"
            f"TEXT:\n{resp}\n"
        )
        resp2 = ollama_generate(fix_prompt, url=args.gen_url, model=args.gen_model)
        try:
            rr = parse_rerank_json(resp2)
            keys = rr.get("keys", [])
        except Exception as e2:
            raise SystemExit(
                f"Failed to parse rerank JSON (even after fallback). Error={e2}\n"
                f"Raw response 1:\n{resp}\n\nRaw response 2:\n{resp2}"
            )

    # Map id -> cand

    by_key = {c["key"]: c for c in cands}

    bad = [k for k in keys if not isinstance(k, str) or k not in by_key]
    if bad:
        print(f"[WARN] Reranker returned {len(bad)} invalid keys (ignored): {bad}")

    valid_keys = [k for k in keys if isinstance(k, str) and k in by_key]

    # Print results
    print()
    print(f"Query: {qtext}")
    print(f"Index: {index_dir}")
    print(f"Retrieved: {len(cands)}  Reranked: {len(valid_keys)} (valid)")
    print(f"Embed model: {args.embed_model}   Gen model: {args.gen_model}")
    print()

    shown = 0
    for k in valid_keys[:args.topk]:
        c = by_key[k]

        print(f"{shown+1:>2}.  sim={c['sim']:.4f}   CH={c['chapter']} SC={c['scene']}")
        print(f"     TITLE: {c['title']}")
        print(f"     FILE:  {c['path']}")
        print("     ---")
        print(snippet_from_body(c["text"], max_chars=args.snippet_chars).replace("\n", "\n     "))
        print()
        shown += 1

if __name__ == "__main__":
    main()
