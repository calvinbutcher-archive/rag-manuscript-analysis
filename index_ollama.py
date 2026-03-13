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


HDR_RE = re.compile(r"^(CHAPTER|SCENE|TITLE|SOURCE):\s*(.*)\s*$", re.IGNORECASE)
SCENE_FILE_RE = re.compile(r"^\d{2}_\d{2}_.*\.txt$", re.IGNORECASE)
SCENE_NUM_RE = re.compile(r"^(\d{2})_(\d{2})_")


def ollama_embed(
    texts: List[str],
    *,
    url: str,
    model: str,
    timeout_s: int = 120,
) -> List[List[float]]:
    """
    Calls Ollama embeddings endpoint.
    Expects POST { "model": "...", "input": ["...", "..."] } to /api/embed
    Returns list of embedding vectors (list[float]).
    """
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


def parse_scene_file(path: Path, corpus_dir: Path) -> Tuple[Dict[str, str], str]:
    """
    Parses header block of KEY: VALUE lines until first blank line.
    Returns metadata dict and remaining text body.
    """
    meta: Dict[str, str] = {}
    rest_lines: List[str] = []
    in_header = True

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if in_header:
                if line.strip() == "":
                    in_header = False
                    continue
                m = HDR_RE.match(line)
                if m:
                    key = m.group(1).lower()
                    meta[key] = m.group(2).strip()
                    continue
            rest_lines.append(line)

    # Use relative path for source so citations are actually useful later.
    rel_path = str(path.relative_to(corpus_dir))
    meta.setdefault("source", rel_path)

    body = "".join(rest_lines).strip()
    return meta, body


def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Simple character-based chunking.
    """
    text = text.strip()
    if not text:
        return []

    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap_chars)

    return chunks


def normalize_for_embed(meta: Dict[str, str], body: str) -> str:
    """
    Text actually sent for embedding.
    """
    title = meta.get("title", "").strip()
    chap = meta.get("chapter", "").strip()
    scene = meta.get("scene", "").strip()

    prefix: List[str] = []
    if chap and scene:
        prefix.append(f"CHAPTER {chap} SCENE {scene}")
    if title:
        prefix.append(f"TITLE: {title}")

    head = "\n".join(prefix).strip()
    return (head + "\n\n" + body).strip() if head else body.strip()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a local vector index using Ollama embeddings."
    )
    ap.add_argument(
        "--corpus-dir",
        required=True,
        help="Directory containing scene .txt files (recursive).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write vectors/meta/manifests.",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_EMBED_MODEL", "embeddinggemma"),
        help="Ollama embedding model name (default: embeddinggemma).",
    )
    ap.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
        help="Ollama embeddings endpoint (default: http://localhost:11434/api/embed).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="Chunk size in characters (default: 6000).",
    )
    ap.add_argument(
        "--overlap-chars",
        type=int,
        default=800,
        help="Chunk overlap in characters (default: 800).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="How many chunks to embed per request (default: 16).",
    )
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise SystemExit(f"Corpus directory does not exist or is not a directory: {corpus_dir}")

    safe_mkdir(out_dir)

    # Recursive search so ACT_I/Chapter_1/... works.
    files = sorted(
        f for f in corpus_dir.rglob("*.txt")
        if SCENE_FILE_RE.match(f.name)
    )

    if not files:
        raise SystemExit(
            f"No matching scene .txt files found under: {corpus_dir}\n"
            "Expected names like: 01_01_The_Lights.txt"
        )

    manifest_path = out_dir / "manifest.txt"
    manifest_path.write_text(
        "\n".join(str(f.relative_to(corpus_dir)) for f in files) + "\n",
        encoding="utf-8",
    )

    meta_path = out_dir / "meta.jsonl"
    vec_path = out_dir / "vectors.npy"

    all_vecs: List[List[float]] = []
    total_chunks = 0

    with meta_path.open("w", encoding="utf-8") as mf:
        batch_texts: List[str] = []
        batch_records: List[Dict[str, object]] = []

        def flush_batch() -> None:
            nonlocal total_chunks
            if not batch_texts:
                return

            embs = ollama_embed(
                batch_texts,
                url=args.ollama_url,
                model=args.model,
            )

            if len(embs) != len(batch_texts):
                raise RuntimeError(
                    f"Embedding count mismatch: got {len(embs)} expected {len(batch_texts)}"
                )

            for emb, rec in zip(embs, batch_records):
                all_vecs.append(emb)
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total_chunks += len(batch_texts)
            batch_texts.clear()
            batch_records.clear()

        for fp in files:
            meta, body = parse_scene_file(fp, corpus_dir)

            # Best-effort chapter/scene inference from filename.
            m = SCENE_NUM_RE.match(fp.name)
            if m:
                meta.setdefault("chapter", str(int(m.group(1))))
                meta.setdefault("scene", str(int(m.group(2))))

            chunks = chunk_text(body, args.max_chars, args.overlap_chars)
            if not chunks:
                continue

            chunk_count = len(chunks)

            for i, chunk in enumerate(chunks):
                embed_text = normalize_for_embed(meta, chunk)

                rec: Dict[str, object] = {
                    "chapter": meta.get("chapter", ""),
                    "scene": meta.get("scene", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", str(fp.relative_to(corpus_dir))),
                    "path": str(fp),
                    "rel_path": str(fp.relative_to(corpus_dir)),
                    "chunk_id": i,
                    "chunk_count": chunk_count,
                }

                batch_texts.append(embed_text)
                batch_records.append(rec)

                if len(batch_texts) >= args.batch_size:
                    flush_batch()

        flush_batch()

    if not all_vecs:
        raise SystemExit("No chunks were generated; nothing was indexed.")

    X = np.asarray(all_vecs, dtype=np.float32)
    np.save(vec_path, X)

    print(f"Corpus:      {corpus_dir}")
    print(f"Out:         {out_dir}")
    print(f"Files:       {len(files)}")
    print(f"Chunks:      {total_chunks}")
    print(f"Vectors:     {X.shape}")
    print(f"Wrote:       {vec_path}")
    print(f"Wrote:       {meta_path}")
    print(f"Wrote:       {manifest_path}")


if __name__ == "__main__":
    main()
