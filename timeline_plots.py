#!/usr/bin/env python3
"""
timeline_plots.py

Usage:
  python timeline_plots.py timeline.csv --outdir out_timeline

Your CSV columns (as provided):
  filename,chapter,scene,type,title,words,lines,avg_line_len,punct,
  punct_per_1k_raw,punct_per_1k,dialogue_line_ratio,unique_token_ratio,
  bigram_novelty_ratio,short_scene_scale,energy_density,z_energy_global,
  z_energy_type,deviation,z_deviation,path

Output:
  - PNG graphs in --outdir
  - SUMMARY.md (succinct descriptions per graph)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotSpec:
    key: str
    ycol: str
    title: str
    ylabel: str
    note: str


PLOTS: List[PlotSpec] = [
    PlotSpec(
        key="01_z_energy_global",
        ycol="z_energy_global",
        title="Timeline — Energy (global z-score)",
        ylabel="z_energy_global",
        note=(
            "Per-scene energy relative to the full manuscript. "
            "0 is the manuscript mean; positive values are denser/more 'energetic', "
            "negative values are calmer/lower-density. Chapter boundaries are marked."
        ),
    ),
    PlotSpec(
        key="02_z_energy_type",
        ycol="z_energy_type",
        title="Timeline — Energy (within-type z-score)",
        ylabel="z_energy_type",
        note=(
            "Energy relative to the scene's TYPE cohort (PROSE vs MEMO vs TRANSCRIPT, etc.). "
            "Useful for spotting outliers *within* a form (e.g., an unusually intense MEMO)."
        ),
    ),
    PlotSpec(
        key="03_z_deviation",
        ycol="z_deviation",
        title="Timeline — Deviation (z)",
        ylabel="z_deviation",
        note=(
            "How atypical each scene is relative to the manuscript baseline. "
            "High deviation often corresponds to hard shifts in mode/voice/format or genuine structural outliers."
        ),
    ),
    PlotSpec(
        key="04_punct_per_1k",
        ycol="punct_per_1k",
        title="Timeline — Punctuation density (punct/1k, dampened)",
        ylabel="punct_per_1k",
        note=(
            "The dampened punctuation density per scene (after your winsorisation / scaling). "
            "Tracks compression vs plainness: clause-heavy scenes rise; flatter declaratives fall."
        ),
    ),
    PlotSpec(
        key="05_punct_per_1k_raw",
        ycol="punct_per_1k_raw",
        title="Timeline — Punctuation density (punct/1k, raw)",
        ylabel="punct_per_1k_raw",
        note=(
            "Raw punctuation density per scene (no winsorisation). "
            "Helpful for checking whether clipping is hiding meaningful extremes."
        ),
    ),
    PlotSpec(
        key="06_words",
        ycol="words",
        title="Timeline — Scene length (words)",
        ylabel="words",
        note=(
            "Scene length over the full manuscript. Long scenes often dominate perceived pacing; "
            "clusters of short scenes can inflate density metrics without short-scene scaling."
        ),
    ),
    PlotSpec(
        key="07_dialogue_line_ratio",
        ycol="dialogue_line_ratio",
        title="Timeline — Dialogue line ratio",
        ylabel="dialogue_line_ratio (0..1)",
        note=(
            "Share of lines classified as dialogue. 0 means no dialogue; 1 means all lines are dialogue. "
            "Good for spotting 'air' vs 'pressure' and mode shifts (briefings, intimacy, action, etc.)."
        ),
    ),
    PlotSpec(
        key="08_unique_token_ratio",
        ycol="unique_token_ratio",
        title="Timeline — Unique token ratio",
        ylabel="unique_token_ratio",
        note=(
            "Approx lexical variety proxy. Higher values suggest more varied diction / technical noun density; "
            "lower values suggest repetition, procedural cadence, or chant-like phrasing."
        ),
    ),
    PlotSpec(
        key="09_bigram_novelty_ratio",
        ycol="bigram_novelty_ratio",
        title="Timeline — Bigram novelty ratio",
        ylabel="bigram_novelty_ratio",
        note=(
            "How 'new' adjacent word-pairs are, relative to the manuscript's accumulated patterns. "
            "Spikes can mark rhetorical invention, new domain vocabulary, or sudden stylistic departures."
        ),
    ),
    PlotSpec(
        key="10_short_scene_scale",
        ycol="short_scene_scale",
        title="Timeline — Short-scene scaling factor",
        ylabel="short_scene_scale",
        note=(
            "The scaling factor applied to dampen short-scene volatility. "
            "Look for where this is very low (short scenes) to interpret spikes cautiously."
        ),
    ),
    PlotSpec(
        key="11_energy_density",
        ycol="energy_density",
        title="Timeline — Energy density (raw metric)",
        ylabel="energy_density",
        note=(
            "Your underlying energy density metric per scene. "
            "This shows the raw signal that z_energy_global standardises."
        ),
    ),
]


REQUIRED_COLS = [
    "filename",
    "chapter",
    "scene",
    "type",
    "title",
    "words",
    "punct_per_1k_raw",
    "punct_per_1k",
    "dialogue_line_ratio",
    "unique_token_ratio",
    "bigram_novelty_ratio",
    "short_scene_scale",
    "energy_density",
    "z_energy_global",
    "z_energy_type",
    "z_deviation",
]


def require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def add_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there's an 'idx' column: 0..N-1 in reading order.
    We sort by (chapter, scene) and assume those are numeric-ish.
    """
    d = df.copy()

    # Coerce chapter/scene to ints if possible; otherwise stable sort by string.
    def to_int_series(s: pd.Series) -> pd.Series:
        try:
            return pd.to_numeric(s, errors="raise").astype(int)
        except Exception:
            return pd.to_numeric(s, errors="coerce").fillna(-1).astype(int)

    d["__chapter_i"] = to_int_series(d["chapter"])
    d["__scene_i"] = to_int_series(d["scene"])

    d = d.sort_values(["__chapter_i", "__scene_i", "filename"], kind="mergesort").reset_index(drop=True)
    d["idx"] = np.arange(len(d), dtype=int)

    d = d.drop(columns=["__chapter_i", "__scene_i"])
    return d

def chapter_boundaries(d: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Return list of (chapter_number, start_idx, end_idx) boundaries in idx space.
    Rows with missing chapter numbers are ignored for boundary purposes.
    """
    out: List[Tuple[int, int, int]] = []
    cur_ch: Optional[int] = None
    start: Optional[int] = None
    last_valid_idx: Optional[int] = None

    for _, row in d.iterrows():
        if pd.isna(row["chapter"]):
            continue

        ch = int(row["chapter"])
        idx = int(row["idx"])

        if cur_ch is None:
            cur_ch = ch
            start = idx
        elif ch != cur_ch:
            assert start is not None
            assert last_valid_idx is not None
            out.append((cur_ch, start, last_valid_idx))
            cur_ch = ch
            start = idx

        last_valid_idx = idx

    if cur_ch is not None and start is not None and last_valid_idx is not None:
        out.append((cur_ch, start, last_valid_idx))

    return out

def draw_chapter_guides(ax: plt.Axes, bounds: List[Tuple[int, int, int]]) -> None:
    # Vertical separators at chapter starts (except first)
    for k, (_, start, _) in enumerate(bounds):
        if k == 0:
            continue
        ax.axvline(start, linewidth=1, alpha=0.35)

    # Labels near top
    ymin, ymax = ax.get_ylim()
    for (ch, start, end) in bounds:
        mid = (start + end) / 2.0
        ax.text(
            mid,
            ymax - (ymax - ymin) * 0.06,
            f"CH {ch}",
            ha="center",
            va="top",
            fontsize=8,
            alpha=0.65,
        )


def save_plot(fig: plt.Figure, outpath: str) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_timeline(d: pd.DataFrame, spec: PlotSpec, outdir: str) -> str:
    x = d["idx"].to_numpy()
    y = d[spec.ycol].to_numpy()

    fig = plt.figure(figsize=(14, 4.2))
    ax = plt.gca()
    ax.plot(x, y)

    ax.set_title(spec.title)
    ax.set_xlabel("Scene index across entire manuscript (computed idx)")
    ax.set_ylabel(spec.ylabel)

    bounds = chapter_boundaries(d)
    draw_chapter_guides(ax, bounds)

    outpath = os.path.join(outdir, f"{spec.key}.png")
    save_plot(fig, outpath)
    return outpath


def plot_heatmap_energy_by_type(d: pd.DataFrame, outdir: str) -> str:
    pivot = d.pivot_table(
        index="type",
        columns="idx",
        values="z_energy_global",
        aggfunc="mean",
    ).reindex(sorted(d["type"].unique()))

    fig = plt.figure(figsize=(14, 4.6))
    ax = plt.gca()
    im = ax.imshow(pivot.to_numpy(), aspect="auto")

    ax.set_title("Heatmap — z_energy_global by TYPE across the manuscript")
    ax.set_xlabel("Scene index (idx)")
    ax.set_ylabel("Type")

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    cols = pivot.columns.to_numpy()
    if len(cols) > 1:
        step = max(1, int(len(cols) / 10))
        xt = np.arange(0, len(cols), step)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(int(cols[i])) for i in xt])

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    outpath = os.path.join(outdir, "12_heatmap_energy_by_type.png")
    save_plot(fig, outpath)
    return outpath


def plot_transition_delta_energy(d: pd.DataFrame, outdir: str) -> str:
    dz = d["z_energy_global"].diff().to_numpy()

    fig = plt.figure(figsize=(14, 4.2))
    ax = plt.gca()
    ax.plot(d["idx"].to_numpy(), dz)

    ax.set_title("Transition stress — Δ z_energy_global between consecutive scenes")
    ax.set_xlabel("Scene index (idx)")
    ax.set_ylabel("delta_z_energy_global (diff)")

    bounds = chapter_boundaries(d)
    draw_chapter_guides(ax, bounds)

    outpath = os.path.join(outdir, "13_transition_delta_energy.png")
    save_plot(fig, outpath)
    return outpath


def write_summary(outdir: str, plot_paths: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Timeline Graphs — Summary")
    lines.append("")
    lines.append(
        "All plots are full-manuscript views. X-axis is a computed scene index (`idx`) "
        "created by sorting by (chapter, scene). Chapter boundaries are marked by vertical lines and labels."
    )
    lines.append("")

    for spec in PLOTS:
        fname = f"{spec.key}.png"
        if any(p.endswith(fname) for p in plot_paths):
            lines.append(f"## {spec.title}")
            lines.append("")
            lines.append(spec.note)
            lines.append("")
            lines.append(f"- File: `{fname}`")
            lines.append("")

    if any(p.endswith("12_heatmap_energy_by_type.png") for p in plot_paths):
        lines.append("## Heatmap — z_energy_global by TYPE across the manuscript")
        lines.append("")
        lines.append(
            "A compact overview of where each TYPE concentrates energy across the novel. "
            "Useful for diagnosing whether 'spikes' are mainly type-driven (e.g., MEMO style) "
            "or arise within long PROSE runs."
        )
        lines.append("")
        lines.append("- File: `12_heatmap_energy_by_type.png`")
        lines.append("")

    if any(p.endswith("13_transition_delta_energy.png") for p in plot_paths):
        lines.append("## Transition stress — Δ z_energy_global between consecutive scenes")
        lines.append("")
        lines.append(
            "Shows the *cuts* that behave like hard gear changes. Large magnitude values correspond to "
            "your 'transition hotspot' concept (prev → cur abruptness)."
        )
        lines.append("")
        lines.append("- File: `13_transition_delta_energy.png`")
        lines.append("")

    outpath = os.path.join(outdir, "SUMMARY.md")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="timeline.csv (one row per scene)")
    ap.add_argument("--outdir", default="out_timeline", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    require_cols(df, REQUIRED_COLS)

    d = add_idx(df)

    plot_paths: List[str] = []
    for spec in PLOTS:
        # only plot if column exists (defensive)
        if spec.ycol in d.columns:
            plot_paths.append(plot_timeline(d, spec, args.outdir))

    plot_paths.append(plot_heatmap_energy_by_type(d, args.outdir))
    plot_paths.append(plot_transition_delta_energy(d, args.outdir))

    summary_path = write_summary(args.outdir, plot_paths)

    print(f"OK — wrote {len(plot_paths)} graphs to: {args.outdir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
