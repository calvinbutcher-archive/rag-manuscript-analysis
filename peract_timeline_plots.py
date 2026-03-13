#!/usr/bin/env python3
"""
mod_timeline_plots.py

Adds “modulated” / spotlight graphs on top of the existing timeline plots:

A) Act windows (zoomed energy plots per act)
B) Transition shock plot: |Δ z_energy_global| with top-N annotated cut points
C) Rolling median + IQR (25–75%) band (“envelope”) for z_energy_global
D) Chapter aggregates: mean z_energy_global + variability + mean |Δ|

Works with your timeline.csv.

Example:
  python mod_timeline_plots.py ./reports/timeline.csv --outdir out_mod
  python mod_timeline_plots.py ./reports/timeline.csv --outdir out_mod_prose --prose-only
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ActRange:
    name: str
    ch_lo: int
    ch_hi: int  # inclusive


DEFAULT_ACTS = [
    ActRange("Act I", 1, 5),
    ActRange("Act II", 6, 9),
    ActRange("Act III", 10, 15),
]


def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_text(s: str, limit: int = 60) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _load_timeline(csv_path: Path, prose_only: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Basic sanity
    required = [
        "filename",
        "chapter",
        "scene",
        "type",
        "title",
        "words",
        "z_energy_global",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"timeline.csv missing required columns: {missing}")

    # Preserve row order: assume the CSV is already in manuscript order.
    df = df.copy()
    df["idx"] = np.arange(len(df), dtype=int) + 1

    # Normalise types
    df["type"] = df["type"].astype(str).str.upper().str.strip()

    if prose_only:
        df = df[df["type"] == "PROSE"].copy()
        df["idx"] = np.arange(len(df), dtype=int) + 1

    # Ensure numerics
    for col in ["chapter", "scene", "words", "z_energy_global"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["chapter", "scene", "words", "z_energy_global"]).copy()
    df["chapter"] = df["chapter"].astype(int)
    df["scene"] = df["scene"].astype(int)
    df["words"] = df["words"].astype(int)

    return df


def _savefig(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_act_windows_energy(
    df: pd.DataFrame,
    outdir: Path,
    acts: List[ActRange],
) -> List[Tuple[str, str]]:
    """
    For each act, save a zoomed z_energy_global vs scene index.
    """
    outputs: List[Tuple[str, str]] = []

    for act in acts:
        sub = df[(df["chapter"] >= act.ch_lo) & (df["chapter"] <= act.ch_hi)].copy()
        if sub.empty:
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(sub["idx"], sub["z_energy_global"])
        plt.axhline(0.0, linewidth=1)

        plt.title(
            f"{act.name}: z_energy_global (Scenes {int(sub['idx'].min())}–{int(sub['idx'].max())}, "
            f"Ch {act.ch_lo}–{act.ch_hi})"
        )
        plt.xlabel("Scene index (within filtered timeline)")
        plt.ylabel("z_energy_global")

        out = outdir / f"mod_01_energy_act_{act.ch_lo:02d}_{act.ch_hi:02d}.png"
        _savefig(out)

        outputs.append(
            (
                out.name,
                f"Zoomed **z_energy_global** for {act.name} (chapters {act.ch_lo}–{act.ch_hi}). "
                f"Shows the energy waveform *within that act* on its own scale.",
            )
        )

    return outputs


def plot_transition_shocks(
    df: pd.DataFrame,
    outdir: Path,
    top_n: int = 15,
) -> List[Tuple[str, str]]:
    """
    Plot absolute delta between adjacent scenes in z_energy_global and annotate top N cut points.
    """
    if len(df) < 3:
        return []

    z = df["z_energy_global"].to_numpy()
    delta = np.diff(z)
    abs_delta = np.abs(delta)

    # Indexing: delta[i] corresponds to cut from scene i -> i+1 (1-based idx in df)
    cut_idx = np.arange(1, len(df))  # 1..len-1, refers to "previous scene position"

    # Pick top N cuts
    top_n = max(1, int(top_n))
    top_ix = np.argsort(abs_delta)[::-1][:top_n]

    plt.figure(figsize=(14, 5))
    plt.plot(cut_idx + 1, abs_delta)  # plot at "current scene index" (i+1)
    plt.title(f"Transition shocks: |Δ z_energy_global| between adjacent scenes (top {top_n} annotated)")
    plt.xlabel("Cut position (current scene index)")
    plt.ylabel("|Δ z_energy_global|")

    # Annotate
    for j in top_ix:
        prev_row = df.iloc[j]
        cur_row = df.iloc[j + 1]
        x = int(cur_row["idx"])
        y = float(abs_delta[j])

        label = (
            f"{_safe_text(prev_row['title'], 22)} → {_safe_text(cur_row['title'], 22)} "
            f"(ch{int(prev_row['chapter'])}:{int(prev_row['scene'])}→"
            f"ch{int(cur_row['chapter'])}:{int(cur_row['scene'])})"
        )
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            rotation=0,
        )

    out = outdir / "mod_02_transition_shocks_abs_delta_z_energy.png"
    _savefig(out)

    return [
        (
            out.name,
            "Plots **|Δ z_energy_global|** at each *cut* (scene-to-scene transition). "
            "Annotated labels identify the **largest discontinuities**, which often correspond to "
            "hard tonal/structural shifts or deliberate “gear changes”.",
        )
    ]


def plot_rolling_envelope(
    df: pd.DataFrame,
    outdir: Path,
    window: int = 11,
) -> List[Tuple[str, str]]:
    """
    Rolling median with IQR band (25–75%) for z_energy_global.
    """
    if len(df) < 5:
        return []

    window = int(window)
    window = max(3, window if window % 2 == 1 else window + 1)

    s = df["z_energy_global"]

    med = s.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    q25 = s.rolling(window, center=True, min_periods=max(3, window // 2)).quantile(0.25)
    q75 = s.rolling(window, center=True, min_periods=max(3, window // 2)).quantile(0.75)

    plt.figure(figsize=(14, 5))
    x = df["idx"].to_numpy()

    plt.plot(x, s, linewidth=1)
    plt.plot(x, med, linewidth=2)
    plt.fill_between(x, q25.to_numpy(), q75.to_numpy(), alpha=0.25)

    plt.axhline(0.0, linewidth=1)
    plt.title(f"z_energy_global with rolling median + IQR band (window={window})")
    plt.xlabel("Scene index (within filtered timeline)")
    plt.ylabel("z_energy_global")

    out = outdir / "mod_03_energy_rolling_median_iqr.png"
    _savefig(out)

    return [
        (
            out.name,
            "Shows raw **z_energy_global** plus a **rolling median** (thicker line) and **IQR band** (25–75%). "
            "Use this to see whether the manuscript truly **narrows / widens in variance** over time, "
            "rather than just changing in peak height.",
        )
    ]


def plot_chapter_aggregates(
    df: pd.DataFrame,
    outdir: Path,
) -> List[Tuple[str, str]]:
    """
    Per-chapter aggregates: mean z_energy_global, std(z_energy_global),
    and mean absolute delta within chapter.
    """
    if df.empty:
        return []

    # Compute per-scene abs delta
    df = df.copy()
    df["delta_z_energy"] = df["z_energy_global"].diff()
    df["abs_delta_z_energy"] = df["delta_z_energy"].abs()

    grp = df.groupby("chapter", sort=True)

    agg = grp.agg(
        scenes=("idx", "count"),
        mean_z=("z_energy_global", "mean"),
        std_z=("z_energy_global", "std"),
        mean_abs_delta=("abs_delta_z_energy", "mean"),
    ).reset_index()

    # Replace NaN std on single-scene chapters
    agg["std_z"] = agg["std_z"].fillna(0.0)
    agg["mean_abs_delta"] = agg["mean_abs_delta"].fillna(0.0)

    # Plot 1: mean energy by chapter
    plt.figure(figsize=(12, 4))
    plt.plot(agg["chapter"], agg["mean_z"], marker="o", linewidth=2)
    plt.axhline(0.0, linewidth=1)
    plt.title("Chapter aggregates: mean z_energy_global by chapter")
    plt.xlabel("Chapter")
    plt.ylabel("Mean z_energy_global")
    out1 = outdir / "mod_04_chapter_mean_z_energy.png"
    _savefig(out1)

    # Plot 2: variability by chapter (std)
    plt.figure(figsize=(12, 4))
    plt.plot(agg["chapter"], agg["std_z"], marker="o", linewidth=2)
    plt.title("Chapter aggregates: variability (std dev) of z_energy_global by chapter")
    plt.xlabel("Chapter")
    plt.ylabel("Std dev z_energy_global")
    out2 = outdir / "mod_05_chapter_std_z_energy.png"
    _savefig(out2)

    # Plot 3: choppiness proxy (mean abs delta)
    plt.figure(figsize=(12, 4))
    plt.plot(agg["chapter"], agg["mean_abs_delta"], marker="o", linewidth=2)
    plt.title("Chapter aggregates: mean |Δ z_energy_global| within chapter (choppiness)")
    plt.xlabel("Chapter")
    plt.ylabel("Mean |Δ z_energy_global|")
    out3 = outdir / "mod_06_chapter_mean_abs_delta.png"
    _savefig(out3)

    return [
        (
            out1.name,
            "Per-chapter **mean z_energy_global**. This matches the question: “where does the book sit, on average, "
            "in each chapter?” and makes Act-level drift obvious.",
        ),
        (
            out2.name,
            "Per-chapter **std dev of z_energy_global** (variance proxy). This is the clearest way to verify your claim "
            "that chapters 10–12 ‘narrow’ (lower variance) and later chapters widen again.",
        ),
        (
            out3.name,
            "Per-chapter **mean |Δ z_energy_global|** (a choppiness / ‘frequentic’ proxy). "
            "Higher values mean more abrupt scene-to-scene energy swings inside that chapter.",
        ),
    ]


def write_summary(outdir: Path, items: List[Tuple[str, str]], title: str) -> None:
    lines = [f"# {title}", "", "Generated graphs:", ""]
    for fname, desc in items:
        lines.append(f"- **{fname}** — {desc}")
    lines.append("")
    (outdir / "SUMMARY_MOD.md").write_text("\n".join(lines), encoding="utf-8")


def parse_act(s: str, name: str) -> ActRange:
    """
    Parse "lo-hi" into ActRange(name, lo, hi).
    """
    try:
        lo_str, hi_str = s.split("-", 1)
        lo = int(lo_str.strip())
        hi = int(hi_str.strip())
        if lo <= 0 or hi <= 0 or hi < lo:
            raise ValueError
        return ActRange(name, lo, hi)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid act range '{s}'. Use like '1-5'.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to timeline.csv")
    ap.add_argument("--outdir", type=str, default="out_mod_timeline", help="Output directory")
    ap.add_argument("--prose-only", action="store_true", help="Filter to TYPE=PROSE only")
    ap.add_argument("--top-n", type=int, default=15, help="Top N transition shocks to annotate")
    ap.add_argument("--window", type=int, default=11, help="Rolling window for median/IQR envelope (odd recommended)")
    ap.add_argument(
        "--act1", type=str, default="1-5", help="Act I chapter range, e.g. 1-5"
    )
    ap.add_argument(
        "--act2", type=str, default="6-9", help="Act II chapter range, e.g. 6-9"
    )
    ap.add_argument(
        "--act3", type=str, default="10-15", help="Act III chapter range, e.g. 10-15"
    )
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    _ensure_outdir(outdir)

    acts = [
        parse_act(args.act1, "Act I"),
        parse_act(args.act2, "Act II"),
        parse_act(args.act3, "Act III"),
    ]

    df = _load_timeline(csv_path, prose_only=args.prose_only)

    # Run plots
    all_items: List[Tuple[str, str]] = []
    all_items += plot_act_windows_energy(df, outdir, acts)
    all_items += plot_transition_shocks(df, outdir, top_n=args.top_n)
    all_items += plot_rolling_envelope(df, outdir, window=args.window)
    all_items += plot_chapter_aggregates(df, outdir)

    suffix = " (PROSE ONLY)" if args.prose_only else ""
    write_summary(outdir, all_items, title=f"Modulated Timeline Plots{suffix}")

    print(f"OK — wrote {len(all_items)} modulated graphs to: {outdir}")
    print(f"Summary: {outdir / 'SUMMARY_MOD.md'}")


if __name__ == "__main__":
    main()
