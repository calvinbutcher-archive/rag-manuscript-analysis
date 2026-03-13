#!/usr/bin/env python3
"""
mod_timeline_plots.py

Generates *separate* diagnostic graphs from a timeline.csv (your RAG timeline export).

New graphs focus on:
- macro arc (rolling mean overlay)
- transition shock (abs Δ z_energy)
- local volatility (rolling variance)
- peak compression (peaks + peak spacing)
- dialogue vs energy coupling (dual-axis)
- type dominance shifts (cumulative by type)
- extremes highlighting (zones where z_energy_global is very high/low)

Usage:
  python mod_timeline_plots.py ./reports/timeline.csv --outdir out_mod_timeline
  python mod_timeline_plots.py ./reports/timeline.csv --outdir out_mod_timeline --rolling 7 --varwin 5

Assumptions:
- Requires: pandas, matplotlib, numpy
- Input CSV has at least:
  filename, chapter, scene, type, title, z_energy_global, dialogue_line_ratio
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _read_timeline_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["chapter", "scene", "type", "title", "z_energy_global", "dialogue_line_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"timeline.csv missing required columns: {missing}")

    # Coerce numeric
    for col in ["chapter", "scene", "z_energy_global", "dialogue_line_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Stable ordering: chapter asc, scene asc (fallback to original if nulls)
    df = df.copy()
    if df["chapter"].isna().any() or df["scene"].isna().any():
        # fallback: try filename natural-ish
        df["_order"] = np.arange(len(df))
        df = df.sort_values(["_order"], kind="mergesort").drop(columns=["_order"])
    else:
        df = df.sort_values(["chapter", "scene"], kind="mergesort")

    df = df.reset_index(drop=True)

    # X axis: scene index (1..N)
    df["idx"] = np.arange(1, len(df) + 1, dtype=int)

    return df


def _chapter_boundaries(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    """
    Returns x positions where chapter changes, and labels for midpoints.
    """
    ch = df["chapter"].fillna(-1).astype(int).to_numpy()
    change = np.where(np.diff(ch) != 0)[0] + 1  # indices where new chapter starts (0-based)
    # x positions (idx is 1-based)
    vlines = [int(df.loc[i, "idx"]) for i in change.tolist()]

    # Build labels at chapter midpoints
    labels = []
    positions = []
    unique_chapters = [c for c in pd.unique(df["chapter"]) if not pd.isna(c)]
    for c in unique_chapters:
        sub = df[df["chapter"] == c]
        if sub.empty:
            continue
        mid = int(sub["idx"].iloc[len(sub) // 2])
        positions.append(mid)
        labels.append(f"Ch {int(c)}")
    # Interleave positions/labels into a compact form (we'll return as labels only; positions used in plot)
    # We'll return labels as strings aligned to positions stored separately elsewhere if needed.
    return vlines, labels


def _add_chapter_guides(ax: plt.Axes, df: pd.DataFrame, alpha: float = 0.12) -> None:
    vlines, _ = _chapter_boundaries(df)
    for x in vlines:
        ax.axvline(x, linewidth=1, alpha=alpha)


def _save(fig: plt.Figure, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _rolling_centered(s: pd.Series, win: int, how: str) -> pd.Series:
    r = s.rolling(window=win, center=True)
    if how == "mean":
        return r.mean()
    if how == "var":
        return r.var()
    raise ValueError(how)


def _local_peaks(y: np.ndarray, min_prominence: float = 0.0) -> np.ndarray:
    """
    Simple peak finder without scipy.
    Peak if y[i-1] < y[i] >= y[i+1]. Optionally filter by "prominence" proxy.
    """
    if len(y) < 3:
        return np.array([], dtype=int)

    peaks = []
    for i in range(1, len(y) - 1):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and np.isfinite(y[i + 1]):
            if y[i] > y[i - 1] and y[i] >= y[i + 1]:
                peaks.append(i)

    peaks = np.array(peaks, dtype=int)

    if min_prominence > 0 and peaks.size:
        # crude prominence proxy: peak - max(neighbours)
        prom = y[peaks] - np.maximum(y[peaks - 1], y[peaks + 1])
        peaks = peaks[prom >= min_prominence]

    return peaks


def _write_summary(outdir: Path, items: List[Tuple[str, str]]) -> None:
    """
    items: (filename, description)
    """
    lines = ["# mod_timeline_plots summary", ""]
    for fn, desc in items:
        lines.append(f"## {fn}")
        lines.append("")
        lines.append(desc.strip())
        lines.append("")
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Plotters
# -----------------------------

def plot_macro_arc(df: pd.DataFrame, outdir: Path, rolling: int) -> Tuple[str, str]:
    x = df["idx"].to_numpy()
    y = df["z_energy_global"].to_numpy()

    y_roll = _rolling_centered(df["z_energy_global"], rolling, "mean").to_numpy()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, y, linewidth=1, alpha=0.55)
    ax.plot(x, y_roll, linewidth=3, alpha=0.85)

    _add_chapter_guides(ax, df)

    ax.set_title(f"Macro Arc: z_energy_global + rolling mean (window={rolling})")
    ax.set_xlabel("Scene index (1..N)")
    ax.set_ylabel("z_energy_global")

    fn = "01_macro_arc_z_energy.png"
    desc = (
        f"This graph shows z_energy_global across the whole novel (scene index on X). "
        f"The thin line is raw per-scene energy; the thick line is a centered rolling mean (window={rolling}) "
        f"to expose the larger ‘arc’ shape without removing spikes. Faint vertical guides indicate chapter boundaries."
    )
    _save(fig, outdir / fn)
    return fn, desc


def plot_transition_shock(df: pd.DataFrame, outdir: Path, shock_threshold: float) -> Tuple[str, str]:
    d = df.copy()
    d["delta_z_energy"] = d["z_energy_global"].diff()
    d["abs_delta_z_energy"] = d["delta_z_energy"].abs()

    x = d["idx"].to_numpy()
    y = d["abs_delta_z_energy"].to_numpy()

    hot = np.isfinite(y) & (y >= shock_threshold)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, y, linewidth=1.4, alpha=0.75)
    if hot.any():
        ax.scatter(x[hot], y[hot], s=28)

    _add_chapter_guides(ax, df)

    ax.set_title(f"Transition Shock: |Δ z_energy_global| (threshold={shock_threshold})")
    ax.set_xlabel("Scene index (1..N)")
    ax.set_ylabel("|Δ z_energy_global| (scene-to-scene)")

    fn = "02_transition_shock_abs_delta_z_energy.png"
    desc = (
        "This graph isolates *cuts* between adjacent scenes by plotting the absolute change in z_energy_global. "
        f"Points above the threshold ({shock_threshold}) are highlighted. "
        "High spikes typically indicate structural seams: hard scene switches, type flips (DOC→PROSE), sudden pacing shifts, "
        "or deliberate ‘gear changes’ in the narrative."
    )
    _save(fig, outdir / fn)
    return fn, desc


def plot_local_volatility(df: pd.DataFrame, outdir: Path, varwin: int) -> Tuple[str, str]:
    d = df.copy()
    d["z_energy_var"] = _rolling_centered(d["z_energy_global"], varwin, "var")

    x = d["idx"].to_numpy()
    y = d["z_energy_var"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, y, linewidth=1.6, alpha=0.8)

    _add_chapter_guides(ax, df)

    ax.set_title(f"Local Volatility: rolling variance of z_energy_global (window={varwin})")
    ax.set_xlabel("Scene index (1..N)")
    ax.set_ylabel("Var(z_energy_global) over local window")

    fn = "03_local_volatility_z_energy_var.png"
    desc = (
        f"This graph shows *local instability zones* by plotting the centered rolling variance of z_energy_global (window={varwin}). "
        "High bands mean the narrative is rapidly shifting energy over short spans; low bands mean a steadier rhythm. "
        "Use it to spot where the book ‘thrashes’, where it settles, and where volatility collapses into a sustained note."
    )
    _save(fig, outdir / fn)
    return fn, desc


def plot_peaks_and_spacing(df: pd.DataFrame, outdir: Path, min_peak_prominence: float) -> List[Tuple[str, str]]:
    x = df["idx"].to_numpy()
    y = df["z_energy_global"].to_numpy()

    peaks = _local_peaks(y, min_prominence=min_peak_prominence)
    out: List[Tuple[str, str]] = []

    # A) Peaks-only overlay
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, y, linewidth=1, alpha=0.35)
    if peaks.size:
        ax.scatter(x[peaks], y[peaks], s=30)
    _add_chapter_guides(ax, df)
    ax.set_title(f"Peaks: local maxima in z_energy_global (min prominence={min_peak_prominence})")
    ax.set_xlabel("Scene index (1..N)")
    ax.set_ylabel("z_energy_global")
    fn1 = "04_peaks_z_energy.png"
    desc1 = (
        "This graph marks local maxima (‘peaks’) of z_energy_global. "
        "It’s a clean way to see where the book repeatedly hits high-energy moments without being distracted by every intermediate oscillation. "
        "If peaks visually ‘staircase’ upward, you’re seeing deliberate escalation; if they cluster tightly, you’re seeing compression."
    )
    _save(fig, outdir / fn1)
    out.append((fn1, desc1))

    # B) Peak spacing (frequency compression)
    if peaks.size >= 2:
        spacing = np.diff(peaks)  # in index steps (0-based deltas)
        # Convert to scene-index spacing (same units)
        sx = x[peaks[1:]]  # align spacing to second peak position
        fig, ax = plt.subplots(figsize=(14, 4.8))
        ax.plot(sx, spacing, linewidth=1.6, alpha=0.8)
        _add_chapter_guides(ax, df)
        ax.set_title("Peak Spacing: distance between successive energy peaks (lower = tighter frequency)")
        ax.set_xlabel("Scene index (aligned to later peak)")
        ax.set_ylabel("Scenes between peaks")
        fn2 = "05_peak_spacing.png"
        desc2 = (
            "This graph measures *frequency tightening* by plotting the number of scenes between successive energy peaks. "
            "Downward trends mean peaks are arriving closer together (compression / faster cadence). "
            "Upward trends mean the narrative is giving itself more runway between climactic beats."
        )
        _save(fig, outdir / fn2)
        out.append((fn2, desc2))
    else:
        # still write something useful
        fn2 = "05_peak_spacing.png"
        (outdir / fn2).write_text("Not enough peaks to compute spacing.\n", encoding="utf-8")
        out.append((fn2, "Not enough peaks were detected to compute spacing."))

    return out


def plot_dialogue_vs_energy(df: pd.DataFrame, outdir: Path, rolling: int) -> Tuple[str, str]:
    x = df["idx"].to_numpy()
    energy = df["z_energy_global"].to_numpy()
    dlg = df["dialogue_line_ratio"].to_numpy()

    energy_roll = _rolling_centered(df["z_energy_global"], rolling, "mean").to_numpy()

    fig, ax1 = plt.subplots(figsize=(14, 4.8))
    ax2 = ax1.twinx()

    ax1.plot(x, energy, linewidth=1.0, alpha=0.35)
    ax1.plot(x, energy_roll, linewidth=2.6, alpha=0.85)
    ax2.plot(x, dlg, linewidth=1.4, alpha=0.6)

    _add_chapter_guides(ax1, df)

    ax1.set_title(f"Coupling: z_energy_global (raw + roll{rolling}) vs dialogue_line_ratio (dual axis)")
    ax1.set_xlabel("Scene index (1..N)")
    ax1.set_ylabel("z_energy_global")
    ax2.set_ylabel("dialogue_line_ratio")

    fn = "06_dialogue_vs_energy_dual_axis.png"
    desc = (
        "This dual-axis graph compares narrative energy against dialogue density. "
        f"Energy is shown raw (thin) and smoothed (rolling mean window={rolling}); dialogue_line_ratio is overlaid on the second axis. "
        "Use it to spot sections where rising energy is carried by speech (argument, coordination, confrontation) "
        "versus sections where energy spikes while dialogue collapses (overwhelm, awe, catastrophe, the ‘too big to speak’ ending)."
    )
    _save(fig, outdir / fn)
    return fn, desc


def plot_type_dominance_cumulative(df: pd.DataFrame, outdir: Path) -> Tuple[str, str]:
    d = df.copy()
    # cumulative energy *within each type* over time
    d["cum_energy"] = d.groupby("type")["z_energy_global"].cumsum()

    fig, ax = plt.subplots(figsize=(14, 5.6))
    for t in sorted(d["type"].dropna().unique().tolist()):
        sub = d[d["type"] == t]
        ax.plot(sub["idx"].to_numpy(), sub["cum_energy"].to_numpy(), linewidth=2.0, alpha=0.75, label=str(t))

    _add_chapter_guides(ax, df)

    ax.set_title("Type Dominance: cumulative z_energy_global per type over time")
    ax.set_xlabel("Scene index (1..N)")
    ax.set_ylabel("Cumulative z_energy_global (per type)")
    ax.legend(loc="best", frameon=False)

    fn = "07_type_dominance_cumulative_energy.png"
    desc = (
        "This graph shows how different scene *types* contribute to energy over time by plotting cumulative z_energy_global for each type. "
        "It’s a dominance lens: you can see where non-PROSE forms drive amplitude (memos/transcripts/reports), "
        "and where the book transitions into PROSE-led momentum (often visible as one line becoming the clear driver in later acts)."
    )
    _save(fig, outdir / fn)
    return fn, desc


def plot_extremes_highlight(df: pd.DataFrame, outdir: Path, hi: float, lo: float) -> Tuple[str, str]:
    x = df["idx"].to_numpy()
    y = df["z_energy_global"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.plot(x, y, linewidth=1.2, alpha=0.7)

    # Shade extremes
    ax.axhline(hi, linewidth=1, alpha=0.25)
    ax.axhline(lo, linewidth=1, alpha=0.25)

    high_mask = np.isfinite(y) & (y >= hi)
    low_mask = np.isfinite(y) & (y <= lo)

    if high_mask.any():
        ax.scatter(x[high_mask], y[high_mask], s=30)
    if low_mask.any():
        ax.scatter(x[low_mask], y[low_mask], s=30)

    _add_chapter_guides(ax, df)

    ax.set_title(f"Extremes: z_energy_global with high/low zones (hi≥{hi}, lo≤{lo})")
    ax.set_xlabel("Scene index (1..N)")
    ax.set_ylabel("z_energy_global")

    fn = "08_extremes_highlight_z_energy.png"
    desc = (
        "This graph highlights the *cathedral moments* (very high energy) and the *deep lulls* (very low energy) "
        f"by marking scenes above {hi} and below {lo}. "
        "It’s a fast way to locate where the book deliberately concentrates intensity or drops into procedural quiet."
    )
    _save(fig, outdir / fn)
    return fn, desc


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate modular timeline diagnostic plots (separate graphs).")
    ap.add_argument("csv", type=str, help="Path to timeline.csv")
    ap.add_argument("--outdir", type=str, default="out_mod_timeline", help="Output directory for PNGs + SUMMARY.md")

    ap.add_argument("--rolling", type=int, default=7, help="Rolling window for macro arc / coupling plots (centered)")
    ap.add_argument("--varwin", type=int, default=5, help="Rolling window for local variance (centered)")
    ap.add_argument("--shock-threshold", type=float, default=2.5, help="Highlight |Δz_energy| points above this")
    ap.add_argument("--peak-prominence", type=float, default=0.0, help="Minimum peak prominence proxy for peak detection")

    ap.add_argument("--hi", type=float, default=2.0, help="High extreme threshold for z_energy_global")
    ap.add_argument("--lo", type=float, default=-2.0, help="Low extreme threshold for z_energy_global")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    _ensure_outdir(outdir)

    df = _read_timeline_csv(csv_path)

    # Build plots
    summary_items: List[Tuple[str, str]] = []

    fn, desc = plot_macro_arc(df, outdir, rolling=args.rolling)
    summary_items.append((fn, desc))

    fn, desc = plot_transition_shock(df, outdir, shock_threshold=args.shock_threshold)
    summary_items.append((fn, desc))

    fn, desc = plot_local_volatility(df, outdir, varwin=args.varwin)
    summary_items.append((fn, desc))

    peak_items = plot_peaks_and_spacing(df, outdir, min_peak_prominence=args.peak_prominence)
    summary_items.extend(peak_items)

    fn, desc = plot_dialogue_vs_energy(df, outdir, rolling=args.rolling)
    summary_items.append((fn, desc))

    fn, desc = plot_type_dominance_cumulative(df, outdir)
    summary_items.append((fn, desc))

    fn, desc = plot_extremes_highlight(df, outdir, hi=args.hi, lo=args.lo)
    summary_items.append((fn, desc))

    _write_summary(outdir, summary_items)

    print(f"OK — wrote {len([i for i in summary_items if i[0].endswith('.png')])} graphs to: {outdir}")
    print(f"Summary: {outdir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
