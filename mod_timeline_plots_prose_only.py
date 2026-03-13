#!/usr/bin/env python3

"""
mod_timeline_plots_prose_only.py

Same diagnostics as mod_timeline_plots.py
BUT restricted to scenes where type == "PROSE".

This isolates pure narrative rhythm without MEMO / DOC / TRANSCRIPT interference.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def read_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = ["chapter", "scene", "type", "title", "z_energy_global", "dialogue_line_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = df[df["type"] == "PROSE"].copy()

    df["chapter"] = pd.to_numeric(df["chapter"], errors="coerce")
    df["scene"] = pd.to_numeric(df["scene"], errors="coerce")
    df["z_energy_global"] = pd.to_numeric(df["z_energy_global"], errors="coerce")
    df["dialogue_line_ratio"] = pd.to_numeric(df["dialogue_line_ratio"], errors="coerce")

    df = df.sort_values(["chapter", "scene"]).reset_index(drop=True)

    # New prose-only index
    df["prose_idx"] = np.arange(1, len(df) + 1)

    return df


def add_chapter_lines(ax, df):
    ch = df["chapter"].fillna(-1).astype(int).to_numpy()
    changes = np.where(np.diff(ch) != 0)[0] + 1
    for i in changes:
        ax.axvline(df.loc[i, "prose_idx"], alpha=0.15)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------
# Graphs
# --------------------------------------------------

def macro_arc(df, outdir, rolling):
    x = df["prose_idx"]
    y = df["z_energy_global"]

    y_roll = y.rolling(rolling, center=True).mean()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(x, y, alpha=0.5)
    ax.plot(x, y_roll, linewidth=3)
    add_chapter_lines(ax, df)

    ax.set_title(f"PROSE ONLY — z_energy_global + rolling mean ({rolling})")
    ax.set_xlabel("Prose Scene Index")
    ax.set_ylabel("z_energy_global")

    save(fig, outdir / "01_prose_macro_arc.png")


def volatility(df, outdir, varwin):
    var = df["z_energy_global"].rolling(varwin, center=True).var()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df["prose_idx"], var)
    add_chapter_lines(ax, df)

    ax.set_title(f"PROSE ONLY — Local Volatility (window={varwin})")
    ax.set_xlabel("Prose Scene Index")
    ax.set_ylabel("Rolling Variance")

    save(fig, outdir / "02_prose_volatility.png")


def transition_shock(df, outdir):
    delta = df["z_energy_global"].diff().abs()

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(df["prose_idx"], delta)
    add_chapter_lines(ax, df)

    ax.set_title("PROSE ONLY — |Δ z_energy_global|")
    ax.set_xlabel("Prose Scene Index")
    ax.set_ylabel("Absolute Delta")

    save(fig, outdir / "03_prose_transition_shock.png")


def dialogue_vs_energy(df, outdir, rolling):
    x = df["prose_idx"]
    energy = df["z_energy_global"]
    energy_roll = energy.rolling(rolling, center=True).mean()
    dlg = df["dialogue_line_ratio"]

    fig, ax1 = plt.subplots(figsize=(14,5))
    ax2 = ax1.twinx()

    ax1.plot(x, energy, alpha=0.3)
    ax1.plot(x, energy_roll, linewidth=3)
    ax2.plot(x, dlg, alpha=0.6)

    add_chapter_lines(ax1, df)

    ax1.set_title("PROSE ONLY — Energy vs Dialogue")
    ax1.set_xlabel("Prose Scene Index")
    ax1.set_ylabel("z_energy_global")
    ax2.set_ylabel("dialogue_line_ratio")

    save(fig, outdir / "04_prose_dialogue_vs_energy.png")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="out_prose_only")
    ap.add_argument("--rolling", type=int, default=7)
    ap.add_argument("--varwin", type=int, default=5)
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_and_filter(csv_path)

    print(f"Prose scenes: {len(df)}")

    macro_arc(df, outdir, args.rolling)
    volatility(df, outdir, args.varwin)
    transition_shock(df, outdir)
    dialogue_vs_energy(df, outdir, args.rolling)

    print(f"OK — wrote prose-only graphs to: {outdir}")


if __name__ == "__main__":
    main()
