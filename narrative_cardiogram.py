#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_cardiogram(df):
    """
    Combine core narrative signals into one cardiogram metric.
    """

    energy = df["z_energy_global"].fillna(0)
    deviation = df["z_deviation"].fillna(0)
    dialogue = df["dialogue_line_ratio"].fillna(0)

    cardiogram = (
        energy * 0.6 +
        deviation * 0.3 +
        dialogue * 2.0
    )

    return cardiogram


def plot_cardiogram(df, outdir):

    cardiogram = compute_cardiogram(df)

    x = range(len(df))

    plt.figure(figsize=(14,6))
    plt.plot(x, cardiogram, linewidth=2)

    plt.title("Narrative Cardiogram")
    plt.xlabel("Scene progression")
    plt.ylabel("Narrative intensity")

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.tight_layout()

    outfile = Path(outdir) / "narrative_cardiogram.png"
    plt.savefig(outfile)

    print(f"Wrote: {outfile}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("timeline_csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    df = pd.read_csv(args.timeline_csv)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    plot_cardiogram(df, args.outdir)


if __name__ == "__main__":
    main()
