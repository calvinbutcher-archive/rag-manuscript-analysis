import argparse
import urllib.request
import json
from pathlib import Path
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(prompt: str, model: str) -> str:
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        }).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
        return data["response"]

def read_timeline_summary(timeline_path):
    energies = []
    deviations = []

    with open(timeline_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                energies.append(float(r["z_energy_global"]))
                deviations.append(float(r["z_deviation"]))
            except:
                continue

    if not energies:
        return {}

    n = len(energies)
    last_slice = energies[int(n * 0.9):]

    return {
        "scene_count": n,
        "mean_energy": sum(energies) / n,
        "max_energy": max(energies),
        "min_energy": min(energies),
        "mean_deviation": sum(deviations) / n,
        "ending_mean_energy": sum(last_slice) / len(last_slice) if last_slice else None
    }

def build_prompt(appraisal: str, skeptic: str, timeline_summary: dict) -> str:

    return f"""
    You are an experienced literary editor.

    You are reading two analyses of a manuscript:

    1) A structural appraisal produced from signal analysis.
    2) A skeptical review that challenges the appraisal.

    You are also given a small summary of the manuscript's signal profile.

    TIMELINE SUMMARY
    ----------------
    {json.dumps(timeline_summary, indent=2)}

    Your task is NOT to critique the manuscript.

    Your task is to write a short editorial synthesis answering:

    "What kind of book does this appear to be?"

    Rules:

    • Do not praise the manuscript.
    • Do not critique the manuscript.
    • Do not suggest revisions.
    • Do not evaluate quality.

    Instead describe:

    - what kind of narrative object the manuscript behaves like
    - the emotional or thematic movement implied by the structure
    - how the ending behaves structurally
    - whether the manuscript resolves, collapses, stabilises, or remains open

    If evidence is uncertain, say so.

    Write in calm human editorial language.

    Length: 200–300 words.

    STRUCTURAL APPRAISAL
    -------------------
    {appraisal}

    SKEPTICAL REVIEW
    ----------------
    {skeptic}

    Write the result as:

    # Conclusion

    Followed by the synthesis.
    """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True)
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    base = Path(f"reports/{args.book}")

    timeline_path = base / "timeline.csv"
    timeline_summary = read_timeline_summary(timeline_path)

    appraisal_path = base / "APPRAISAL.md"
    skeptic_path = base / "SKEPTIC_REVIEW.md"
    out_path = base / "CONCLUSION.md"

    with open(appraisal_path) as f:
        appraisal = f.read()

    with open(skeptic_path) as f:
        skeptic = f.read()

    prompt = build_prompt(appraisal, skeptic, timeline_summary)

    print("Generating editorial conclusion...")

    result = ollama_generate(prompt, args.model)

    with open(out_path, "w") as f:
        f.write(result)

    print(f"Conclusion written to: {out_path}")


if __name__ == "__main__":
    main()
