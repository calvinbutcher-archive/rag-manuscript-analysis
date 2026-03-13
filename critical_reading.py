import argparse
import json
import urllib.request
from pathlib import Path

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


def load_scene_excerpt(scene_path: Path, max_chars=2000):
    try:
        with open(scene_path) as f:
            txt = f.read()
            return txt[:max_chars]
    except:
        return ""


def load_flagged_scenes(flags_path: Path, limit=15):
    with open(flags_path) as f:
        flags = json.load(f)

    scenes = []

    for s in flags[:limit]:
        scenes.append({
            "filename": s["filename"],
            "path": s["path"],
            "reason": s["reason"]
        })

    return scenes


def load_scene_texts(scene_list):
    excerpts = []

    for s in scene_list:
        path = Path(s["path"])
        text = load_scene_excerpt(path)

        excerpts.append({
            "file": s["filename"],
            "reason": s["reason"],
            "text": text
        })

    return excerpts


def build_prompt(appraisal, skeptic, conclusion, scenes, book_name):
    scene_text = ""

    for s in scenes:
        scene_text += f"""
--- SCENE: {s['file']} ({s['reason']}) ---

{s['text']}

"""

    return f"""
You are a literary critic reading a completed manuscript.

You have been given:

1. A structural appraisal based on signal analysis
2. A skeptical review of that appraisal
3. A final editorial synthesis
4. Several scenes representing structural extremes of the manuscript

Your task is NOT to critique the author.

Your task is to interpret what the manuscript appears to be about.

Focus on:

• thematic concerns
• recurring ideas
• symbolic patterns
• the emotional movement of the narrative
• the relationship between structure and theme
• what kind of story this behaves like

Do NOT:

• rewrite scenes
• suggest revisions
• evaluate quality
• give writing advice

Base your interpretation primarily on the text excerpts.

If evidence is uncertain, say so.

Write 400–600 words.

STRUCTURAL APPRAISAL
--------------------
{appraisal}

SKEPTICAL REVIEW
----------------
{skeptic}

EDITORIAL CONCLUSION
--------------------
{conclusion}

SELECTED SCENES
---------------
{scene_text}

Write the result as:

# Critical Reading: {book_name}

Followed by the interpretation.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True)
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    base = Path(f"reports/{args.book}")

    appraisal = (base / "APPRAISAL.md").read_text()
    skeptic = (base / "SKEPTIC_REVIEW.md").read_text()
    conclusion = (base / "CONCLUSION.md").read_text()

    flags_path = base / "APPRAISAL_FLAGS.json"

    flagged = load_flagged_scenes(flags_path)
    scenes = load_scene_texts(flagged)

    prompt = build_prompt(
        appraisal,
        skeptic,
        conclusion,
        scenes,
        args.book
    )

    print("Generating critical reading...")

    result = ollama_generate(prompt, args.model)

    out_path = base / "CRITICAL_READING.md"

    with open(out_path, "w") as f:
        f.write(result)

    print(f"Critical reading written to {out_path}")


if __name__ == "__main__":
    main()
