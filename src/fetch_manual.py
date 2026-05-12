"""Fetch Wikipedia pages from manual_urls.json for flowers that fetch_descriptions.py missed.

Merges the results into data/flower_descriptions.json.
"""
import json
from pathlib import Path

import wikipedia

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANUAL_PATH = DATA_DIR / "manual_urls.json"
DESCRIPTIONS_PATH = DATA_DIR / "flower_descriptions.json"


def main():
    with open(MANUAL_PATH) as f:
        manual = json.load(f)

    with open(DESCRIPTIONS_PATH) as f:
        descriptions = json.load(f)

    failed = []
    for cat_id, entry in manual.items():
        flower_name = entry["name"]
        url = entry["url"].strip()

        if len(url) == 0:
            print(f"[SKIPPED - no URL] {flower_name}")
            failed.append(cat_id)
            continue

        title = url.rstrip("/").split("/wiki/")[-1].replace("_", " ")
        try:
            page = wikipedia.page(title, auto_suggest=False)
            descriptions[cat_id] = {
                "name": flower_name,
                "content": page.content,
                "url": page.url,
            }
            print(f"[OK] {flower_name}")
        except Exception as e:
            print(f"[FAILED] {flower_name}: {e}")
            failed.append(cat_id)

    with open(DESCRIPTIONS_PATH, "w") as f:
        json.dump(descriptions, f, indent=2)

    if len(failed) > 0:
        print(f"\nStill failed: {failed}")
    else:
        print("\nAll manual entries fetched successfully.")


if __name__ == "__main__":
    main()
