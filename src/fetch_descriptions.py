"""Fetch Wikipedia descriptions for every flower in cat_to_name.json.

Writes the result to data/flower_descriptions.json and reports any failures.
"""
import json
import time
from pathlib import Path

import wikipedia

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CAT_TO_NAME_PATH = DATA_DIR / "cat_to_name.json"
OUTPUT_PATH = DATA_DIR / "flower_descriptions.json"


def fetch_page(query):
    """Search Wikipedia for the flower and return the first page that mentions it."""
    results = wikipedia.search(query, results=3)
    if len(results) == 0:
        return None
    for result in results:
        try:
            page = wikipedia.page(result, auto_suggest=False)
            if query.lower() in page.title.lower() or query.lower() in page.content.lower():
                return page
        except wikipedia.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                if query.lower() in page.title.lower() or query.lower() in page.content.lower():
                    return page
            except Exception:
                continue
        except wikipedia.PageError:
            continue
    return None


def attempt_fetch(cat_id, flower_name, descriptions):
    """Try to fetch one flower's page and store its content. Return True on success."""
    try:
        page = fetch_page(flower_name)
        if page is None:
            return False
        descriptions[cat_id] = {
            "name": flower_name,
            "content": page.content,
            "url": page.url,
        }
        print(f"[OK] {flower_name}")
        return True
    except Exception:
        return False


def main():
    with open(CAT_TO_NAME_PATH) as f:
        cat_to_name = json.load(f)

    descriptions = {}
    failed = {}
    for cat_id, flower_name in cat_to_name.items():
        success = attempt_fetch(cat_id, flower_name, descriptions)
        if not success:
            failed[cat_id] = flower_name
            print(f"[FAILED] {flower_name}")
        time.sleep(0.5)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(descriptions, f, indent=2)

    if len(failed) > 0:
        print(f"\nFailed ({len(failed)} flowers):")
        for cat_id, flower_name in failed.items():
            print(f"  {cat_id}: {flower_name}")
    else:
        print("\nAll flowers fetched successfully.")


if __name__ == "__main__":
    main()
