"""Measure agent response times across a mix of text and image queries.

Each query runs fresh with no memory carried over from prior queries. Results
are saved to a CSV file for plotting and analysis.

NOTE: this script calls the Groq LLM (with fallback to Gemini).
Estimated token usage: 15,000 to 25,000 tokens.

Usage:
    python src/benchmark.py                    # saves to benchmark_results.csv
    python src/benchmark.py --out results.csv  # custom output path
"""
import argparse
import contextlib
import csv
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from load_env import load_local_env
load_local_env()

TEXT_QUERIES = [
    ("plant_lookup", "What does a sunflower look like?"),
    ("plant_lookup", "How do I care for a rose?"),
    ("plant_lookup", "What are the growing conditions for lavender?"),
    ("plant_lookup", "Is foxglove toxic to pets?"),
    ("plant_lookup", "What pollinators are attracted to a water lily?"),
    ("plant_lookup", "What is the habitat of a lotus flower?"),
    ("plant_lookup", "Describe the petal shape of a dahlia."),
    ("plant_lookup", "What pests commonly affect geraniums?"),
    ("find_similar_flowers", "What flowers have similar colors to a sunflower?"),
    ("find_similar_flowers", "What flowers grow in similar conditions to a water lily?"),
    ("find_similar_flowers", "What flowers have similar petal shapes to a rose?"),
]

SAMPLE_CLASS_IDS = ["1", "17", "34", "51", "68", "85"]


def collect_image_queries():
    """Pick one validation image from each of the sample class folders."""
    valid_dir = PROJECT_ROOT / "data" / "dataset" / "valid"
    queries = []
    for class_id in SAMPLE_CLASS_IDS:
        class_dir = valid_dir / class_id
        images = sorted(class_dir.glob("*.jpg"))
        if len(images) > 0:
            queries.append(("identify_flower_from_image", str(images[0])))
    return queries


def extract_tool_and_model(messages, prev_len):
    """Walk new messages from the agent turn and return (tool_selected, model_used)."""
    from langchain_core.messages import AIMessage
    tool_selected = "none"
    model_used = "unknown"
    for msg in messages[prev_len:]:
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None) and tool_selected == "none":
            tool_selected = msg.tool_calls[0]["name"]
        if msg.response_metadata:
            model_used = msg.response_metadata.get("model", msg.response_metadata.get("model_name", "unknown"))
    return tool_selected, model_used


def run_benchmark(out_path):
    """Run the full text + image benchmark and write results to out_path."""
    from agent import build_agent, _get_classifier
    from langchain_core.messages import HumanMessage

    print("Initializing Smart Sprout...", end="", flush=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _get_classifier()
    print(" ready.\n")

    agent = build_agent()
    image_queries = collect_image_queries()
    all_queries = TEXT_QUERIES + image_queries
    total = len(all_queries)

    print(f"Running {total} queries ({len(TEXT_QUERIES)} text, {len(image_queries)} image)...\n")

    rows = []
    for i, (expected_tool, query) in enumerate(all_queries, 1):
        query_type = "image" if expected_tool == "identify_flower_from_image" else "text"
        display = Path(query).name if query_type == "image" else query
        print(f"  [{i}/{total}] {display}")

        messages = [HumanMessage(
            content=query if query_type == "text" else f"What flower is in this image? {query}"
        )]
        prev_len = len(messages)

        t0 = time.time()
        try:
            result = agent.invoke({"messages": messages})
            elapsed = time.time() - t0
            tool_selected, model_used = extract_tool_and_model(result["messages"], prev_len)
        except Exception as e:
            elapsed = time.time() - t0
            tool_selected = "error"
            model_used = "unknown"
            print(f"    [Error: {e}]")

        print(f"    tool={tool_selected}  model={model_used}  time={elapsed:.2f}s")
        rows.append({
            "query": display,
            "query_type": query_type,
            "expected_tool": expected_tool,
            "tool_selected": tool_selected,
            "model_used": model_used,
            "response_time_s": round(elapsed, 3),
        })

    text_times = [r["response_time_s"] for r in rows if r["query_type"] == "text"]
    image_times = [r["response_time_s"] for r in rows if r["query_type"] == "image"]

    print("\n── Summary ─────────────────────────────────────────────────")
    if len(text_times) > 0:
        print(f"  Text queries:  avg {sum(text_times)/len(text_times):.2f}s  "
              f"min {min(text_times):.2f}s  max {max(text_times):.2f}s")
    if len(image_times) > 0:
        print(f"  Image queries: avg {sum(image_times)/len(image_times):.2f}s  "
              f"min {min(image_times):.2f}s  max {max(image_times):.2f}s")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmark_results.csv",
                        help="Output CSV path (default: benchmark_results.csv)")
    args = parser.parse_args()

    print("=" * 60)
    print("SMART SPROUT - RESPONSE TIME BENCHMARK")
    print("NOTE: calls Groq LLM, estimated 15,000 to 25,000 tokens")
    print("=" * 60 + "\n")

    run_benchmark(args.out)


if __name__ == "__main__":
    main()
