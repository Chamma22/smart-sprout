"""Evaluate the vision classifier and the RAG retrieval pipeline.

Runs entirely locally. No Groq, Gemini, or other API is called.

  1. Vision classifier accuracy (top-1, top-3, throughput) on the local
     HuggingFace classifier (dima806/oxford_flowers_image_detection). Reflects
     the pre-trained model we integrated, not one we trained.

  2. RAG retrieval accuracy (per flower hit rate) by querying ChromaDB with
     local HuggingFace embeddings. Measures our pipeline design: chunking
     strategy, embedding model choice, name prepending.

Results are saved to CSV for plotting and analysis.

Usage:
    python src/evaluate.py                        # both evals, eval_vision.csv + eval_rag.csv
    python src/evaluate.py --out results          # results_vision.csv + results_rag.csv
    python src/evaluate.py --sample 100           # random sample of N images for vision eval
    python src/evaluate.py --vision-only          # skip RAG evaluation
    python src/evaluate.py --rag-only             # skip vision evaluation
"""
import argparse
import csv
import json
import os
import random
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

DATA_DIR = PROJECT_ROOT / "data"
VALID_DIR = DATA_DIR / "dataset" / "valid"
CAT_TO_NAME_PATH = DATA_DIR / "cat_to_name.json"
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VISION_MODEL_ID = "dima806/oxford_flowers_image_detection"
RETRIEVAL_K = 4

# The dima806 model was trained on the original Oxford 102 class label strings,
# which contain two oddities. cat_to_name.json was corrected so Wikipedia fetches
# would succeed; this map normalizes the model's legacy labels back to the
# corrected names so the eval doesn't flag them as wrong.
LABEL_ALIASES = {
    "barbeton daisy": "barberton daisy",
    "lotus lotus": "lotus",
}


def normalize_label(label: str) -> str:
    """Return the lower-stripped label, mapping legacy spellings to corrected ones."""
    label = label.lower().strip()
    return LABEL_ALIASES.get(label, label)


def load_cat_to_name() -> dict[str, str]:
    """Load the class-id to flower-name map from disk."""
    with open(CAT_TO_NAME_PATH) as f:
        return json.load(f)


def collect_valid_images(cat_to_name, sample=None):
    """Walk the validation set and collect (path, class_id, expected_name) tuples."""
    items = []
    for class_dir in VALID_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        expected = cat_to_name.get(class_id, "").lower()
        if len(expected) == 0:
            continue
        for img in class_dir.glob("*.jpg"):
            items.append((img, class_id, expected))
    if sample is not None:
        random.shuffle(items)
        items = items[:sample]
    return items


def evaluate_vision(cat_to_name, sample=None, out_path="eval_vision.csv"):
    """Run the vision classifier on the validation set and write per-image results."""
    print("── Vision Classifier Accuracy ──────────────────────────────")
    print("  Model: dima806/oxford_flowers_image_detection (pre-trained, not trained by us)")
    print("  What we built: two-stage pipeline, confidence threshold, Gemini fallback\n")

    from transformers import pipeline as hf_pipeline
    print("  Loading classifier...", end="", flush=True)
    classifier = hf_pipeline("image-classification", model=VISION_MODEL_ID)
    print(" ready.")

    items = collect_valid_images(cat_to_name, sample=sample)
    total = len(items)
    print(f"  Evaluating {total} images from the validation set...\n")

    top1 = top3 = 0
    rows = []

    for i, (img_path, class_id, expected) in enumerate(items, 1):
        t0 = time.time()
        results = classifier(str(img_path), top_k=3)
        elapsed = time.time() - t0

        predicted = [normalize_label(r["label"]) for r in results]
        expected_norm = normalize_label(expected)
        confidence = results[0]["score"]
        is_top1 = predicted[0] == expected_norm
        is_top3 = expected_norm in predicted

        if is_top1:
            top1 += 1
        if is_top3:
            top3 += 1

        rows.append({
            "image_path": img_path.name,
            "class_id": class_id,
            "expected_name": expected,
            "predicted_name": predicted[0],
            "top1_confidence": round(confidence, 4),
            "top1_correct": is_top1,
            "top3_correct": is_top3,
            "time_s": round(elapsed, 4),
        })

        if i % 100 == 0 or i == total:
            print(f"  {i}/{total}...")

    avg_conf = sum(r["top1_confidence"] for r in rows) / total
    total_time = sum(r["time_s"] for r in rows)

    print()
    print(f"  Top-1 accuracy:  {top1/total:.1%}  ({top1}/{total})")
    print(f"  Top-3 accuracy:  {top3/total:.1%}  ({top3}/{total})")
    print(f"  Avg confidence:  {avg_conf:.1%}")
    print(f"  Time:            {total_time:.1f}s  ({total/total_time:.1f} images/sec)")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


def evaluate_rag(cat_to_name, out_path="eval_rag.csv"):
    """Query ChromaDB for each flower and write per-flower retrieval hit results."""
    print("\n── RAG Retrieval Accuracy ──────────────────────────────────")
    print("  What we built: chunking strategy, embedding model, name-prepending")
    print("  Metric: for each flower, does the retriever surface at least one")
    print("          correctly-tagged chunk in the top-k results?\n")

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    print("  Loading vector store...", end="", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    print(" ready.")

    flowers = list(cat_to_name.values())
    total = len(flowers)
    hits = 0
    rows = []

    for name in flowers:
        query = f"What does a {name} look like?"
        t0 = time.time()
        docs = vectorstore.similarity_search(query, k=RETRIEVAL_K)
        elapsed = time.time() - t0

        retrieved_names = [doc.metadata.get("name", "").lower() for doc in docs]
        hit = name.lower() in retrieved_names
        if hit:
            hits += 1

        rows.append({
            "flower_name": name,
            "query": query,
            "retrieved_names": "|".join(retrieved_names),
            "hit": hit,
            "time_s": round(elapsed, 4),
        })

    total_time = sum(r["time_s"] for r in rows)
    print(f"\n  Flowers queried: {total}")
    print(f"  Retrieval hit rate: {hits/total:.1%}  ({hits}/{total})")
    print(f"  Time:               {total_time:.1f}s")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate a random sample of N images (vision only)")
    parser.add_argument("--out", default="eval",
                        help="Output filename prefix (default eval, gives eval_vision.csv + eval_rag.csv)")
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--rag-only", action="store_true")
    args = parser.parse_args()

    cat_to_name = load_cat_to_name()

    print("=" * 60)
    print("SMART SPROUT - EVALUATION")
    print("=" * 60)

    if not args.rag_only:
        evaluate_vision(cat_to_name, sample=args.sample, out_path=f"{args.out}_vision.csv")

    if not args.vision_only:
        evaluate_rag(cat_to_name, out_path=f"{args.out}_rag.csv")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
