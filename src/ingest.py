"""Build the botanical knowledge ChromaDB from flower_descriptions.json.

Chunks each flower entry with RecursiveCharacterTextSplitter, prepends the
flower name to each chunk so retrieval keeps source context, and writes the
result to a persistent ChromaDB store.
"""
import json
import shutil
from pathlib import Path

from load_env import load_local_env
load_local_env()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "flower_descriptions.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


def load_documents() -> list[Document]:
    """Load and chunk the flower descriptions, prepending the flower name to each chunk."""
    with open(DATA_PATH) as f:
        flowers = json.load(f)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = []

    for cat_id, entry in flowers.items():
        name = entry["name"]
        content = entry.get("content", "").strip()
        url = entry.get("url", "")

        if len(content) == 0:
            print(f"[SKIPPED - no content] {name}")
            continue

        chunks = splitter.split_text(content)
        for chunk in chunks:
            docs.append(Document(
                page_content=f"{name.capitalize()}: {chunk}",
                metadata={"cat_id": cat_id, "name": name, "url": url},
            ))

    return docs


def main():
    print("Loading and chunking documents...")
    docs = load_documents()
    print(f"  {len(docs)} chunks from {DATA_PATH}")

    if CHROMA_DIR.exists():
        print(f"Clearing existing vector store at {CHROMA_DIR}/...")
        shutil.rmtree(CHROMA_DIR)

    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Storing in ChromaDB...")
    Chroma.from_documents(docs, embedding=embeddings, persist_directory=str(CHROMA_DIR))
    print(f"  Done. Vector store saved to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
