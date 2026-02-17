"""
vector_test.py

Demonstrates that the project can create and query a Chroma vector store.
"""
from load_env import load_local_env
load_local_env()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = [
        "Seedlings need light, water, and nutrients to grow.",
        "Mature plants have established root systems.",
        "Germination is the process where seeds begin to sprout."
    ]

    vectorstore = Chroma.from_texts(docs, embedding=embeddings)

    results = vectorstore.similarity_search("What do seedlings need?")
    for r in results:
        print(r.page_content)

if __name__ == "__main__":
    main()