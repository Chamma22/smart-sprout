"""
vector_test.py

Demonstrates that the project can create and query a Chroma vector store.
"""
from load_env import load_local_env
load_local_env()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def main():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

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