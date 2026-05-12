"""Build a RAG chain over the flower descriptions and compare RAG vs base LLM on test queries."""
import os
import random
import re
import time
from pathlib import Path

from load_env import load_local_env
load_local_env()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
RETRIEVER_K = int(os.getenv("RAG_RETRIEVER_K", "5"))
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "4"))
INITIAL_BACKOFF_SECONDS = float(os.getenv("RAG_RETRY_INITIAL_BACKOFF", "1.0"))
RUN_BASELINE = os.getenv("RAG_RUN_BASELINE", "1") == "1"

TEST_QUERIES = [
    "What does a sunflower look like?",
    "How do I care for a rose?",
    "What are the characteristics of a water lily?",
]


def load_retriever():
    """Load the persistent ChromaDB and return a retriever bound to it."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})


def invoke_with_retry(call_fn, label):
    """Retry transient LLM and API failures with exponential backoff and jitter."""
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_fn()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
            hinted_delay = float(retry_match.group(1)) if retry_match else 0.0
            sleep_for = max(backoff, hinted_delay) + random.uniform(0, 0.5)
            print(
                f"[{label}] attempt {attempt}/{MAX_RETRIES} failed: {exc}. "
                f"Retrying in {sleep_for:.1f}s..."
            )
            time.sleep(sleep_for)
            backoff *= 2


def is_quota_exhausted_error(exc):
    """Return True if the exception text looks like a provider quota error."""
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota exceeded" in text


def summarize_error(exc):
    """Flatten an exception message to a single short line for logging."""
    text = str(exc).replace("\n", " ").strip()
    return text if len(text) <= 240 else f"{text[:237]}..."


def build_rag_chain(retriever, llm):
    """Wire retriever, prompt, and LLM into a runnable RAG chain."""
    prompt = ChatPromptTemplate.from_template(
        "You are a plant assistant. Use only the provided context to answer.\n"
        "If the context does not contain the answer, say \"I don't know based on the provided context.\"\n"
        "Keep the response concise and factual.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    print("Loading retriever and LLM...")
    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    rag_chain = build_rag_chain(retriever, llm)
    quota_exhausted = False

    for query in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        if quota_exhausted:
            print("Skipping remaining model calls: API quota is currently exhausted.")
            continue

        if RUN_BASELINE:
            print("\n--- Base LLM ---")
            try:
                base_response = invoke_with_retry(lambda: llm.invoke(query), "base-llm")
                print(base_response.content)
            except Exception as exc:
                print(f"Base LLM failed after retries: {summarize_error(exc)}")
                if is_quota_exhausted_error(exc):
                    quota_exhausted = True
                    print("Detected quota exhaustion. Remaining queries will be skipped.")
                    continue

        print("\n--- RAG ---")
        try:
            rag_response = invoke_with_retry(lambda: rag_chain.invoke(query), "rag")
            print(rag_response)
        except Exception as exc:
            print(f"RAG failed after retries: {summarize_error(exc)}")
            if is_quota_exhausted_error(exc):
                quota_exhausted = True
                print("Detected quota exhaustion. Remaining queries will be skipped.")


if __name__ == "__main__":
    main()
