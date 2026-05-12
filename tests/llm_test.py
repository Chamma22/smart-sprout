"""
tests/llm_test.py

Verifies that the project can successfully call an LLM through LangChain.
Tests both Groq (used for agent text work) and Gemini (used for vision).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_env import load_local_env
load_local_env()

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def test_groq():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("Say hello and describe what seedlings are.")
    print(response.content)

def test_gemini():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("Say hello and describe what seedlings are.")
    print(response.content)

if __name__ == "__main__":
    test_groq()
    test_gemini()
