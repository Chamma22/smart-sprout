"""
src.llm_test

Demonstrates that the project can successfully call an LLM through LangChain.
This script verifies that:

    1. Local environment variables are loaded (for local development).
    2. The Google Gemini provider is correctly configured.
    3. A text generation call works end-to-end.
"""
from load_env import load_local_env
load_local_env()

from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("Say hello and describe what seedlings are.")
    print(response.content)

if __name__ == "__main__":
    main()