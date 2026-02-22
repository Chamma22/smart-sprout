"""
src/llm_test.py

Demonstrates that the project can successfully call an LLM through LangChain.
This script verifies that:

    1. Local environment variables are loaded (for local development).
    2. The Google Gemini provider is correctly configured.
    3. A text generation call works end-to-end.
"""
import os
from load_env import load_local_env
load_local_env()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

def test_gemini():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("Say hello and describe what seedlings are.")
    print(response.content)

def test_openrouter(): # Doesn't currently work due to privacy settings for account the API key is from.
    llm = ChatOpenAI(
        model="gpt-oss-20b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )
    response = llm.invoke("Say hello and describe what seedlings are.")
    print(response.content)


if __name__ == "__main__":
    test_gemini()
    test_openrouter()