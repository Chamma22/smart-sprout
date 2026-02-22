"""
src/agent_test.py

Demonstrates that the project can create a simple LangChain agent.
"""
from load_env import load_local_env
load_local_env()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    def get_seedling_fact(_):
        return "Seedlings are young plants that have recently germinated."

    tools = [
        Tool(
            name="seedling_fact",
            func=get_seedling_fact,
            description="Returns a simple fact about seedlings."
        )
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

    response = agent.run("Tell me a fact about seedlings.")
    print(response)

if __name__ == "__main__":
    main()