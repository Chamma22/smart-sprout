"""Verify that the project can build a tool-calling agent and invoke a tool."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_env import load_local_env
load_local_env()

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


@tool
def seedling_fact(query: str) -> str:
    """Return a simple fact about seedlings."""
    return "Seedlings are young plants that have recently germinated."


def main():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    tools = [seedling_fact]
    agent = create_react_agent(llm, tools)

    result = agent.invoke({"messages": [HumanMessage(content="Tell me a fact about seedlings.")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
