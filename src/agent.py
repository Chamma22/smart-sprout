"""Builds a LangGraph ReAct agent for Smart Sprout.

Tools exposed to the agent:
  - identify_flower_from_image
  - plant_lookup
  - find_similar_flowers
  - record_correction

Run modes:
  python src/agent.py        interactive chat
  python src/agent.py demo   scripted multi turn demo with memory trace
"""
import contextlib
import io
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Force UTF-8 on stdout/stderr so Unicode box-drawing and em dashes don't crash
# on Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

from load_env import load_local_env
load_local_env()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from vision import identify_flower as _identify_flower, get_classifier as _get_classifier


class ReflectionNotFound(Exception):
    """Raised when a reflection lookup expects a match and finds none."""


PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
REFLECTIONS_DIR = str(PROJECT_ROOT / "chroma_reflections")

SYSTEM_PROMPT = (
    "You are Smart Sprout, a helpful gardening assistant. "
    "You help gardeners learn about flowers and plants using a knowledge base of 102 flowers.\n\n"
    "When the user provides an image path:\n"
    "- Call identify_flower_from_image exactly once. Do not call any other tools afterward.\n"
    "- Report the result and stop. Do not look up additional information unless the user asks.\n"
    "- If the tool's response begins with [CLARIFY], you MUST end your reply with a question to the user. "
    "Do not state the top candidate as the answer. Quote both candidates with their confidence percentages and ask for distinguishing features. "
    "Example: tool returns '[CLARIFY] I'm not certain, could be a sweet pea (58%) or cyclamen (3%). Can you describe...' "
    "then your reply: 'I'm not sure, it could be a sweet pea (58%) or a cyclamen (3%). Can you describe one or two distinctive features?'\n"
    "- If the tool's response begins with [REFLECTION], a past correction has already been retrieved. State the corrected identification directly to the user. Do NOT call record_correction. The correction is already saved.\n\n"
    "When the user corrects an identification in their CURRENT message (e.g. 'no, that's actually a bleeding heart'), "
    "call record_correction with: predicted_flower = the WRONG name the model produced earlier (look back at the previous "
    "identify_flower_from_image response and use its top candidate), actual_flower = the CORRECT name the user just gave, "
    "and notes = any distinguishing features the user mentioned. "
    "Example: if a prior tool response said 'could be a sweet pea (58%) or a cyclamen (3%)' and the user replies "
    "'actually that's a bleeding heart', call record_correction(predicted_flower='sweet pea', actual_flower='bleeding heart', notes=...). "
    "Only call record_correction when the user is providing a new correction. Never when reading a past reflection back from a tool response.\n\n"
    "When answering questions about a specific flower:\n"
    "- Use plant_lookup to find relevant information.\n"
    "- If the retrieved context answers the question, use it to answer directly. "
    "Do not say you were unable to find information if you are about to provide it.\n"
    "- If the context doesn't answer the specific question but contains other information about the flower, "
    "say you don't have that specific information, share what you do know, and offer to find flowers where that information is available.\n"
    "- If the context contains nothing useful, say so and ask if you can help another way.\n\n"
    "When the user asks about flowers with similar traits, colors, shapes, or growing conditions, "
    "use find_similar_flowers.\n\n"
    "If you are unsure what the user is asking, ask a clarifying question."
)


def get_llm():
    """Wire three models into a Groq, Groq, Gemini fallback chain."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    primary = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    fallbacks = [
        ChatGroq(model="llama-3.1-8b-instant", temperature=0),
        ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
    ]
    return primary.with_fallbacks(fallbacks)


def load_vectorstore():
    """Load the botanical knowledge vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def load_reflections_store():
    """Load the Reflexion style reflections store as a separate Chroma collection."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=REFLECTIONS_DIR,
        embedding_function=embeddings,
        collection_name="reflections",
    )


with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    vectorstore = load_vectorstore()
    reflections_store = load_reflections_store()


def _retrieve_reflection(prediction_text: str) -> str:
    """Return a formatted note on the matching past correction.

    Raises ReflectionNotFound when the store has no match. The stored predicted
    name must also appear in the current prediction text; this keeps false
    matches from drifting in via embedding noise.
    """
    hits = reflections_store.similarity_search_with_score(prediction_text, k=1)
    if len(hits) == 0:
        raise ReflectionNotFound(
            f"No reflection in store for prediction {prediction_text!r}."
        )
    doc, distance = hits[0]
    predicted = (doc.metadata.get("predicted") or "").lower()
    if len(predicted) == 0 or predicted not in prediction_text.lower() or distance >= 1.0:
        raise ReflectionNotFound(
            f"Closest reflection (distance {distance:.2f}, predicted={predicted!r}) "
            f"did not match the current prediction."
        )
    actual = doc.metadata.get("actual", "unknown")
    notes = doc.metadata.get("notes", "")
    note_clause = f" Notes from that correction: {notes}" if len(notes) > 0 else ""
    return (
        f"Past correction on record: when the model predicted '{predicted}', "
        f"the actual flower was '{actual}'.{note_clause}"
    )


@tool
def identify_flower_from_image(image_path: str) -> str:
    """Identify a flower from an image file path. Returns name and key features."""
    def _waiting_message():
        print("  [Still analyzing image, local model running or Gemini fallback in progress...]")

    timer = threading.Timer(5.0, _waiting_message)
    timer.start()
    try:
        result = _identify_flower(image_path)
    except ValueError as e:
        return f"Could not identify flower: {e}"
    except Exception as e:
        return f"Image identification failed: {e}"
    finally:
        timer.cancel()

    try:
        _retrieve_reflection(result)
    except ReflectionNotFound:
        return result

    hits = reflections_store.similarity_search_with_score(result, k=1)
    actual = hits[0][0].metadata.get("actual")
    notes = hits[0][0].metadata.get("notes", "")
    note_clause = f" Distinguishing features from the prior correction: {notes}" if len(notes) > 0 else ""
    return (
        f"[REFLECTION] This is a {actual} (recalled from a past correction).{note_clause} "
        f"Do not call record_correction; the correction is already on record."
    )


@tool
def record_correction(predicted_flower: str, actual_flower: str, notes: str = "") -> str:
    """Save a reflection on a wrong identification so the agent recalls it next time.

    Parameters
    ----------
    predicted_flower : str
        The WRONG name the model produced earlier in the conversation.
        Example: if the previous identify_flower_from_image response said
        "could be a sweet pea (58%) or a cyclamen (3%)", and the user now says
        "that's actually a bleeding heart", then predicted_flower="sweet pea".
        Do NOT pass the actual flower name here. That belongs in actual_flower.
    actual_flower : str
        The CORRECT name the user provided in their current message.
    notes : str
        Short reflection on distinguishing features the user mentioned.
    """
    try:
        predicted_norm = predicted_flower.lower().strip()
        actual_norm = actual_flower.lower().strip()
        existing = reflections_store.get(where={
            "$and": [{"predicted": predicted_norm}, {"actual": actual_norm}]
        })
        if existing and len(existing.get("ids", [])) > 0:
            return f"Already on record: '{predicted_flower}' to '{actual_flower}'. No change made."

        timestamp = datetime.now().isoformat(timespec="seconds")
        content = (
            f"When the model predicted '{predicted_flower}', the actual flower was "
            f"'{actual_flower}'. {notes}".strip()
        )
        doc = Document(
            page_content=content,
            metadata={
                "predicted": predicted_norm,
                "actual": actual_norm,
                "notes": notes,
                "timestamp": timestamp,
            },
        )
        reflections_store.add_documents([doc])
        return (
            f"Got it, saved a reflection: '{predicted_flower}' was actually "
            f"'{actual_flower}'. I'll remember this next time."
        )
    except Exception as e:
        return f"Could not save correction: {e}"


@tool
def plant_lookup(query: str) -> str:
    """Look up descriptions and traits of a specific flower from the RAG corpus."""
    try:
        docs = vectorstore.similarity_search(query, k=4)
        if len(docs) == 0:
            return "No relevant plant information found."
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Plant lookup failed: {e}"


@tool
def find_similar_flowers(characteristics: str) -> str:
    """Find flowers that share characteristics like color, shape, or growing conditions."""
    try:
        docs = vectorstore.similarity_search(characteristics, k=5)
        if len(docs) == 0:
            return "No similar flowers found."
        seen = set()
        results = []
        for doc in docs:
            name = doc.metadata.get("name", "unknown")
            if name not in seen:
                seen.add(name)
                results.append(f"{name.capitalize()}: {doc.page_content[:300]}...")
        return "\n\n".join(results)
    except Exception as e:
        return f"Similarity search failed: {e}"


def build_agent():
    """Build the LangGraph agent with all four tools."""
    llm = get_llm()
    tools = [identify_flower_from_image, plant_lookup, find_similar_flowers, record_correction]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


_SAMPLE_IMAGE = str(PROJECT_ROOT / "data" / "dataset" / "train" / "1" / "image_06734.jpg")
_OOD_IMAGE = str(PROJECT_ROOT / "data" / "bh.jpg")

DEMO_QUERIES = [
    "What does a sunflower look like?",
    "Is it toxic to cats or dogs?",
    "What flowers have similar colors or petal shapes?",
    f"What flower is in this image? {_SAMPLE_IMAGE}",
    f"What flower is in this image? {_OOD_IMAGE}",
    "Actually that's a bleeding heart, the flowers hang in heart shaped clusters along an arching stem. Please remember this for next time.",
    f"Now try identifying this image again: {_OOD_IMAGE}",
]
DEMO_PAUSE_SECONDS = 10


def demo(show_timing: bool = False) -> None:
    """Run the scripted multi turn session that demonstrates the learning loop.

    Each iteration follows initial state, tool call, observation, memory update.
    The second and third queries use pronouns ("it", "similar") that only make sense
    if the agent retains memory from prior turns. That proves the state update is real.
    """
    print("Initializing Smart Sprout...", end="", flush=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _get_classifier()
    print(" ready.\n")

    agent = build_agent()
    messages = []
    timings = []
    current_model = None

    print("=" * 60)
    print("SMART SPROUT - LEARNING LOOP DEMO")
    print("=" * 60)

    last_elapsed = 0.0
    for i, query in enumerate(DEMO_QUERIES, start=1):
        if i > 1:
            wait = max(0.0, DEMO_PAUSE_SECONDS - last_elapsed)
            if wait > 0:
                print(f"\n[Pausing {wait:.1f}s before next iteration "
                      f"(prior iteration took {last_elapsed:.1f}s of the {DEMO_PAUSE_SECONDS}s budget)...]")
                time.sleep(wait)
            else:
                print(f"\n[Skipping pause, prior iteration took {last_elapsed:.1f}s, "
                      f"already past the {DEMO_PAUSE_SECONDS}s budget.]")
        print(f"\n--- Iteration {i} ---")
        print(f"Memory state entering this iteration: {len(messages)} messages in history")
        print(f"User: {query}\n")

        messages.append(HumanMessage(content=query))
        prev_len = len(messages)
        start = time.time()

        try:
            result = agent.invoke({"messages": messages})
            result_messages = result["messages"]

            for msg in result_messages[prev_len:]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args_preview = ", ".join(
                            f"{k}={str(v)[:40]!r}" for k, v in (tc.get("args") or {}).items()
                        )
                        print(f"  > Tool selected: {tc['name']}({args_preview})")

            messages = result_messages
            answer = messages[-1].content
            model_used = next(
                (m.response_metadata.get("model", m.response_metadata.get("model_name", "unknown"))
                 for m in reversed(result_messages)
                 if isinstance(m, AIMessage) and m.response_metadata),
                "unknown"
            )
        except Exception as e:
            answer = f"[Agent error: {e}]"
            model_used = current_model
            messages.pop()

        elapsed = time.time() - start
        timings.append(elapsed)
        last_elapsed = elapsed

        if current_model is None:
            print(f"Model: {model_used}")
            current_model = model_used
        elif model_used != current_model:
            print(f"Model switched: {current_model} → {model_used}")
            current_model = model_used

        print(f"\nSmart Sprout: {answer}")
        print(f"\nMemory state after update: {len(messages)} messages in history")
        print(f"Response time: {elapsed:.2f}s")
        print("-" * 60)

    if show_timing:
        print("\n── Timing Summary ──────────────────────────────────────────")
        for i, t in enumerate(timings, 1):
            print(f"  Iteration {i}: {t:.2f}s")
        print(f"  Average:     {sum(timings)/len(timings):.2f}s")
    print("\nDemo complete.")


def is_image_path(text: str) -> bool:
    """Return True if the input string looks like an existing image path."""
    path = Path(text.strip())
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path.exists()


def chat() -> None:
    """Run the interactive chat loop."""
    agent = build_agent()
    messages = []

    print("Smart Sprout is ready.")
    print("Ask a question or enter an image path to identify a flower.")
    print("Type 'exit' to end the session.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if len(user_input) == 0:
            continue

        if is_image_path(user_input):
            agent_input = f"What flower is in this image? {user_input}"
        else:
            agent_input = user_input

        messages.append(HumanMessage(content=agent_input))

        try:
            result = agent.invoke({"messages": messages})
            messages = result["messages"]
            answer = messages[-1].content
        except Exception as e:
            print(f"[Error: {e}]\n")
            messages.pop()
            continue

        print(f"\nSmart Sprout: {answer}\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo(show_timing="--timing" in sys.argv)
    else:
        chat()


if __name__ == "__main__":
    main()
