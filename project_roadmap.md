# Agentic AI Project - Infrastructure Roadmap

This is a living document and will be updated as the architecture evolves.

**Implementation status: v1 complete**
The prototype is fully built and evaluated. The README documents architecture and decisions in detail.

- LLM: Groq `llama-3.3-70b-versatile` (primary) → `llama-3.1-8b-instant` → Gemini 2.0 Flash (fallback chain)
- Embeddings: HuggingFace `all-MiniLM-L6-v2` (local, no API quota)
- Vision: local ViT (`dima806/oxford_flowers_image_detection`) → Gemini Vision fallback
- Agent: LangGraph ReAct agent with 4 tools (plant_lookup, find_similar_flowers, identify_flower_from_image, record_correction)
- Reflexion memory: separate ChromaDB collection (`chroma_reflections/`) of past corrections, retrieved by similarity to the current prediction
- Memory: running message list across turns
- Vector store: ChromaDB (persisted), 102 flower Wikipedia corpus

**Evaluation results:**
- Vision classifier top-1 accuracy (pre-trained model, not ours): 100.0% (818/818 validation images, after label normalization for two corrected typos)
- RAG retrieval hit rate: 100% (102/102 flowers)
- Agent response time on fresh Groq quota: about 1.5s per iteration end to end
- Agent response time (image queries with Gemini Vision retry overhead): avg about 17.7s

---

## 1. Project Overview

**Project idea:** Smart Sprout is an agentic computer vision system that identifies garden sprouts and provides guidance on what to do next (thin, wait, or replant). Because early sprouts often look nearly identical (especially in companion planting) the agent handles uncertainty by asking clarifying questions and requesting additional images when needed.

**Target domain:** Early stage plant images, seedling development and morphology (structure of sprouts), plant species characteristics, and optional textual descriptions of sprout traits.

**Target users:** Home gardeners, small scale growers, and anyone practicing companion planting.

Users may ask questions like:  
- “What sprout is this?”  
- “Should I thin or wait?”  
- “Did any of my basil come up?” 

---

## 2. LLM Provider Selection

**Our choice:**

- **Generation:** Groq `llama-3.3-70b-versatile` primary, with a fallback chain to Groq `llama-3.1-8b-instant` and then Gemini 2.0 Flash
- **Embeddings:** HuggingFace all-MiniLM-L6-v2
- **Vision:** Local ViT (`dima806/oxford_flowers_image_detection`) primary, Gemini Vision fallback

**Why:**
Originally planned to use Gemini for everything. Gemini's free tier rate limits were hit frequently during development, so the text agent moved to Groq's Llama 3.3 70B, which has much better tool calling reliability than the smaller free tier alternatives. Gemini is kept as the third LLM in the fallback chain (different provider, independent quota) and as the vision fallback when local confidence is low.

HuggingFace embeddings are used instead of Gemini embeddings because the Google embedding wrapper in LangChain was unstable. The local MiniLM model runs offline with no API quota and is fast for retrieval.

Vision uses a local pre-trained ViT for the common case (no API call) and Gemini Vision only when local confidence is low, which avoids the rate limit problems and keeps the demo responsive.

### LangChain Integration

```python
# Generation
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vision
from transformers import pipeline
```

---

## 3. Corpus & Data Plan

**What data the agent uses:**

| Source | Format | Approx Size | Notes |
|--------|--------|-------------|-------|
| Oxford 102 Flowers Dataset | Images (JPG) | 8.2k | 102 labeled flower classes with train/valid/test splits. https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ |
| Wikipedia pages (102 flowers) | Text (JSON) | ~3 MB | One Wikipedia page per Oxford 102 class, fetched via `fetch_descriptions.py` and stored in `flower_descriptions.json`. Used for the RAG knowledge base. |

**Chunking strategy (for text-based RAG):**
- **Source:** Full Wikipedia page per flower, stored in `flower_descriptions.json`
- **Chunk size:** 400 tokens
    Short enough to isolate specific plant traits, but long enough to capture full descriptions.
- **Chunk overlap:** 50 tokens
    Ensures continuity for morphological details that span chunk boundaries.
- **Splitter:** `RecursiveCharacterTextSplitter`
    Works well for short structured botanical descriptions.
- **Flower name prepending:** Prepend the common flower name to every chunk (for example "Fire lily: ...") so the agent knows which plant a chunk refers to, even when the text doesn't mention it by name.

**Image data:**
- Image identification runs through a local ViT classifier (`dima806/oxford_flowers_image_detection`) trained on Oxford 102.
- The vision pipeline routes on top-1 confidence: high (≥60%) returns immediately, mid (40-60%) asks the user a clarifying question built from the top-2 candidates, low (<40%) falls back to Gemini Vision with retry and backoff.
- Image content is not embedded into the vector store. The store holds text descriptions only. Reflexion memory carries past corrections by text similarity to the current prediction.

### Notes on Data Cleaning
- `cat_to_name.json` contained two typos that caused Wikipedia fetches to fail: "barbeton daisy" (should be "barberton daisy") and "lotus lotus" (should be "lotus"). Both were corrected manually. A `LABEL_ALIASES` map in `evaluate.py` keeps the corrected names compatible with the model's original training labels for accuracy scoring.

### Notes on Dataset Limitations

Early stage seedling datasets are scarce. The original goal of sprout identification was infeasible on public data alone (the largest available has 5.5k images across only 12 classes). The implementation pivoted to Oxford 102 Flowers, which has 102 well labeled classes with proper train/valid/test splits. The agent architecture is the same. The dataset can be swapped when a usable seedling corpus exists.

---

## 4. Architecture Overview

### Text RAG Pipeline
```
Wikipedia descriptions (102 flowers)
  → RecursiveCharacterTextSplitter (400 tokens, 50 overlap)
  → HuggingFace embeddings (all-MiniLM-L6-v2)
  → ChromaDB (persisted to disk)

User query → embed → similarity search (top-k) → prompt + context → LLM → response
```

### Agent Layer
```
User query → LangGraph ReAct agent (Groq Llama 3.3 70B, with fallback chain) decides which tool to use:
    → identify_flower_from_image  (vision pipeline)
    → plant_lookup                (RAG retrieval over the Wikipedia corpus)
    → find_similar_flowers        (similarity search over the same store)
    → record_correction           (write to Reflexion memory)
```

### Vision Pipeline (three-band confidence routing)
```
Image → Local ViT (dima806/oxford_flowers_image_detection)
      → top-1 confidence?
            ≥60%   → return immediately
            40-60% → ask the user a clarifying question
            <40%   → Gemini Vision fallback (retry + backoff)
```

### Reflexion Memory (nonparametric)
```
User correction → record_correction writes to chroma_reflections/
Next identification → reflection retrieved by similarity to the current prediction
                    → if match, agent surfaces the past correction
```

---

## 5. Repo Structure

```
smart-sprout/
├── README.md
├── requirements.txt
├── .env/sprout.env              # API keys (not committed)
├── data/
│   ├── cat_to_name.json         # Class IDs to flower names
│   ├── flower_descriptions.json # Wikipedia corpus
│   ├── bh.jpg                   # Out-of-distribution demo image (bleeding heart)
│   └── dataset/                 # Oxford 102 Flowers (train/valid/test splits)
├── src/
│   ├── agent.py                 # LangGraph agent, tools, Reflexion store, chat/demo modes
│   ├── vision.py                # Three-band image identification pipeline
│   ├── ingest.py                # Builds chroma_db from flower_descriptions.json
│   ├── retrieval.py             # RAG chain demo (base LLM vs. RAG comparison)
│   ├── evaluate.py              # Vision accuracy + RAG retrieval hit rate
│   ├── benchmark.py             # Agent response-time benchmark
│   ├── analyze_failures.py      # Failure-pattern analysis from eval_vision.csv
│   ├── fetch_descriptions.py    # Wikipedia fetch for cat_to_name.json
│   ├── fetch_manual.py          # Wikipedia fetch for the manual-URL backfill
│   └── load_env.py              # Loads .env/sprout.env
├── tests/                       # Smoke tests for each subsystem
├── skills/
│   └── plant-care-card/         # Anthropic-style skill (assignment 3)
├── chroma_db/                   # Botanical knowledge vector store (not committed)
└── chroma_reflections/          # Reflexion memory store (not committed)
```

All paths in source files are anchored to `Path(__file__).parent` so the project runs correctly from any working directory.

---

## 6. Track Selection

### Track B: Image RAG via Description (Intermediate)
- Images go through a local ViT classifier (`dima806/oxford_flowers_image_detection`) for primary identification.
- Gemini Vision serves as a fallback for low confidence cases (with retry and backoff).
- Text descriptions live in a separate Wikipedia derived RAG store and are queried by `plant_lookup` and `find_similar_flowers`.
- Reflexion memory stores past corrections as text in a separate ChromaDB collection, retrieved by similarity to the current prediction.
- Focus: bridging modalities through tool selection rather than direct image embedding, handling uncertainty explicitly, and storing nonparametric learning signals across sessions.

---

## 7. Development Milestones

### Milestone 1: Infrastructure Setup
- [x] GitHub repo created with chosen structure
- [x] Codespace configured with Python, dependencies
- [x] API keys stored as Codespace secrets
- [x] LLM provider connected - can make a basic call through LangChain
- [x] Vector store initialized - can add and query test embeddings

### Milestone 2: RAG Pipeline
- [x] Corpus loaded and chunked
- [x] Embeddings generated and stored in ChromaDB
- [x] Retrieval chain works - queries return relevant chunks
- [x] Side by side comparison: RAG vs. base LLM (`retrieval.py`)

### Milestone 3: Agent Layer
- [x] Retriever wrapped as LangChain tool (`plant_lookup`)
- [x] Three additional tools implemented (`find_similar_flowers`, `identify_flower_from_image`, `record_correction`)
- [x] Agent demonstrates tool selection with verbose trace (demo mode)
- [x] Queries show agent choosing correct tool for different question types
- [x] Multi turn memory proven via queries in the scripted demo (`python src/agent.py demo`) that use pronouns whose meaning depends on prior turns. Iteration 1 asks about sunflowers, then iteration 2 asks "Is **it** toxic to cats or dogs?" which only resolves correctly if the agent still has the sunflower context from iteration 1. Iteration 3 then asks about "**similar** colors or petal shapes," which only makes sense given the previous answers.
- [x] Reflexion loop proven by the correction-then-reidentify sequence in the same demo. Iteration 5 shows the agent an out of distribution image (a bleeding heart, not one of the 102 Oxford classes). Iteration 6 is the user's correction, which writes a reflection. Iteration 7 re-identifies the same image and the agent surfaces the stored correction.

### Milestone 4: Extensions (Stretch Goals)
- [x] Multimodal retrieval - two stage vision pipeline (local ViT + Gemini Vision fallback)
- [x] Evaluation metrics - `evaluate.py` (vision accuracy + RAG retrieval hit rate), `benchmark.py` (response times)
- [x] Reflexion memory - separate ChromaDB collection (`chroma_reflections/`) for nonparametric learning from user corrections
- [x] Multi provider LLM resilience - `.with_fallbacks()` chains Groq 70B, Groq 8B, and Gemini 2.0 Flash
- [x] Failure analysis - `analyze_failures.py` over `eval_vision.csv`
- [x] Anthropic style skill - `skills/plant-care-card/` (assignment 3)
- [ ] Additional tools (web search, code execution, APIs)
- [ ] Production interface (Streamlit, Gradio)
- [ ] Self critique reflections (agent reflects on its own low confidence runs)
- [ ] Image embedding based reflection retrieval (recall by similar image, not similar prediction text)
- [ ] LangSmith tracing for full observability into agent reasoning

---

## 8. Technical Reference

### Package Installation
See `requirements.txt` for exact pins. The key packages:

```
# Core LangChain stack (1.x)
langchain
langchain-core
langchain-text-splitters
langchain-chroma

# Providers
langchain-groq
langchain-google-genai
langchain-huggingface

# Agent framework
langgraph

# Embeddings and vision
sentence-transformers
transformers

# Vector store and helpers
chromadb
python-dotenv
Pillow
wikipedia
```

### API Key Management
```python
# Store as Codespace secrets (Settings → Secrets → Codespaces)
# Locally, .env/sprout.env is loaded by src/load_env.py
import os
groq_key = os.environ["GROQ_API_KEY"]
google_key = os.environ["GOOGLE_API_KEY"]

# NEVER commit keys to the repo
# .env/ and .env/sprout.env are in .gitignore
```

### ChromaDB Persistence
```python
# Save to disk (persists between sessions)
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Reload
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# chroma_db/ and chroma_reflections/ are gitignored - don't commit vector stores
```

### LLM Fallback Chain
```python
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

primary = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
fallbacks = [
    ChatGroq(model="llama-3.1-8b-instant", temperature=0),
    ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
]
llm = primary.with_fallbacks(fallbacks)
```

---

## 9. Future Directions

As the project matures, consider:

- **Sprout dataset** - collect or curate seedling images to revisit the original goal. The agent architecture is the same; only the dataset and classifier change.
- **Self-critique reflections** - extend Reflexion so the agent writes its own reflections after low-confidence runs, not just user-triggered ones.
- **Image-embedding-based reflection retrieval** - store image embeddings alongside text reflections so similar images (not just similar prediction text) trigger recall.
- **Fine-tuned classifier** - train a custom Oxford 102 classifier (or open-vocabulary variant like CLIP) to remove the API dependency for the low-confidence path.
- **Streamlit or Gradio UI** to replace the CLI for end users.
- **LangSmith tracing** - full observability into agent reasoning and tool selection for debugging.
