# Smart Sprout

An uncertainty aware flower identification agent built for CS 675 (Fundamentals of Statistical Learning). Smart Sprout combines a local vision classifier, a RAG knowledge base, and a Reflexion style memory store so the agent can ask clarifying questions when it isn't sure and learn from user corrections without retraining.

The long term vision is sprout identification for companion planting. The current prototype is built on the Oxford 102 Flowers dataset because public seedling datasets are too small. The architecture is the same. The eventual sprout system is a dataset away.

## Origin and Motivation

Companion planting puts multiple species in the same bed. When sprouts come up in spring they look nearly identical, and the right action (thin, wait, or replant) depends on which species is which. Existing plant ID apps return a single confident answer with no reasoning. Smart Sprout demonstrates a different shape. An agent that handles uncertainty transparently and learns from being wrong, without retraining the model.

## The Dataset Pivot

The original vision was sprout identification from seedling images. Early research found that public seedling datasets are scarce. The largest available (V2 Plant Seedlings) has 5,500 images across 12 classes, which isn't enough to support the architecture this project demonstrates. The implementation pivoted to the **Oxford 102 Flowers dataset** (102 classes, train/valid/test splits) so the architecture could be built and evaluated on a realistic dataset. The long term goal of sprout identification stands. The current system is a working prototype on feasible data.

## Key Design Decisions

### LLM: Groq primary with a multi-provider fallback chain

Originally planned to use Gemini for everything. Switched to Groq (`llama-3.3-70b-versatile`) for the text agent because Gemini's free tier rate limits were hit frequently. Gemini is kept for the vision fallback because Groq doesn't support image understanding.

`get_llm()` uses LangChain's `.with_fallbacks()` to chain three models:

1. **Groq `llama-3.3-70b-versatile`.** Primary, best tool calling reliability.
2. **Groq `llama-3.1-8b-instant`.** Separate daily quota from the 70B model. Weaker tool calling, may pick the wrong tool.
3. **Gemini `gemini-2.0-flash`.** Different provider entirely, different quota pool.

The Groq free tier limit is 100k tokens/day per model. Hitting that limit on the 70B model doesn't crash the session. It degrades gracefully to the next available model.

### Embeddings: HuggingFace instead of Gemini

Gemini's embedding API had instability issues in LangChain's Google wrapper. Switched to HuggingFace `all-MiniLM-L6-v2`, which runs locally with no API quotas and is fast for RAG retrieval.

### LangGraph instead of LangChain AgentExecutor

LangChain 1.0 removed `create_tool_calling_agent` and `AgentExecutor` in favor of LangGraph's `create_react_agent`. Migrated to LangGraph. Memory is now a running message list instead of a separate `chat_history` key, and there's no separate executor object. LangGraph V1.0 later moved `create_react_agent` to `langchain.agents.create_agent` (with `system_prompt=` replacing the `prompt=` keyword). Updated accordingly.

### Corpus: Wikipedia (102 flowers)

The knowledge base is built from Wikipedia pages for each of the 102 flower classes. Data was fetched with `fetch_descriptions.py` and stored in `flower_descriptions.json`. Two typos in `cat_to_name.json` caused fetches to fail ("barbeton daisy" should be "barberton daisy", "lotus lotus" should be "lotus") and were corrected manually.

Chunks are 400 tokens with 50 token overlap. The flower name is prepended to every chunk (for example "Sunflower: ...") because retrieval returns chunks without their source context. Without the name prepended, the agent can't tell which plant a chunk is about.

### Confidence Threshold: 60%

The vision pipeline uses 60% as the cutoff between high confidence (return immediately) and low confidence (try Gemini fallback). Between 40 and 60%, the agent asks the user a clarifying question instead of burning an API call. Below 40%, the system falls back to Gemini Vision. 60% was a reasonable starting point for a 102 class classifier and can be tuned based on observed false positive and negative rates in practice.

## What's in here

- **Agent.** LangGraph ReAct agent with four tools (`identify_flower_from_image`, `plant_lookup`, `find_similar_flowers`, `record_correction`).
- **LLM fallback chain.** Groq `llama-3.3-70b-versatile`, then Groq `llama-3.1-8b-instant`, then Gemini 2.0 Flash. Each fallback is a separate quota pool.
- **Vision.** Local ViT (`dima806/oxford_flowers_image_detection`) with three band confidence routing. High returns immediately, mid asks the user a clarifying question, low falls back to Gemini Vision.
- **RAG.** ChromaDB built from Wikipedia pages for all 102 Oxford flowers, with name prepended chunks (400 tokens, 50 overlap) embedded by HuggingFace `all-MiniLM-L6-v2`.
- **Reflexion memory.** A separate `chroma_reflections/` collection storing past corrections, retrieved by similarity to the current prediction.

## Setup

### 1. Get the code and create a venv

```bash
git clone git@github.com:Chamma22/smart-sprout.git
cd smart-sprout
python -m venv .venv
source .venv/Scripts/activate    # Windows / Git Bash
# or:  source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

If `python` resolves to the wrong interpreter (common on Windows / Git Bash), call the venv's Python directly. For example, `.venv/Scripts/python <script>`.

### 2. Add API keys

You need both:
- `GROQ_API_KEY` from [console.groq.com](https://console.groq.com)
- `GOOGLE_API_KEY` from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

**Local:** create `.env/sprout.env` with:
```
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

**Codespaces:** add the keys under Settings → Secrets → Codespaces.

### 3. Build the vector store (one time)

```bash
python src/ingest.py
```

This reads `data/flower_descriptions.json`, chunks each entry, embeds with `all-MiniLM-L6-v2`, and writes the result to `chroma_db/`.

## Running

```bash
# Scripted multi turn demo (7 iterations, recommended first run)
python src/agent.py demo

# Same demo with a per-iteration timing summary
python src/agent.py demo --timing

# Interactive chat
python src/agent.py

# Identify a single image
python src/vision.py data/dataset/train/1/image_06734.jpg

# Run evaluations (vision accuracy + RAG retrieval hit rate)
python src/evaluate.py
python src/evaluate.py --vision-only --sample 100
python src/evaluate.py --rag-only

# Agent response time benchmark (calls Groq, costs tokens)
python src/benchmark.py
```

## Tests

Smoke tests for individual subsystems live in `tests/`:

```bash
python tests/llm_test.py      # Groq + Gemini connectivity
python tests/vector_test.py   # Chroma + HF embeddings
python tests/agent_test.py    # Minimal tool calling agent
```

These are connection checks, not pytest suites. Each one prints output and exits on success.

## Project Structure

```
smart-sprout/
├── README.md
├── requirements.txt
├── .env/sprout.env              # API keys (not committed)
├── data/
│   ├── cat_to_name.json         # Class IDs → flower names
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
│   ├── fetch_descriptions.py    # One-time Wikipedia fetch for cat_to_name.json
│   ├── fetch_manual.py          # Wikipedia fetch for the manual-URL backfill
│   └── load_env.py              # Loads .env/sprout.env
├── tests/                       # Smoke tests for each subsystem
├── skills/
│   └── plant-care-card/         # Anthropic-style skill (assignment 3)
├── chroma_db/                   # Botanical knowledge vector store (not committed)
└── chroma_reflections/          # Reflexion memory store (not committed)
```

All paths in source files are anchored to `Path(__file__).parent` so the project runs correctly from any working directory.

## Evaluation Results

Vision classifier (the pre-trained `dima806/oxford_flowers_image_detection` model we integrated, not a model we trained):

| Metric | Value |
|---|---|
| Top-1 accuracy | 100.0% (818/818 validation) |
| Top-3 accuracy | 100.0% |
| Avg confidence | 94.3% |
| Avg time / image | ~159 ms |

The initial eval reported 96.7%. All 27 "errors" were label string mismatches in two classes (`barberton daisy`, `lotus`) caused by typo fixes in `cat_to_name.json`. After normalizing equivalent labels in `evaluate.py`, true accuracy is 100% on this split.

RAG retrieval (our pipeline, chunking strategy + embedding model + name prepending):

| Metric | Value |
|---|---|
| Retrieval hit rate | 100% (102/102 flowers) |
| Avg query time | ~13 ms |

Agent response time on fresh Groq quota: about 1.5s per iteration end to end. See `eval_vision.csv`, `eval_rag.csv`, and `benchmark_results.csv` for the per-row results.

## Issues Encountered

- **Gemini rate limits.** Hit frequently on the free tier. Switched to Groq for text generation, kept Gemini for vision and as a final LLM fallback.
- **Gemini embedding API instability.** Switched to HuggingFace embeddings, which run locally with no API quota.
- **LangChain 1.x breaking change.** LangChain 1.0 removed `create_tool_calling_agent` and `AgentExecutor` in favor of LangGraph's `create_react_agent`, later moved to `langchain.agents.create_agent` in LangGraph V1.0. Migrated the agent code to match the new API.
- **Groq daily token limit.** The free tier caps `llama-3.3-70b-versatile` at 100k tokens/day. Extended sessions or repeated demo runs can exhaust this. Fixed with LangChain's `.with_fallbacks()` chaining Groq 70B, then Groq 8B (separate daily quota), then Gemini 2.0 Flash (different provider).
- **HuggingFace terminal noise.** Several sources of unwanted output (progress bars, BertModel load reports, unauthenticated HF Hub warnings). Suppressed via env vars (`TRANSFORMERS_VERBOSITY`, `HF_HUB_DISABLE_PROGRESS_BARS`, `HF_HUB_DISABLE_SYMLINKS_WARNING`, `HF_HUB_VERBOSITY`, `TQDM_DISABLE`) and `contextlib.redirect_stdout` around the vectorstore load.
- **Windows console encoding.** The default `cp1252` codec can't render box-drawing characters used in the timing summary banner. Fixed by reconfiguring `sys.stdout` to UTF-8 at the top of `agent.py`.
