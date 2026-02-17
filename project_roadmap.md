# Agentic AI Project — Infrastructure Roadmap

This is a living document and will be updated as the architecture evolves.

**Implementation status:**
The full Smart Sprout vision includes multimodal Gemini models and Google embeddings.
The current assignment implementation is a **text‑only subset**:
- LLM: Google Gemini (text)
- Embeddings: HuggingFace (all‑MiniLM‑L6‑v2)
- No vision or image pipelines implemented yet

---

## 1. Project Overview

**Project idea:** Smart Sprout is an agentic computer vision system that identifies garden sprouts and provides guidance on what to do next (thin, wait, or replant). Because early sprouts often look nearly identical (especially in companion planting) the agent handles uncertainty by asking clarifying questions and requesting additional images when needed.

**Target domain:** Early‑stage plant images, seedling development and morphology (structure of sprouts), plant species characteristics, and optional textual descriptions of sprout traits.

**Target users:** Home gardeners, small‑scale growers, and anyone practicing companion planting.

Users may ask questions like:  
- “What sprout is this?”  
- “Should I thin or wait?”  
- “Did any of my basil come up?” 

---

## 2. LLM Provider Selection

**Our choice:**

- **Generation:** Google Gemini 2.5 Flash
- **Embeddings:** HuggingFace all‑MiniLM‑L6‑v2 
- **Vision:** Gemini native vision

**Why:**
Gemini provides a unified ecosystem for text, embeddings, and vision. As this project is multimodel due to the use of text and images using a provider who can handle all of it in one ecosystem. This simplifies development and reduces integration overhead. Flash is fast and inexpensive, which is ideal for an agent that may loop multiple times when confidence is low. The native vision model removes the need to host or fine‑tune a separate classifier. Overall, Gemini offers the best balance of cost, speed, quality, and multimodal support for this project

HuggingFace embeddings are used instead of Gemini embeddings due to current API instability in LangChain’s Google embedding wrapper.

### LangChain Integration

```python
# Generation
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
```

---

## 3. Corpus & Data Plan

**What data will your agent use?**

| Source | Format | Approx Size | Notes |
|--------|--------|-------------|-------|
| V2 Plant Seedlings Dataset | Images (PNG) | 5.5k | This dataset includes 12 seedling types. https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset?select=Maize |
| PlanVillage Dataset | Images (JPG) | 20.6k | This dataset would be used for the alternate project direction (disease detection). https://www.kaggle.com/datasets/emmarex/plantdisease |
| TreeProject Seedling Database (pending permission) | Images + metadata | TBD | High-quality seedling photos. Will collect manually or via permission-based download |

**Chunking strategy: (for text-based RAG)**
- **Chunk size:** 300–500 tokens
    Short enough to isolate specific plant traits, but long enough to capture full descriptions.
- **Chunk overlap:** 50 tokens
    Ensures continuity for morphological details.
- **Splitter:** `RecursiveCharacterTextSplitter`
    Works well for short structured botanical descriptions.


**Image data (if applicable):**
- **Description approach:**  
  Gemini Vision will be used to generate captions and extract morphological traits from seedling images (e.g., cotyledon shape, color, leaf arrangement).

- **Linking images to retrievable content:**  
    Each image will be stored with metadata (species, source, growth stage, file path).  
    Captions and extracted traits will be embedded and stored in the vector database.  
    This allows the agent to retrieve both text and image-derived information during reasoning.

- **Planned preprocessing:**
  - Resize and normalize images
  - Remove duplicates
    - Compare file hashes and hashes of the image content.
  - Tag images with growth stage (seed, cotyledon, first true leaves)
    - If the information is not given: manually tag a small subset and use the agent to infer growth stage. Then check a subset for accuracy.

### Notes on Dataset Limitations

Early-stage seedling datasets are scarce, so the plan includes:
- combining multiple public sources
- requesting permission from TreeProject
- keeping the system flexible enough to pivot to leaf-ID or disease-ID if needed

This ensures the project remains feasible even if one dataset falls through.

---

## 4. Architecture Overview

### Basic Text RAG Pipeline
```
Plant descriptions → Loaders → Chunking → Embeddings → Vector Store
                                                          ↓
User Query → Embed Query → Retrieve Top-K → Prompt + Context → LLM → Response
```

### With Agent Layer
```
User Query → Agent decides which tool to use:
    → Tool 1: Vision classifier (seedling identification)
    → Tool 2: Clarifying-question tool (ask user about expected species)
    → Tool 3: RAG retriever (optional plant trait lookup)
    → Direct LLM response (if high confidence)
```

### Multimodal Extension (if applicable)
```
Images → Gemini Vision → Descriptions/embeddings → Vector Store
    ↓
Agent retrieves both text and image-based context
```

---

## 5. Repo Structure

### Simple Structure (Good Starting Point)

```
project-name/
├── README.md                  # Setup instructions, architecture overview
├── requirements.txt
├── .env.example               # Template for API keys
├── .gitignore                 # Include .env, __pycache__, chroma_db/
├── data/
│   └── raw/                   # Your corpus files
├── src/
│   ├── config.py              # Provider setup, model selection
│   ├── ingest.py              # Loading, chunking, indexing
│   ├── retrieval.py           # Vector store, retriever
│   ├── agent.py               # Agent, tools, executor
│   └── utils.py               # Helpers
├── notebooks/
│   └── demo.ipynb             # Demonstrations and comparisons
└── docs/
    └── architecture.md
```

### Production-Style Structure (For Larger Projects or Later Refactor)

```
project-name/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── settings.yaml          # Model params, retrieval settings, chunking config
├── data/
│   ├── raw/
│   └── processed/
├── prompts/
│   ├── system.txt             # System prompt for agent
│   ├── rag_template.txt       # RAG prompt template
│   └── router_template.txt    # Router/classification prompt
├── src/
│   ├── ingest/
│   │   ├── loaders.py         # Document loading logic
│   │   └── chunking.py        # Chunking strategies
│   ├── retrieval/
│   │   ├── vectorstore.py     # ChromaDB setup and management
│   │   └── retriever.py       # Retriever configuration
│   ├── tools/
│   │   ├── rag_tool.py        # RAG retriever as agent tool
│   │   ├── calculator.py      # Example second tool
│   │   └── ...                # Additional tools
│   ├── agent/
│   │   ├── agent.py           # Agent construction
│   │   └── executor.py        # Execution and orchestration
│   └── llm/
│       └── providers.py       # Factory for Gemini/Groq/etc.
├── notebooks/
│   └── demo.ipynb
└── docs/
    └── architecture.md
```

---

## 6. Track Selection

### Track B: Image RAG via Description (Intermediate)
- Images captioned by a vision model → stored as text → retrieved as text
- Gemini: native vision makes this straightforward
- Groq: pair with Gemini or HuggingFace BLIP for the vision step
- Focus: bridging modalities, preprocessing pipelines, metadata linking

---

## 7. Development Milestones

### Milestone 1: Infrastructure Setup
- [x] GitHub repo created with chosen structure
- [x] Codespace configured with Python, dependencies
- [x] API keys stored as Codespace secrets
- [x] LLM provider connected — can make a basic call through LangChain
- [x] Vector store initialized — can add and query test embeddings

### Milestone 2: RAG Pipeline
- [ ] Corpus loaded and chunked
- [ ] Embeddings generated and stored in ChromaDB
- [ ] Retrieval chain works — queries return relevant chunks
- [ ] Side-by-side comparison: RAG vs. base LLM for 3+ queries

### Milestone 3: Agent Layer
- [ ] Retriever wrapped as LangChain tool
- [ ] At least one additional tool implemented
- [ ] Agent demonstrates tool selection with verbose trace
- [ ] Queries show agent choosing correct tool for different question types

### Milestone 4: Extensions (Stretch Goals)
- [ ] Additional tools (web search, code execution, APIs, etc.)
- [ ] Multimodal retrieval (Track B or C)
- [ ] Evaluation metrics for retrieval quality
- [ ] Production interface (Streamlit, Gradio)
- [ ] Multi-agent orchestration (LangGraph, CrewAI)
- [ ] Migration to Databricks or cloud infrastructure

---

## 8. Technical Reference

### Package Installation
```
# Base
langchain
langchain-chroma
langchain-community

# Providers
langchain-google-genai
```

### API Key Management
```python
# Store as Codespace secrets (Settings → Secrets → Codespaces)
# Access in code:
import os
key = os.environ["GOOGLE_API_KEY"]

# NEVER commit keys to the repo
# .env files go in .gitignore
```

### ChromaDB Persistence
```python
# Save to disk (persists in Codespaces between sessions)
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

# Reload
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Add chroma_db/ to .gitignore — don't commit vector stores
```

---

## 9. Future Directions

As your project matures, consider:

- **Databricks** — persistent Vector Search, MLflow experiment tracking, Unity Catalog for data governance
- **AWS Bedrock** — managed model hosting, enterprise-scale deployment
- **LangGraph** — stateful multi-agent orchestration with cycles and conditional logic
- **CrewAI** — role-based multi-agent coordination
- **LangSmith** — tracing, evaluation, and debugging for LangChain pipelines
- **Provider comparison** — same pipeline, different backends, measure quality/speed/cost
