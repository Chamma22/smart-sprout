# Smart Sprout

Smart Sprout is an agentic AI system designed to help gardeners understand early‑stage plant growth. The long‑term vision is a multimodal agent that identifies sprouts from images, asks clarifying questions when uncertain, and retrieves botanical knowledge to guide decisions like thinning, replanting, or waiting.

The system will eventually combine:
- **Gemini Vision** for sprout identification
- **RAG pipelines** for plant trait lookup
- **Clarifying‑question logic** for uncertainty handling
- **Agentic decision‑making** to choose between tools

Example questions Smart Sprout aims to answer:
- “What sprout is this?”
- “Should I thin or wait?”
- “Did any of my basil come up?”

## Current Implementation Status (v1)
This repository contains the initial infrastructure for Smart Sprout:
- **LLM:** Google Gemini (text‑only)
- **Embeddings:** HuggingFace all‑MiniLM‑L6‑v2
- **Vector Store:** Chroma
- **Agent:** Simple retrieval‑augmented agent

Vision, multimodal retrieval, and full agent orchestration will be added in later milestones.

## Setup Instructions

### 1. Open in Codespaces or locally

#### For Codespaces
Click the green "Code" button → "Codespaces" → "Create codespace on main"

#### For local
Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Add API Keys

Get API keys:
- `GOOGLE_API_KEY` (get from  https://aistudio.google.com/app/apikey)

#### For Codespaces
- Go to your GitHub Settings → Codespaces → Secrets.
- Add API keys.

#### For local
Create `.env/sprout.env`:
GOOGLE_API_KEY=your_key_here

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
# Test LLM connection
python src/test_llm.py

# Test vector store
python src/test_vectorstore.py

# Test agent
python src/test_agent.py
```

## Provider Choice

**LLM:** Google Gemini (2.5 Flash) 
**Why:** 
- Strong reasoning performance
- Native multimodal support (future vision pipeline)
- Fast and inexpensive for iterative agent loops

**Embeddings:** HuggingFace (all-MiniLM-L6-v2)  
**Why:** 
- Local, fast, and stable
- No API quotas
- Ideal for early RAG development

**Vision:** (Future)
- Gemini Vision for sprout identification
- Image‑derived metadata stored in vector DB
- Clarifying‑question loop when confidence is low

## Project Structure
```
smart-sprout/
├── README.md
├── requirements.txt
├── .env/
│   └── sprout.env
└── src/
    ├── llm_test.py        # Proves LLM connection works
    ├── vector_test.py     # Proves Chroma + HF embeddings work
    ├── agent_test.py      # Simple retrieval-augmented agent
    └── load_env.py        # Loads API keys from .env
```

## Roadmap
Contained in project_roadmap.md

## Issues Encountered
- **Gemini embedding API instability:** Switched to HuggingFace for embeddings instead of Gemini to avoid API quota issues.