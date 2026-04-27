# Lab 06 — Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.8-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Build, evaluate and debug RAG (Retrieval-Augmented Generation) systems using LangChain, ChromaDB and Ollama (or OpenAI).

---

## What is RAG?

Large Language Models have a **knowledge boundary** — they only know what was in their training data. RAG solves this by:

1. **Storing** your documents in a vector database
2. **Retrieving** the most relevant chunks when a question is asked
3. **Augmenting** the LLM prompt with that retrieved context
4. **Generating** a grounded answer

```
User Query
    │
    ▼
┌─────────────┐     similarity search    ┌──────────────┐
│  Embeddings │ ───────────────────────► │  Vector DB   │
│   Model     │                          │  (ChromaDB)  │
└─────────────┘                          └──────┬───────┘
                                                │ top-k chunks
                                                ▼
                                        ┌──────────────┐
                                        │  LLM Prompt  │
                                        │ (augmented)  │
                                        └──────┬───────┘
                                                │
                                                ▼
                                          Final Answer
```

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running locally **or** an OpenAI API key

### 1. Clone and enter the repo

```bash
git clone https://github.com/KevalSetu/lab-05-starter.git
cd lab-05-starter
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy the example below into a `.env` file in the project root:

```dotenv
# LLM Provider: 'local' (Ollama) or 'openai'
LLM_PROVIDER=local
LLM_MODEL=gemma3:1b

# Embeddings Provider: 'local' (Ollama) or 'openai'
EMBEDDINGS_PROVIDER=local
EMBEDDINGS_MODEL=nomic

# Only needed if using OpenAI
# OPENAI_API_KEY=sk-...
```

### 5. Pull Ollama models (local only)

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### 6. Verify setup

```bash
python test_setup.py
```

Expected output:

```
[OK] Ollama running at http://localhost:11434
[OK] LLM working: Yes, I am working...
[OK] Embeddings working: 768 dimensions
[OK] ChromaDB working
```

---

## Project Structure

```
lab-05-starter/
│
├── config/
│   └── llm_config.py              # LLM + embeddings factory (auto-loads .env)
│
├── test_setup.py                  # Environment health check
│
├── section1_knowledge_boundary.py # RAG motivation: what LLMs don't know
├── section1_updating_knowledge.py # Why RAG beats fine-tuning for freshness
│
├── section2_chunking.py           # Chunking strategies (fixed, paragraph, sentence)
├── section2_chunking1.py          # LangChain RecursiveCharacterTextSplitter + ChromaDB
├── section2_chunking3.py          # Similarity search on vector store
│
├── section3_naive_problems.py     # Naive RAG pipeline + known failure modes
│
├── section4_context_ordering.py   # Context-first vs question-first vs interleaved
├── section4_instructions.py       # Weak vs strong prompt instructions
├── section4_lost_in_middle.py     # Lost-in-the-middle attention problem
├── section4_compression.py        # Contextual compression (93% token reduction)
│
├── section5_manual_eval.py        # Manual scoring + faithfulness + retrieval eval
├── section5_faithfulness.py       # Standalone faithfulness checker
├── section5_retrieval_eval.py     # Precision & recall metrics for retrieval
├── section5_hallucination.py      # Hedging detection + contradiction checking
│
├── section6_debug_retrieval.py    # Debugging: poor retrieval, edge cases,
│                                  # multi-hop reasoning, cold start
│
├── TechRetail.py                  # Standalone missing-info test (parental leave)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running the Lab

Run each section in order:

```bash
# Section 1 — Why RAG exists
python section1_knowledge_boundary.py
python section1_updating_knowledge.py

# Section 2 — Chunking & vector store
python section2_chunking.py
python section2_chunking1.py
python section2_chunking3.py

# Section 3 — Naive RAG
python section3_naive_problems.py

# Section 4 — Context & prompt engineering
python section4_context_ordering.py
python section4_instructions.py
python section4_lost_in_middle.py
python section4_compression.py

# Section 5 — Evaluation
python section5_manual_eval.py
python section5_faithfulness.py
python section5_retrieval_eval.py
python section5_hallucination.py

# Section 6 — Debugging
python section6_debug_retrieval.py
```

---

## Key Concepts Covered

| Concept | File |
|---|---|
| Knowledge boundary | `section1_knowledge_boundary.py` |
| Chunking strategies | `section2_chunking.py` |
| Vector similarity search | `section2_chunking3.py` |
| Prompt formatting | `section4_context_ordering.py` |
| Contextual compression | `section4_compression.py` |
| Faithfulness checking | `section5_faithfulness.py` |
| Precision & recall | `section5_retrieval_eval.py` |
| Hallucination detection | `section5_hallucination.py` |
| Edge case handling | `section6_debug_retrieval.py` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | RAG orchestration |
| `langchain-ollama` | Local LLM & embeddings via Ollama |
| `langchain-openai` | OpenAI LLM & embeddings |
| `langchain-community` | ChromaDB vector store integration |
| `chromadb==1.5.8` | Vector database |
| `python-dotenv` | Environment config |
| `numpy` | Vector operations |

---

## Troubleshooting

**Ollama not running**
```bash
ollama serve
```

**Model not found**
```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

**Unicode errors on Windows**
> All print statements use ASCII-safe markers `[OK]`, `[WARN]`, `[INFO]` — no emoji.

**ChromaDB collection conflict**
> Each script uses a unique `collection_name` so re-runs don't collide.

---

## Authors

- Keval — [@KevalSetu](https://github.com/KevalSetu)

---

*Built as part of the SETU Building AI Systems module — Week 5/6.*
