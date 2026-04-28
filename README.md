# Lab 06 — Retrieval-Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.2-green?logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5.8-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Build, evaluate, debug, and master RAG (Retrieval-Augmented Generation) systems
> using LangChain, ChromaDB, and Ollama — from first principles to production-quality evaluation.

---

## What is RAG?

Large Language Models have a **knowledge boundary** — they only know what was in their training data.
RAG solves this by connecting a model to *your* documents at query time:

```
User Query
    |
    v
+-------------+     similarity search    +--------------+
|  Embeddings | -----------------------> |  Vector DB   |
|   Model     |                          |  (ChromaDB)  |
+-------------+                          +------+-------+
                                                | top-k chunks
                                                v
                                        +--------------+
                                        |  LLM Prompt  |
                                        | (augmented)  |
                                        +------+-------+
                                                |
                                                v
                                          Final Answer
```

**Why RAG beats fine-tuning for freshness:**
- Fine-tuning bakes knowledge in permanently (expensive, slow to update)
- RAG swaps the document store without touching the model
- You can see exactly which chunks were retrieved — full transparency

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
|
+-- config/
|   +-- llm_config.py              # LLM + embeddings factory (auto-loads .env)
|
+-- test_setup.py                  # Environment health check
|
+-- section1_knowledge_boundary.py # RAG motivation: what LLMs don't know
+-- section1_updating_knowledge.py # Why RAG beats fine-tuning for freshness
|
+-- section2_chunking.py           # Chunking strategies (fixed, paragraph, sentence)
+-- section2_chunking1.py          # LangChain RecursiveCharacterTextSplitter + ChromaDB
+-- section2_chunking3.py          # Similarity search on vector store
|
+-- section3_naive_problems.py     # Naive RAG pipeline + known failure modes
|
+-- section4_context_ordering.py   # Context-first vs question-first vs interleaved
+-- section4_instructions.py       # Weak vs strong prompt instructions
+-- section4_lost_in_middle.py     # Lost-in-the-middle attention problem
+-- section4_compression.py        # Contextual compression (93% token reduction)
|
+-- section5_manual_eval.py        # Manual scoring + faithfulness + retrieval eval
+-- section5_faithfulness.py       # Standalone faithfulness checker
+-- section5_retrieval_eval.py     # Precision & recall metrics for retrieval
+-- section5_hallucination.py      # Hedging detection + contradiction checking
|
+-- section6_debug_retrieval.py    # Debugging: poor retrieval, edge cases,
|                                  # multi-hop reasoning, cold start
|
+-- TechRetail.py                  # Standalone missing-info test (parental leave)
|
+-- exercise7_1.py   ***           # [Exercise] Full RAG system on academic policies
+-- exercise7_2.py   ***           # [Exercise] Chunking strategy comparison
+-- exercise7_3.py   ***           # [Exercise] Prompt engineering for RAG
+-- exercise7_4.py   ***           # [Exercise] Comprehensive evaluation framework
+-- exercise7_5.py   ***           # [Exercise] Failure mode analysis (5 modes)
+-- exercise7_6.py   ***           # [Exercise] Context window optimization
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

## Running the Lab

### Sections 1-6 (Core Lab)

```bash
# Section 1 -- Why RAG exists
python section1_knowledge_boundary.py
python section1_updating_knowledge.py

# Section 2 -- Chunking & vector store
python section2_chunking.py
python section2_chunking1.py
python section2_chunking3.py

# Section 3 -- Naive RAG
python section3_naive_problems.py

# Section 4 -- Context & prompt engineering
python section4_context_ordering.py
python section4_instructions.py
python section4_lost_in_middle.py
python section4_compression.py

# Section 5 -- Evaluation
python section5_manual_eval.py
python section5_faithfulness.py
python section5_retrieval_eval.py
python section5_hallucination.py

# Section 6 -- Debugging
python section6_debug_retrieval.py
```

### Exercise 7 (Applied RAG on Academic Policies)

Each exercise builds on the same domain: **university academic regulations**.
The `exercise7_1.py` file contains the shared policy documents used throughout.

```bash
# 7.1 -- Build a complete end-to-end RAG system
python exercise7_1.py

# 7.2 -- Compare three chunking strategies on a 800+ word document
python exercise7_2.py

# 7.3 -- Engineer and score five different prompt templates
python exercise7_3.py

# 7.4 -- Run a 10-query evaluation framework (accuracy + faithfulness)
python exercise7_4.py

# 7.5 -- Intentionally break the system in five ways and document root causes
python exercise7_5.py

# 7.6 -- Compare three context window selection strategies (k=10 -> pick 3)
python exercise7_6.py
```

---

## Exercise 7 — Deep Dive

> **Domain:** University academic regulations (attendance, submission, appeals,
> integrity, exams). Five detailed policies, each 500-900 words.

### 7.1 — Complete RAG System

Builds a full pipeline from scratch:

```
academic_policies (5 docs)
        |
   paragraph chunking (600 chars, 100 overlap)
        |
   ChromaDB vector store
        |
   5 test queries  -->  k=3 retrieval  -->  LLM answer  -->  0-2 score
        |
   [RESULT] Total: X/10 | Average: Y/2.0
```

**Chunking justification:** Paragraph-based splitting preserves numbered policy
sections (e.g. "3. General requirement") as self-contained units. This avoids
the context fragmentation seen with fixed-size chunking.

---

### 7.2 — Chunking Strategy Comparison

Tests three strategies on the Attendance Policy (~800 words):

| Strategy | Chunk Size | Overlap | Coherence |
|---|---|---|---|
| Fixed-size | 300 chars | 0 | Low — breaks mid-sentence |
| Paragraph-based | 800 chars | 0 | High — full sections intact |
| Sentence groups | 500 chars | 100 | Medium — bridged splits |

Runs 5 retrieval queries against all three vectorstores and prints a side-by-side
comparison. Includes a 240-word written deliverable justifying the best strategy.

---

### 7.3 — Prompt Engineering for RAG

Pits five templates against each other on 5 academic policy queries:

```
Template 1: Basic                  -->  Context + Question + "Answer:"
Template 2: Role + Constraints     -->  Advisor role, 75-word limit, cite sections
Template 3: Question-First         -->  Query primed before context
Template 4: Citation Required      -->  Structured: Policy ref / Answer / Confidence
Template 5: Chain-of-Thought       -->  Step 1: area / Step 2: rule / Step 3: answer
```

Scored on **accuracy** (0-2), **conciseness** (0-2), and **citation** (0-1)
for each of the 5 queries. Includes a 145-word deliverable naming the winner and why.

---

### 7.4 — Evaluation Framework

10-query test set with three difficulty levels:

| Difficulty | Count | Example |
|---|---|---|
| Easy | 4 | "What is the minimum attendance?" |
| Medium | 4 | "Is self-plagiarism a violation?" |
| Hard | 2 | "What are ALL consequences of poor attendance?" |

Metrics computed for every query:
- **Accuracy score** (0-2) via LLM-as-judge
- **Faithfulness check** — does the response contradict the retrieved context?
- **Results table** with per-difficulty breakdown

---

### 7.5 — Failure Mode Analysis

Five documented breakages, each showing: what changed, example output,
root cause, and proposed fix.

```
MODE 1  Tiny chunks (50 chars)      -->  Fragmented context, incoherent answers
MODE 2  k=1 retrieval               -->  Misses related sections / exceptions
MODE 3  k=20 retrieval              -->  Noise overwhelms relevant content
MODE 4  No-instruction prompt       -->  Model ignores context, adds caveats
MODE 5  Out-of-domain query         -->  Hallucination or irrelevant retrieval
```

---

### 7.6 — Context Window Optimization

Retrieves **10 candidates** per query, then selects only **3** for the prompt
using three different selection strategies:

```
Strategy 1: Top-3 by similarity score    -- fast, reliable for simple queries
Strategy 2: Diverse (top-2 + bottom-1)  -- covers a second policy angle
Strategy 3: LLM re-ranking              -- best for complex multi-hop queries
```

Tested across 5 queries, with a comparison deliverable explaining when each
strategy wins and where each fails.

---

## Key Concepts Covered

| Concept | File |
|---|---|
| Knowledge boundary | `section1_knowledge_boundary.py` |
| Chunking strategies | `section2_chunking.py`, `exercise7_2.py` |
| Vector similarity search | `section2_chunking3.py` |
| Prompt formatting | `section4_context_ordering.py`, `exercise7_3.py` |
| Contextual compression | `section4_compression.py` |
| Faithfulness checking | `section5_faithfulness.py`, `exercise7_4.py` |
| Precision & recall | `section5_retrieval_eval.py` |
| Hallucination detection | `section5_hallucination.py`, `exercise7_5.py` |
| Edge case handling | `section6_debug_retrieval.py`, `exercise7_5.py` |
| End-to-end RAG system | `exercise7_1.py` |
| Evaluation framework | `exercise7_4.py` |
| Context window strategies | `exercise7_6.py` |

---

## RAG Quality Checklist

Before shipping a RAG system, verify each item:

```
Retrieval
  [ ] chunk_size matches typical query scope (300-800 chars for policies)
  [ ] overlap prevents splits cutting key rules in half
  [ ] k is tuned (3-5 for most cases; never k=1 or k=20)
  [ ] similarity threshold rejects out-of-domain queries

Prompt
  [ ] Model assigned a clear role
  [ ] Cite-only instruction present ("use only the documentation above")
  [ ] Explicit fallback ("if not in documentation, say so")
  [ ] Word limit or format constraint included

Evaluation
  [ ] At least 5 labelled query-answer pairs
  [ ] Accuracy scored, not just eyeballed
  [ ] Faithfulness check catches contradictions
  [ ] Out-of-domain query tested (failure mode 5)
```

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

**exercise7_2.py through 7_6.py import error**
> Run scripts from the `lab-06` root directory so `exercise7_1.py` is importable:
> ```bash
> cd lab-06
> python exercise7_2.py
> ```

**LLM-as-judge scores seem off**
> `gemma3:1b` is a small model. Scoring prompts may return inconsistent digits.
> The code clamps scores to `[0, 2]` and defaults to `1` on parse failure.
> Use a larger model (e.g. `llama3.1:8b`) in `.env` for more reliable evaluation.

---

## Authors

- Keval — [@KevalSetu](https://github.com/KevalSetu)

---

*Built as part of the SETU Building AI Systems module — Week 5/6.*
