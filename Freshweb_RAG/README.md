# FreshWeb-RAG: Adaptive, Memory-Augmented, Web-Aware RAG Framework

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-blue)](https://langchain.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-dc2626)](https://qdrant.tech)
[![Tests](https://img.shields.io/badge/Tests-115%2F115_Passing-brightgreen)](tests/)

FreshWeb-RAG is a production-oriented Retrieval-Augmented Generation backend architecture combining **Adaptive Routing**, **CRAG Corrective Evaluation**, **Live Web Intelligence**, and **Persistent Long-Term Memory (MAQR)**.

---

## 🌟 Key Capabilities

1. **Adaptive Query Routing** — Evaluates query complexity (`no_retrieval`, `single_hop`, `multi_hop`) and freshness demand independently.
2. **Corrective Retrieval (CRAG)** — Evaluates retrieved knowledge base content (`correct`, `ambiguous`, `incorrect`) before answer generation, automatically triggering web search when internal documents are insufficient or stale.
3. **Live Web Intelligence** — Real-time search via Tavily Search API with SSRF safety guards, readable webpage content extraction, and primary source prioritization.
4. **Long-Term Memory Engine** — Remembers user preferences, architectural decisions, and previous project context across 4 memory categories:
   - *Episodic*: Milestone discussions, decisions, and past problem-solving sessions.
   - *Semantic*: Technical facts, configured tools, preferences.
   - *Procedural*: Reusable workflows and development instructions.
   - *Short-Term*: Active conversational turns.
5. **Memory-Aware Query Router (MAQR)** — Detects and resolves historical references such as:
   - *"Continue my previous project."*
   - *"What model did we discuss yesterday?"*
   - *"Improve the architecture we designed earlier."*
   - *"Use the same embedding model as before."*
6. **Freshness & Reliability Scoring** — Category-specific exponential decay scoring and source authority evaluation.
7. **Strict Grounding & Faithfulness Verification** — Verifies answers against evidence, extracts claims, validates citations, and bounds regeneration loops.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
[Memory-Aware Query Router (MAQR)]
    ├── Historical Reference Detected?
    │       ├── YES ──> [Semantic Memory Retrieval] ──> [Context Enrichment]
    │       └── NO  ──> (Proceed with raw query)
    │
    ▼
[Adaptive Query Analyzer]
    ├── Complexity: no_retrieval ──────────────> [Direct LLM Answer]
    ├── Freshness Required: true (no KB) ───────> [Web Search Agent]
    └── Complexity: single_hop / multi_hop
            │
            ▼
    [Hybrid Static Retrieval (Dense + BM25 + RRF)]
            │
            ▼
    [CRAG Retrieval Evaluator]
            ├── CORRECT   ──> [Evidence Fusion]
            ├── AMBIGUOUS ──> [Web Search Agent] ──> [Evidence Fusion]
            └── INCORRECT ──> [Web Search Agent (Discard KB)] ──> [Evidence Fusion]
                                    │
                                    ▼
                    [Freshness & Reliability Scoring]
                                    │
                                    ▼
                    [Conflict Resolution Engine]
                                    │
                                    ▼
                    [Grounded Answer Synthesizer]
                                    │
                                    ▼
                    [Faithfulness & Citation Verifier]
                                    │
                            ├── PASS ──> [Deliver Response]
                            └── FAIL ──> [Single Regeneration Loop]
                                                │
                                                ▼
                                    [Async Memory Extraction & Storage]
                                                │
                                                ▼
                                              (END)
```

---

## 📦 Tech Stack

- **Framework**: FastAPI (Pydantic v2, Python 3.10+)
- **Orchestration**: LangGraph StateGraph
- **LLM**: Google Gemini (via `langchain-google-genai`) / OpenAI-compatible abstraction
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) / Google Embeddings
- **Vector DB**: Qdrant (dual collections: `documents` & `memory`)
- **Sparse Retrieval**: BM25 (`rank-bm25`) + Reciprocal Rank Fusion (RRF)
- **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **Web Intelligence**: Tavily Search API + HTTPX + BeautifulSoup4 + Trafilatura
- **Relational Storage**: PostgreSQL 16 (SQLAlchemy 2.0 Async + Alembic)
- **Caching**: Redis 7
- **Rate Limiting**: Slowapi
- **Testing**: Pytest + Pytest-Asyncio + RAGAS evaluation harness

---

## 🚀 Quick Start

### Option A: Docker Compose (Recommended)

1. Clone repository and set up environment:
```bash
git clone <repo-url>
cd Freshweb_RAG
cp .env.example .env
```

2. Add your API keys to `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

3. Launch all services:
```bash
docker-compose up --build
```

The API will be live at `http://localhost:8000`. Swagger documentation is available at `http://localhost:8000/docs`.

### Option B: Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 API Endpoints Reference

| Category | Method | Path | Description |
|---|---|---|---|
| **Health** | `GET` | `/api/v1/health` | Health check endpoint |
| | `GET` | `/api/v1/config` | Public runtime configuration |
| **Chat** | `POST` | `/api/v1/chat` | End-to-end adaptive RAG query execution |
| **Documents** | `POST` | `/api/v1/documents/upload` | Upload & index PDF, DOCX, TXT, Markdown |
| | `GET` | `/api/v1/documents` | List uploaded user documents |
| | `GET` | `/api/v1/documents/{id}` | Get document metadata & indexing status |
| | `DELETE`| `/api/v1/documents/{id}` | Delete document & Qdrant vectors |
| **Search** | `POST` | `/api/v1/search` | Direct hybrid/dense/sparse search over KB |
| **Memory** | `GET` | `/api/v1/memory` | List active memories |
| | `GET` | `/api/v1/memory/{id}` | Retrieve memory by ID |
| | `PATCH`| `/api/v1/memory/{id}` | Update memory content / importance |
| | `DELETE`| `/api/v1/memory/{id}` | Delete memory from DB & Qdrant |
| | `POST` | `/api/v1/memory/search` | Semantic vector search over memories |
| | `POST` | `/api/v1/memory/forget` | Bulk soft-delete memories |
| **MAQR** | `POST` | `/api/v1/memory-aware/analyze` | Detect historical references & intent |
| | `POST` | `/api/v1/memory-aware/resolve` | Resolve historical references to entities |
| | `POST` | `/api/v1/memory-aware/enrich` | Enrich query with user context |
| **Conversations**| `POST` | `/api/v1/conversations` | Create conversation |
| | `GET` | `/api/v1/conversations` | List user conversations |
| | `GET` | `/api/v1/conversations/{id}/history` | Get conversation message history |
| **Analytics**| `GET` | `/api/v1/analytics` | Query routes, cost, latency, & cache hits |

---

## 🧪 Testing & Evaluation

### Run Test Suite (115 Automated Tests)
```bash
pytest tests/unit/ tests/integration/ -v
```

### Run Evaluation Benchmark
```bash
python scripts/run_evaluation.py --dataset data/eval_dataset.json
```

---

## 📄 License

MIT
