# FreshWeb-RAG: An Adaptive Web-Aware RAG Framework for Fresh & Reliable QA

## 1. Design Goals

FreshWeb-RAG needs to solve three problems that plain RAG doesn't handle well:

1. **Freshness** — static vector stores go stale; some questions need today's data, others don't.
2. **Reliability** — retrieved evidence (especially from the open web) can be wrong, contradictory, or low-quality, so it must be graded and corrected, not blindly trusted.
3. **Efficiency** — you don't want to hit the web (or even the retriever) for every query; routing should be adaptive to query complexity and staleness risk.

The architecture below borrows the three ideas with the best evidence behind them right now:

- **Adaptive-RAG**'s complexity-based routing (some queries need no retrieval, some need one-shot retrieval, some need iterative multi-hop retrieval).
- **CRAG (Corrective RAG)**'s lightweight retrieval evaluator that grades documents as Correct / Ambiguous / Incorrect and triggers web search as a corrective action when local knowledge is insufficient.
- **Agentic RAG**'s pattern of falling back to live web search automatically when the knowledge base can't answer, as seen in production pipelines like Haystack's agentic setups.

---

## 2. High-Level Architecture

```
                         ┌─────────────────────┐
   User Query  ────────▶ │  Query Analyzer      │
                         │  - complexity class   │
                         │  - freshness intent    │
                         │  - decomposition       │
                         └──────────┬───────────┘
                                    │
                     ┌──────────────┼───────────────┐
                     ▼              ▼               ▼
              [NO_RETRIEVAL]  [STATIC_RETRIEVAL] [WEB_REQUIRED]
                     │              │               │
                     │              ▼               │
                     │      ┌───────────────┐        │
                     │      │ Vector Store   │        │
                     │      │ Retriever (KB) │        │
                     │      └───────┬───────┘        │
                     │              ▼                │
                     │      ┌───────────────┐        │
                     │      │ Retrieval      │        │
                     │      │ Evaluator (CRAG)│───┐   │
                     │      └───────┬───────┘   │   │
                     │        Correct│  Ambiguous/  │
                     │              │   Incorrect  │
                     │              │        └──────┼───┐
                     │              ▼               ▼   ▼
                     │      ┌─────────────────────────────┐
                     │      │     Web Retrieval Agent       │
                     │      │  - query rewriting            │
                     │      │  - search API + fetch          │
                     │      │  - domain/date filtering        │
                     │      └──────────────┬───────────────┘
                     │                     ▼
                     │           ┌────────────────────┐
                     │           │ Freshness & Source   │
                     │           │ Reliability Scorer    │
                     │           └──────────┬─────────┘
                     │                      ▼
                     │           ┌────────────────────┐
                     │           │ Evidence Fusion &    │
                     │           │ Conflict Resolution   │
                     └──────────▶│  (KB + Web merge)     │
                                 └──────────┬─────────┘
                                            ▼
                                 ┌────────────────────┐
                                 │ Answer Synthesizer   │
                                 │ (grounded, cited)     │
                                 └──────────┬─────────┘
                                            ▼
                                 ┌────────────────────┐
                                 │ Faithfulness /        │
                                 │ Hallucination Checker │
                                 └──────────┬─────────┘
                                            ▼
                                        Final Answer
```

---

## 3. Core Components

### 3.1 Query Analyzer (router)

Classifies each incoming query along two independent axes before anything else runs:

- **Complexity**: `no_retrieval` (general knowledge, arithmetic) → `single_hop` (one fact lookup) → `multi_hop` (needs decomposition into sub-questions).
- **Freshness intent**: does the query contain or imply time-sensitivity? Signals: explicit temporal words ("today", "latest", "current", "as of 2026"), named entities whose state changes (prices, scores, holders of office, versions), or a topic your KB is known to be stale on.

Implementation: a small fine-tuned classifier (or a cheap LLM call with structured output) is enough — this doesn't need to be a large model. Adaptive-RAG's original approach trains a tiny classifier on synthetic labels derived from whether static retrieval alone answered benchmark questions correctly; you can replicate that cheaply.

```python
class QueryAnalysis(BaseModel):
    complexity: Literal["no_retrieval", "single_hop", "multi_hop"]
    freshness_required: bool
    freshness_reason: str
    sub_questions: list[str]  # empty unless multi_hop
```

### 3.2 Static Retriever (KB path)

Standard hybrid retrieval over your indexed corpus:

- Dense embeddings (e.g., a modern embedding model) + BM25/sparse, fused with reciprocal rank fusion.
- Chunking with overlap, metadata filters (source, date indexed, domain).
- Store a `last_verified_at` timestamp per chunk — this is what feeds the freshness scorer later.

### 3.3 Retrieval Evaluator (the CRAG piece)

This is the component that makes the system "corrective" rather than just "adaptive." After the static retriever returns top-k chunks, a lightweight evaluator (small LLM call or trained relevance model) scores the *set* of retrieved documents against the query:

- **Correct** → confidence is high, proceed to synthesis using KB evidence only.
- **Ambiguous** → keep the KB evidence but *supplement* with a web search.
- **Incorrect** → discard KB evidence, go straight to web search.

```python
class RetrievalGrade(BaseModel):
    verdict: Literal["correct", "ambiguous", "incorrect"]
    confidence: float
    missing_aspects: list[str]  # what the query needs that isn't covered
```

### 3.4 Web Retrieval Agent

Triggered when `freshness_required=True`, or when the evaluator returns `ambiguous`/`incorrect`. Responsibilities:

1. **Query rewriting** — turn the natural-language question into 1–3 concrete search queries (short, specific, reformulated if the first pass returns nothing useful — this mirrors how a careful researcher iterates).
2. **Search + fetch** — call a search API, then fetch full page content for the top results rather than trusting snippets alone.
3. **Recency filtering** — prefer sources published/updated close to "now"; down-rank or discard stale pages when the query is explicitly time-sensitive.
4. **Domain quality filtering** — maintain an allow/deny list logic: prioritize primary sources (official docs, government/.gov, company blogs, peer-reviewed venues) over aggregators and SEO content farms.

### 3.5 Freshness & Source Reliability Scorer

Every piece of evidence — KB or web — gets two scores before fusion:

| Signal | KB chunk | Web page |
|---|---|---|
| Recency | `last_verified_at` age | publish/last-modified date |
| Source authority | curated at ingest time | domain reputation heuristic (primary source > news > blog > forum) |
| Corroboration | cross-chunk agreement | agreement across ≥2 independent web sources |
| Specificity | contains concrete facts vs. vague | same |

Combine into a single `reliability_score` per evidence item using a weighted sum or a small learned ranker — this becomes the tiebreaker in the fusion step.

### 3.6 Evidence Fusion & Conflict Resolution

When KB and web evidence disagree (e.g., KB says a role-holder is X, web says Y):

- Prefer higher-recency evidence when the fact is inherently time-varying (prices, office-holders, versions).
- Prefer higher-authority evidence when the fact is stable but contested (scientific claims, historical facts).
- If genuinely unresolved, surface **both** with attribution rather than silently picking one — this is a reliability requirement, not an edge case to special-case away.

### 3.7 Answer Synthesizer

Standard grounded generation, but with hard constraints:

- Every factual claim must trace back to a specific evidence chunk/page — track this with inline citation markers during generation, not after.
- If no evidence supports a claim, the model must say so rather than filling the gap from parametric memory. Prompt this explicitly: *"If the retrieved evidence does not contain the answer, say what you don't know rather than guessing."*

### 3.8 Faithfulness / Hallucination Checker

A second-pass verification step (can reuse a cheap LLM call or NLI-style entailment model): for each claim in the draft answer, check whether it's entailed by the cited evidence. Flag/remove unsupported claims before returning the final answer. This is the same idea as Self-RAG's critique tokens, implemented as an explicit post-hoc pass instead of requiring a specially fine-tuned model.

---

## 4. Suggested Tech Stack

| Layer | Options |
|---|---|
| Orchestration | LangGraph or a custom state machine (recommended — routing logic is exactly a graph with conditional edges) |
| Vector store | Qdrant, Weaviate, or pgvector |
| Embeddings | Any current strong embedding model (e.g., `voyage-3`, `text-embedding-3-large`, or an open model like `bge-large`) |
| Sparse retrieval | BM25 via Elasticsearch/OpenSearch, or `rank_bm25` for smaller corpora |
| Web search | A search API (Brave Search API, Tavily, SerpAPI, or Bing) — Tavily is purpose-built for RAG-style agent search |
| LLM(s) | A capable model for synthesis + a cheaper/smaller model for the router, evaluator, and verifier (don't use your biggest model for classification tasks) |
| Evaluation | RAGAS or TruLens for faithfulness/relevance/answer-correctness metrics |
| Caching | Redis for query→evidence caching with short TTLs for freshness-sensitive queries |

---

## 5. Implementation Phases

### Phase 1 — Baseline RAG
Get a plain hybrid-retrieval RAG pipeline working end-to-end over your static corpus. Ship this first; it's your fallback path and your evaluation baseline.

### Phase 2 — Query Router + Retrieval Evaluator
Add the complexity/freshness classifier and the CRAG-style grading step. At this point the system can decide "I don't trust what I found" — even without web access yet, this alone reduces hallucination by refusing to answer confidently on bad evidence.

### Phase 3 — Web Retrieval Agent
Wire in the search API, fetch + parse pipeline, and recency/domain filtering. Add the fusion/conflict-resolution logic.

### Phase 4 — Verification Layer
Add the faithfulness checker as a gate before returning answers. Log flagged claims for later analysis — this data tells you where your retrieval or fusion logic is weakest.

### Phase 5 — Caching & Cost Control
Cache web search results with TTLs proportional to how fast the topic changes (minutes for stock prices, weeks for slower-moving facts). Cache the router/evaluator decisions per query pattern where sensible.

### Phase 6 — Evaluation Harness
Build a benchmark set with three query types: (a) stable factual questions your KB should nail, (b) genuinely time-sensitive questions requiring web search, (c) trick questions where your KB is stale and only web search gives the right answer. Track: answer correctness, faithfulness (RAGAS), retrieval precision/recall, and latency/cost per query type.

---

## 6. Evaluation Metrics to Track

- **Answer correctness** against a labeled QA set, split by query type (static / fresh / multi-hop).
- **Faithfulness** — fraction of claims entailed by cited evidence (RAGAS `faithfulness` or a custom NLI pass).
- **Retrieval precision/recall** at the evaluator stage — is CRAG's grading well-calibrated? (Are "correct" verdicts actually correct?)
- **Freshness accuracy** — for a held-out set of known time-sensitive questions, does the router correctly trigger web search?
- **Cost/latency per query class** — this is where adaptive routing pays for itself; measure it explicitly so you can tune the router's thresholds.

---

## 7. Key Failure Modes to Design Against

- **Router false negatives**: query looks stable but isn't (e.g., "who is the CEO of X" for a company mid-transition). Mitigate with a periodic staleness audit of your KB's "stable" facts, and bias the router toward web search when confidence is borderline.
- **Web evidence poisoning**: SEO spam or coordinated misinformation ranking highly. Mitigate with corroboration requirements (≥2 independent sources) for any web-sourced claim used without KB backing.
- **Silent conflict resolution**: picking one of two contradicting sources without telling the user. Always surface disagreement when the fusion step can't confidently resolve it.
- **Cost blowup**: naive agentic loops re-searching indefinitely. Cap iterations (e.g., 2 search rounds max) and always have a "best effort with caveats" exit path.

---

## 8. Suggested Repo Structure

```
freshweb_rag/
├── router/            # query analyzer + complexity/freshness classifier
├── retrieval/
│   ├── static/        # vector store + hybrid retriever
│   ├── evaluator/      # CRAG-style grading
│   └── web/           # search agent, fetch, parsing, recency filter
├── scoring/           # reliability & freshness scoring
├── fusion/            # evidence merge + conflict resolution
├── synthesis/         # answer generation with citation tracking
├── verification/      # faithfulness / hallucination checker
├── eval/              # RAGAS harness, benchmark sets
└── cache/             # Redis-backed evidence + decision caching
```

---

## References / Further Reading

- Adaptive-RAG (Jeong et al., NAACL 2024) — complexity-based retrieval routing.
- CRAG: Corrective Retrieval Augmented Generation — retrieval evaluator + corrective web search fallback.
- Self-RAG — self-reflective critique tokens for retrieval/generation decisions.
- WeKnow-RAG — adaptive web search + knowledge graph integration.
- RAGAS (`docs.ragas.io`) — for faithfulness/relevance evaluation once you have a working pipeline.
