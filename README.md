# 🔍 VeriRAG — Agentic Research Intelligence & Scientific Claim Verification

### Production-Grade Agentic RAG Platform for Scientific Research

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EKS%20%2B%20ECR-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Pipeline-FF6B35?style=for-the-badge)
![GPT-4o-mini](https://img.shields.io/badge/GPT--4o--mini-Router%20%2B%20Generator-412991?style=for-the-badge&logo=openai&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid%20Retrieval-DC244C?style=for-the-badge)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-1C3C3C?style=for-the-badge)
![DeepEval](https://img.shields.io/badge/DeepEval-100%25%20Faithfulness-4CAF50?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-EKS%20HPA-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 📖 Overview

**VeriRAG** is a **production-grade agentic RAG platform** that enables researchers to upload scientific papers, ask questions grounded in retrieved context, and verify whether specific research claims have been superseded by newer literature.

Users upload papers → VeriRAG routes each query through an **intelligent LangGraph pipeline** — classifying intent, performing hybrid BM25 + dense retrieval with RRF fusion, running dual web searches for claim verification, and streaming answers token-by-token — all deployed on **AWS EKS** with HPA autoscaling, zero-downtime rolling updates, full **LangSmith tracing**, **SlowAPI rate limiting**, and evaluated using **DeepEval** achieving **100% Faithfulness and 100% Answer Relevancy**.

---

## ❌ Problem

- Researchers waste hours manually cross-referencing papers to verify if findings are still current
- Static retrieval pipelines treat all queries the same — factual questions, claim verification, and general knowledge need different workflows
- Single-vector search misses exact terms like author names, arXiv IDs, and formula tokens critical in scientific literature
- No observable, production-grade RAG system purpose-built for scientific research workflows

---

## ✅ Solution

| Problem | Solution |
|---|---|
| Manual claim verification | Agentic verify_claim route — dual Tavily search (web + arXiv) with structured LLM verdict |
| One-size-fits-all retrieval | LangGraph router classifies every query into retrieve / verify_claim / direct_answer |
| Dense-only search misses exact terms | Hybrid BM25 + dense retrieval with Qdrant RRF fusion |
| Poor retrieval quality | Relevancy check node + automatic query rewrite loop |
| No session isolation | Per-session Qdrant collections — zero cross-session data leakage |
| No observability | LangSmith @traceable on all nodes — per-node latency + token costs |
| No abuse protection | SlowAPI rate limiting — 30 req/min per IP |
| Monolithic architecture | FastAPI backend + Streamlit frontend as independent containers |

---

## 🏗️ High-Level Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#1e3a5f', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#4a90d9', 'lineColor': '#4a90d9', 'secondaryColor': '#0d2137', 'tertiaryColor': '#0a1628', 'clusterBkg': '#0d2137', 'clusterBorder': '#4a90d9', 'titleColor': '#ffffff', 'edgeLabelBackground': '#1e3a5f', 'nodeTextColor': '#ffffff'}}}%%
flowchart TD
    subgraph Frontend[Streamlit Frontend — Port 8501]
        UI[Chat Interface]
        SB[Sidebar — Upload PDF / URL / ArXiv]
        SESS[Session Manager — UUID-based isolation]
    end

    subgraph Backend[FastAPI Backend — Port 8000]
        API[REST API — 5 routes]
        RATE[SlowAPI Rate Limiter — 30 req/min]
        GRAPH[LangGraph Pipeline]
    end

    subgraph Pipeline[Agentic LangGraph Pipeline]
        ROUTER[Router Node — GPT-4o-mini\nretrieve / verify_claim / direct_answer]
        AGENT[Agent Node — Tool Caller]
        RETRIEVAL[Hybrid Retrieval — Qdrant BM25 + Dense + RRF]
        WEBSEARCH[Web Search — Tavily]
        RELEVANCY[Relevancy Check Node]
        REWRITE[Query Rewrite Node]
        VERIFY[Claim Verification Node — Dual Tavily]
        GENERATE[Generate Answer Node — Token Streaming]
    end

    subgraph Storage[Storage Layer]
        QDRANT[Qdrant Cloud — Per-session collections]
        SQLITE[SQLite — LangGraph Checkpointer]
        CACHE[Embedding Cache — CacheBackedEmbeddings]
    end

    UI --> API
    SB --> API
    API --> RATE --> GRAPH
    GRAPH --> ROUTER --> AGENT
    AGENT --> RETRIEVAL --> QDRANT
    AGENT --> WEBSEARCH
    RETRIEVAL --> RELEVANCY
    RELEVANCY --> REWRITE --> AGENT
    RELEVANCY --> GENERATE
    ROUTER --> VERIFY --> GENERATE
    ROUTER --> GENERATE
    GRAPH --> SQLITE
    RETRIEVAL --> CACHE

    style UI fill:#1a5276,stroke:#4a90d9,color:#fff
    style SB fill:#1a5276,stroke:#4a90d9,color:#fff
    style SESS fill:#1a5276,stroke:#4a90d9,color:#fff
    style API fill:#145a32,stroke:#27ae60,color:#fff
    style RATE fill:#145a32,stroke:#27ae60,color:#fff
    style GRAPH fill:#145a32,stroke:#27ae60,color:#fff
    style ROUTER fill:#6e2f8a,stroke:#bb8fce,color:#fff
    style AGENT fill:#6e2f8a,stroke:#bb8fce,color:#fff
    style RETRIEVAL fill:#784212,stroke:#f0b27a,color:#fff
    style WEBSEARCH fill:#784212,stroke:#f0b27a,color:#fff
    style RELEVANCY fill:#6e2f8a,stroke:#bb8fce,color:#fff
    style REWRITE fill:#6e2f8a,stroke:#bb8fce,color:#fff
    style VERIFY fill:#922b21,stroke:#f1948a,color:#fff
    style GENERATE fill:#1a5276,stroke:#4a90d9,color:#fff
    style QDRANT fill:#922b21,stroke:#f1948a,color:#fff
    style SQLITE fill:#1b4f72,stroke:#5dade2,color:#fff
    style CACHE fill:#1b4f72,stroke:#5dade2,color:#fff
```

---

## 🔁 End-to-End Request Flow

```mermaid
sequenceDiagram
    participant User
    participant ST as Streamlit :8501
    participant FA as FastAPI :8000
    participant LG as LangGraph Pipeline
    participant QD as Qdrant Cloud
    participant TV as Tavily API
    participant OA as OpenAI API

    User->>ST: Type question + Enter
    ST->>FA: POST /sessions/{id}/query (httpx stream)
    Note over FA: SlowAPI rate limit check (30/min)
    FA->>LG: graph.stream(input_state, stream_mode=messages)
    Note over LG: router_node — GPT-4o-mini classifies query
    LG->>OA: classify → retrieve / verify_claim / direct_answer

    alt retrieve route
        LG->>QD: Hybrid BM25 + dense search (RRF fusion)
        QD-->>LG: top-k chunks
        Note over LG: relevancy_check_node
        opt not relevant
            Note over LG: query_rewrite_node → retry once
        end
    else verify_claim route
        LG->>TV: General web search + arXiv-targeted search
        TV-->>LG: Recent papers + snippets
        LG->>OA: Structured verdict — is_superseded + citations
    else direct_answer route
        Note over LG: No retrieval — answer from model knowledge
    end

    LG->>OA: generate_answer_node (streaming)
    OA-->>FA: token chunks
    FA-->>ST: {type: token, data: chunk}\n (NDJSON stream)
    ST-->>User: Word-by-word rendering
    FA-->>ST: {type: done, data: {answer, route, sources}}
    ST-->>User: Sources expander rendered
```

---

## 🔍 Claim Verification Flow

```mermaid
flowchart TD
    Q2[User: Is this claim still valid?]
    ROUTER2[Router → verify_claim]
    TV1[Tavily General Search\nrecent research superseding: claim]
    TV2[Tavily ArXiv Search\nsite:arxiv.org claim]
    MERGE[Merge results — titles + URLs + 300-char snippets]
    LLM2[GPT-4o-mini Structured Output\nClaimVerificationResult]
    VERDICT{is_superseded?}
    YES[Verdict: OUTDATED\n+ superseding papers with links]
    NO[Verdict: STILL VALID\n+ supporting evidence]

    Q2 --> ROUTER2 --> TV1 & TV2 --> MERGE --> LLM2 --> VERDICT
    VERDICT --> YES
    VERDICT --> NO
```

---

## 📊 Production Results

### DeepEval Evaluation — gpt-4o-mini judge

```
PDF:        Openclaw Research Report (34 pages, 92 chunks)
Goldens:    10 synthesized test cases
Judge:      GPT-4o-mini (gpt-5.4-mini)
Cost:       ~$0.049 per full evaluation run

┌──────────────────────────┬────────┬────────┐
│ Metric                   │ Score  │ Status │
├──────────────────────────┼────────┼────────┤
│ Faithfulness             │ 100%   │  ✅    │
│ Answer Relevancy         │ 100%   │  ✅    │
│ Contextual Precision     │ 100%   │  ✅    │
│ Contextual Recall        │  90%   │  ✅    │
└──────────────────────────┴────────┴────────┘

Note: Contextual Relevancy evaluated at chunk level.
Research PDF chunks contain mixed content (references, tables,
unrelated sections) alongside relevant text. Core RAG quality
metrics — Faithfulness and Answer Relevancy — are both 100%.
```

### Live Test on AWS EKS — Research Paper Q&A

```
Input:     "Give me summary of attention is all you need based on uploaded document"
Route:     retrieve
Chunks:    4 hybrid BM25+dense results
Output:    Structured summary of encoder-decoder attention, multi-head
           attention, and positional encodings
Latency:   ~3.2s end-to-end (including token streaming)
```
![summary](docs/images/summ.png)

### Claim Verification — Live Result

```
Input:     "Are positional encodings from original Transformer still used
            in modern LLMs, or have they been replaced?"
Route:     verify_claim
Searches:  2 Tavily calls (general + arXiv)
Verdict:   OUTDATED — RoPE has replaced sinusoidal positional encodings
Papers:    3 superseding papers with URLs and summaries
Latency:   ~5.1s (dual search + structured generation)
```
![claim](docs/images/claim.png)

---

## ⚡ Infrastructure

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#1e3a5f', 'primaryTextColor': '#ffffff', 'primaryBorderColor': '#4a90d9', 'lineColor': '#4a90d9', 'clusterBkg': '#0d2137', 'clusterBorder': '#4a90d9', 'titleColor': '#ffffff', 'nodeTextColor': '#ffffff'}}}%%
flowchart TB
    subgraph AWS[AWS — us-east-1]
        subgraph EKS[EKS Cluster — Kubernetes 1.32 — Live]
            subgraph NS[verirag namespace]
                FA2[FastAPI Pods\nverirag-backend\nmin:2 max:5 HPA]
                ST2[Streamlit Pod\nverirag-frontend\nreplicas:1]
                HPA3[HPA — CPU 70% target]
                VOL1[PV: verirag_data\nSQLite checkpoints]
                VOL2[PV: verirag_cache\nEmbedding cache]
            end
            ALB[AWS ALB — Internet Facing\nk8s-verirag-veriragi-34154a2410\n-1207122187.us-east-1.elb.amazonaws.com]
        end
        ECR3[ECR — 2 repositories\nverirag-backend + verirag-frontend]
        QDRANT4[Qdrant Cloud\nPer-session collections\nus-east-1]
    end

    ALB --> FA2 & ST2
    ECR3 --> FA2 & ST2
    FA2 --> QDRANT4
    FA2 <--> VOL1
    FA2 <--> VOL2
    HPA3 --> FA2

    style FA2 fill:#145a32,stroke:#27ae60,color:#fff
    style ST2 fill:#1a5276,stroke:#4a90d9,color:#fff
    style VOL1 fill:#1b4f72,stroke:#5dade2,color:#fff
    style VOL2 fill:#1b4f72,stroke:#5dade2,color:#fff
    style ECR3 fill:#784212,stroke:#f0b27a,color:#fff
    style QDRANT4 fill:#922b21,stroke:#f1948a,color:#fff
    style ALB fill:#6e2f8a,stroke:#bb8fce,color:#fff
    style HPA3 fill:#6e2f8a,stroke:#bb8fce,color:#fff
```

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Agent Orchestration | LangGraph StateGraph — conditional routing, cycles, shared RAGState |
| LLM Integration | LangChain ChatOpenAI — GPT-4o-mini |
| LLM Tracing | LangSmith @traceable — all nodes, zero overhead when key not set |
| Dense Embeddings | OpenAI text-embedding-3-small — 1536-dim with CacheBackedEmbeddings |
| Sparse Embeddings | FastEmbed BM25 (Qdrant/bm25) — local, no API key |
| Vector DB | Qdrant Cloud — hybrid collections, RRF server-side fusion |
| Web Search | Tavily API — general + arXiv-targeted dual search |
| Session State | SQLite + LangGraph SqliteSaver checkpointer |
| Rate Limiting | SlowAPI — 30 req/min per IP on /query route |
| Evaluation | DeepEval — 5 metrics, gpt-4o-mini judge, Synthesizer goldens |
| Serving | FastAPI async — NDJSON streaming, Pydantic schemas |
| Frontend | Streamlit — token streaming, session sidebar, sources expander |
| Containerization | Docker — 2 independent containers, named volumes |
| Infrastructure | AWS EKS (Kubernetes 1.32) + ECR + ALB — live deployed |
| Autoscaling | HPA — FastAPI 2-5 pods, CPU 70% target |
| CI/CD | GitHub Actions — build → ECR → rolling deploy to EKS |
| Monitoring | LangSmith — per-node latency, token costs, trace history |

---

## 🔢 Key Numbers — At a Glance

| Metric | Value |
|---|---|
| LangGraph nodes | 7 (router, agent, retrieval, relevancy, rewrite, verify, generate) |
| Embedding dimensions | 1536 (text-embedding-3-small) |
| Hybrid search | BM25 sparse + dense cosine, RRF server-side |
| Session isolation | Per-session Qdrant collection (UUID-keyed) |
| Rate limit | 30 req/min per IP on /query |
| Faithfulness | 100% (DeepEval, gpt-4o-mini judge) |
| Answer Relevancy | 100% (DeepEval, gpt-4o-mini judge) |
| Contextual Recall | 90% (DeepEval) |
| Eval cost | ~$0.049 per full run (10 test cases) |
| Stream protocol | NDJSON — token events + done event |
| Verification searches | 2 Tavily calls per verify_claim query |
| EKS nodes | 2 × t3.medium, us-east-1 |
| FastAPI pods | 2 min → 5 max (HPA) |
| ECR repositories | 2 (verirag-backend, verirag-frontend) |

---

## 🚀 Local Setup

```bash
git clone https://github.com/akashagalave/VeriRAG
cd VeriRAG

# Setup environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Required: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY, TAVILY_API_KEY
# Optional: LANGSMITH_API_KEY (enables LangSmith tracing)

# Run locally — 2 terminals
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload   # Terminal 1
streamlit run app.py                                            # Terminal 2
```

### Docker Compose

```bash
docker compose up --build
# FastAPI  → http://localhost:8000/health
# Swagger  → http://localhost:8000/docs
# Streamlit → http://localhost:8501
```

### Run Evaluation

```bash
python evaluate.py
# Generates goldens.json + eval_results.json
# Prints per-metric scores and pass rates
```

---

## 🔑 System Modes

### Mode 1 — Document Q&A
```
Upload PDF / URL / ArXiv ID → Ask questions → Grounded answers with sources
Route: retrieve
```

### Mode 2 — Claim Verification
```
Ask if a research finding is still current → Verdict with superseding papers
Route: verify_claim
```

### Mode 3 — Direct Answer
```
General knowledge questions — no retrieval cost
Route: direct_answer
```

### Mode 4 — /btw Side Channel
```
Quick question not saved to session history
Usage: /btw what is RoPE?
```

---

## 👨‍💻 Author

**Akash Agalave**
- GitHub: [@akashagalave](https://github.com/akashagalave)
- LinkedIn: [linkedin.com/in/akash-agalave](https://linkedin.com/in/akash-agalave)