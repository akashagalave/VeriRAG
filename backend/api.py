from dotenv import load_dotenv
load_dotenv()  

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph
from backend.vector_store import add_paper, collection_stats, list_papers

logger = logging.getLogger(__name__)


_rate = os.getenv("RATE_LIMIT_PER_MINUTE", "30")
limiter = Limiter(key_func=get_remote_address)



_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logger.info("Building LangGraph…")
    _graph = build_graph(db_path=os.getenv("CHECKPOINT_DB", "checkpoints.db"))
    logger.info("LangGraph ready.")
    yield
    logger.info("VeriRAG FastAPI shutdown.")


def get_graph():
    if _graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not initialised yet — retry in a moment.",
        )
    return _graph



app = FastAPI(
    title="VeriRAG API",
    description=(
        "**VeriRAG — Agentic Research Intelligence & Scientific Claim Verification Platform**\n\n"
        "The core RAG pipeline (LangGraph + Qdrant hybrid retrieval) is exposed "
        "here as a REST service. Streamlit is just one UI on top.\n\n"
        "**Features:**\n"
        "- Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion\n"
        "- Agentic query routing (retrieve / verify_claim / direct_answer)\n"
        "- Scientific claim verification against recent arXiv literature\n"
        "- Web search via Tavily for live supplementary context\n\n"
        "All traces, per-node latency, and token counts are visible in "
        "[LangSmith](https://smith.langchain.com) under project **VeriRAG**."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question, claim to verify, or general query.",
        examples=["What methodology did the authors use?"],
    )


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[list] = None
    route: Optional[str] = None


class IngestResponse(BaseModel):
    session_id: str
    message: str
    chunks_added: int


class SessionInfoResponse(BaseModel):
    session_id: str
    exists: bool
    chunk_count: int
    hybrid: bool
    collection_name: Optional[str] = None
    paper_titles: list[str]



@app.get(
    "/health",
    tags=["ops"],
    summary="Liveness probe",
    description="Returns 200 immediately. Used by k8s liveness probe and ALB health check.",
)
async def health():
    return {"status": "ok"}


@app.get(
    "/ready",
    tags=["ops"],
    summary="Readiness probe",
    description="Returns 200 only after the LangGraph has been compiled and is ready to serve.",
)
async def ready():
    if _graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not ready yet.",
        )
    return {"status": "ready"}


#Session info 

@app.get(
    "/sessions/{session_id}/info",
    response_model=SessionInfoResponse,
    tags=["sessions"],
    summary="Collection stats for a session",
    description="Returns chunk count, hybrid-retrieval status, and loaded paper titles.",
)
async def session_info(session_id: str):
    stats = collection_stats(session_id)
    titles = list_papers(session_id) if stats["exists"] else []
    return SessionInfoResponse(
        session_id=session_id,
        paper_titles=titles,
        **stats,
    )


#Session history

@app.get(
    "/sessions/{session_id}/history",
    response_model=list[ChatMessage],
    tags=["sessions"],
    summary="Reload chat history for a session",
    description="Reads the SQLite checkpointer and returns all messages for this session.",
)
async def session_history(session_id: str):
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
    except Exception:
        return []

    messages = state.values.get("messages", [])
    chats: list[ChatMessage] = []

    for msg in messages:
        type_name = type(msg).__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if type_name == "HumanMessage":
            chats.append(ChatMessage(role="user", content=content))

        elif type_name in ("AIMessage", "AIMessageChunk"):
            if not content.strip():
                continue
            chats.append(ChatMessage(
                role="assistant",
                content=content,
                sources=[],
                route=state.values.get("route"),
            ))

    return chats


#Ingest 

@app.post(
    "/sessions/{session_id}/ingest",
    response_model=IngestResponse,
    tags=["sessions"],
    summary="Ingest a document into a session",
    description=(
        "Accepts a file upload (PDF / TXT / MD), a URL, or an arXiv ID / title. "
        "Exactly one of `file`, `url`, or `arxiv_query` must be provided."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def ingest(
    session_id: str,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
    arxiv_query: Optional[str] = Form(default=None),
):
    provided = sum([file is not None, bool(url), bool(arxiv_query)])
    if provided == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of: file, url, arxiv_query.",
        )
    if provided > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only one of file / url / arxiv_query may be supplied per request.",
        )

    try:
        if file:
            suffix = Path(file.filename).suffix
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await file.read())
                    tmp_path = tmp.name
                docs = load_document(tmp_path)
                for doc in docs:
                    doc.metadata.setdefault("title", Path(file.filename).stem)
            finally:
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

        elif url:
            docs = load_webpage(url.strip())

        else:
            docs = load_arxiv(arxiv_query.strip())

        add_paper(docs, session_id)
        return IngestResponse(
            session_id=session_id,
            message="Documents ingested successfully.",
            chunks_added=len(docs),
        )

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingest failed for session %s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingest failed: {exc}",
        )


@app.post(
    "/sessions/{session_id}/query",
    tags=["query"],
    summary="Run the RAG pipeline (streaming)",
    description=(
        "Streams the answer token-by-token from the generate_answer node.\n\n"
        "**Stream protocol** — each line is a newline-delimited JSON object:\n\n"
        "```\n"
        "{\"type\": \"token\", \"data\": \"text chunk\"}\n"
        "{\"type\": \"done\",  \"data\": {\"answer\": \"...\", "
        "\"route\": \"retrieve\", \"sources\": [...]}}\n"
        "{\"type\": \"error\", \"data\": \"error message\"}\n"
        "```\n\n"
        "Rate-limited to RATE_LIMIT_PER_MINUTE requests per IP (default 30/min)."
    ),
    response_class=StreamingResponse,
)
@limiter.limit(f"{_rate}/minute")
async def query_session(
    request: Request,  
    session_id: str,
    body: QueryRequest,
):
    graph = get_graph()

    async def _stream() -> AsyncIterator[str]:
        config = {"configurable": {"thread_id": session_id}}
        input_state = {
            "messages": [HumanMessage(content=body.question)],
            "session_id": session_id,
            "query": body.question,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }

        try:
            
            for chunk, metadata in graph.stream(
                input_state, config, stream_mode="messages"
            ):
                if (
                    metadata.get("langgraph_node") == "generate_answer"
                    and hasattr(chunk, "content")
                    and chunk.content
                ):
                    yield json.dumps({"type": "token", "data": chunk.content}) + "\n"

            final_values = graph.get_state(config).values
            retrieved_docs = final_values.get("retrieved_docs") or []
            sources = [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in retrieved_docs
            ]

            yield json.dumps({
                "type": "done",
                "data": {
                    "answer": final_values.get("answer") or "",
                    "route": final_values.get("route"),
                    "sources": sources,
                },
            }) + "\n"

        except Exception as exc:
            logger.exception("Stream failed for session %s", session_id)
            yield json.dumps({"type": "error", "data": str(exc)}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")