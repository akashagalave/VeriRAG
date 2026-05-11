
import logging
import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1536          # text-embedding-3-small output size
DENSE_VECTOR_NAME  = "dense"  # named vector key stored in Qdrant
SPARSE_VECTOR_NAME = "sparse" # named sparse vector key stored in Qdrant

# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_file_store = LocalFileStore("./embedding_cache/")

# Disk-backed embedding cache — avoids re-embedding the same chunk twice.
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

# BM25 sparse encoder via FastEmbed — no API key, runs locally.
# First call downloads the model (~50 MB); subsequent calls are cached.
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection helpers ────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    """Map a session UUID to a Qdrant collection name (hyphens → underscores)."""
    return f"papeer_{session_id.replace('-', '_')}"


def _collection_is_hybrid(collection_name: str) -> bool:
    """
    Return True if the collection was created with sparse vectors.
    Used to detect legacy dense-only collections and keep them working.
    """
    try:
        info = qdrant_client.get_collection(collection_name)
        sparse_cfg = info.config.params.sparse_vectors_config
        return bool(sparse_cfg)
    except Exception:
        return False


def _ensure_collection(collection_name: str) -> bool:
    """
    Create a hybrid (dense + sparse) collection if it does not yet exist.
    Returns True if a new collection was created, False if it already existed.
    """
    if qdrant_client.collection_exists(collection_name):
        return False

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info("Created hybrid collection: %s", collection_name)
    return True


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    """
    Return a QdrantVectorStore for the session.

    • New collections  → HYBRID mode (dense + BM25 + RRF)
    • Legacy dense-only collections → DENSE mode (backward compat)
    """
    collection_name = get_collection_name(session_id)
    _ensure_collection(collection_name)

    if _collection_is_hybrid(collection_name):
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    # Legacy path — keeps old sessions working without re-ingestion
    logger.warning(
        "Collection %s is dense-only (legacy). "
        "Re-ingest documents to enable hybrid retrieval.",
        collection_name,
    )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    """
    Embed and store document chunks in the session's Qdrant collection.
    For hybrid collections, both dense and sparse vectors are generated
    and stored in a single upsert call.
    """
    vs = get_vectorstore(session_id)
    vs.add_documents(docs)
    logger.info(
        "Stored %d chunks for session %s (collection: %s)",
        len(docs),
        session_id,
        get_collection_name(session_id),
    )


def list_papers(session_id: str) -> list[str]:
    """
    Return a deduplicated list of document titles loaded into this session.
    Scrolls the full collection in pages of 100.
    """
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break

    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """
    Hybrid similarity search (BM25 + dense + RRF).
    Falls back to dense-only for legacy collections.

    Args:
        query:      Natural-language or keyword query string.
        session_id: Session UUID — determines which Qdrant collection to search.
        k:          Number of top results to return (post-RRF).

    Returns:
        List of LangChain Document objects ranked by RRF score.
    """
    return get_vectorstore(session_id).similarity_search(query, k=k)


def collection_stats(session_id: str) -> dict:
    """
    Return collection metadata useful for the FastAPI /sessions/{id}/info endpoint.
    Includes chunk count and whether hybrid retrieval is active.
    """
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return {"exists": False, "chunk_count": 0, "hybrid": False}

    info = qdrant_client.get_collection(collection_name)
    return {
        "exists": True,
        "chunk_count": info.points_count,
        "hybrid": _collection_is_hybrid(collection_name),
        "collection_name": collection_name,
    }