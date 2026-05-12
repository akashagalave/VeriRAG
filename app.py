"""
app.py
------
Streamlit frontend for VeriRAG.

What this file does:
  - Session sidebar (new chat, switch sessions, auto-naming)
  - Document upload (PDF / URL / ArXiv) via FastAPI
  - Chat interface with token streaming
  - Sources expander per turn (retrieved chunks)
  - /btw side channel (local, not saved to history)

All AI logic lives in backend/api.py (FastAPI).
This file only handles UI and calls the API via httpx.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

import os

st.set_page_config(page_title="VeriRAG", page_icon="📚", layout="centered")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SESSIONS_FILE = Path("sessions.json")

# Used only for generating session names — pure UI logic
_rename_llm = ChatOpenAI(model="gpt-4o-mini")


# ── HTTP client ───────────────────────────────────────────────────────────────

def _client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND_URL, timeout=120.0)


# ── Session helpers ───────────────────────────────────────────────────────────

def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke([
            {
                "role": "system",
                "content": (
                    "Generate a concise 3-5 word title for a research chat session "
                    "based on the user's first message. Return only the title, "
                    "no punctuation at the end, no quotes."
                ),
            },
            {"role": "user", "content": first_message[:500]},
        ])
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def load_session_chats(session_id: str) -> list[dict]:
    """Reload past messages from FastAPI backend when switching sessions."""
    try:
        with _client() as client:
            resp = client.get(f"/sessions/{session_id}/history")
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(
            1 for m in st.session_state.chats[session_id]
            if m["role"] == "assistant"
        )
        st.session_state.turns[session_id] = turn_count


# ── Bootstrap ─────────────────────────────────────────────────────────────────

if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("+ New Chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    st.divider()
    st.markdown("## 💬 Sessions")

    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type=btn_type,
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()

    st.divider()
    st.markdown("## 📄 Documents")

    # ── File upload ────────────────────────────────────────────────────────────
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"uploader_{active_sid}",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            if processed_key not in st.session_state:
                st.session_state[processed_key] = set()
            with st.spinner("Processing files…"):
                for f in uploaded_files:
                    if f.name in st.session_state[processed_key]:
                        st.info(f"Already loaded: {f.name}")
                        continue
                    try:
                        with _client() as client:
                            resp = client.post(
                                f"/sessions/{active_sid}/ingest",
                                files={"file": (f.name, f.read(), f.type)},
                            )
                        resp.raise_for_status()
                        data = resp.json()
                        st.session_state[processed_key].add(f.name)
                        st.success(f"Added: {f.name} ({data['chunks_added']} chunks)")
                    except httpx.HTTPStatusError as e:
                        st.error(f"Failed: {f.name} — {e.response.text}")
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
            st.rerun()
        else:
            st.warning("No files selected.")

    # ── Web URL loader ─────────────────────────────────────────────────────────
    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key=f"url_area_{active_sid}",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        if urls:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        with _client() as client:
                            resp = client.post(
                                f"/sessions/{active_sid}/ingest",
                                data={"url": url},
                            )
                        resp.raise_for_status()
                        data = resp.json()
                        st.success(f"Loaded: {url[:60]} ({data['chunks_added']} chunks)")
                    except httpx.HTTPStatusError as e:
                        st.error(f"Failed: {url[:60]} — {e.response.text}")
                    except Exception as e:
                        st.error(f"Failed: {url[:60]} — {e}")
            st.rerun()
        else:
            st.warning("Enter at least one URL.")

    # ── ArXiv loader ───────────────────────────────────────────────────────────
    st.markdown("**ArXiv Papers**")
    arxiv_title = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_input_{active_sid}",
        label_visibility="collapsed",
        placeholder="1706.03762  or  Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if arxiv_title.strip():
            with st.spinner("Loading from ArXiv…"):
                try:
                    with _client() as client:
                        resp = client.post(
                            f"/sessions/{active_sid}/ingest",
                            data={"arxiv_query": arxiv_title.strip()},
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    st.success(f"Loaded ({data['chunks_added']} chunks)")
                except httpx.HTTPStatusError as e:
                    st.error(f"Failed: {e.response.text}")
                except Exception as e:
                    st.error(f"Failed: {e}")
            st.rerun()
        else:
            st.warning("Enter a paper title or ArXiv ID.")

    # ── Loaded Documents list ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Loaded Documents")
    try:
        with _client() as client:
            info_resp = client.get(f"/sessions/{active_sid}/info")
        info_resp.raise_for_status()
        info = info_resp.json()
        doc_titles = info.get("paper_titles", [])
        hybrid_on = info.get("hybrid", False)
    except Exception:
        doc_titles = None
        hybrid_on = False

    if doc_titles is None:
        st.caption("Could not load document list — is the backend running?")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
        if hybrid_on:
            st.caption("🔀 Hybrid retrieval active (BM25 + dense)")
    else:
        st.caption("No documents loaded yet.")


# ── Page header ───────────────────────────────────────────────────────────────

st.title("📚 VeriRAG — Agentic Research Intelligence")
st.markdown(
    "🔍 **Ask questions** from your uploaded papers &nbsp;·&nbsp; "
    "✅ **Verify claims** against recent literature &nbsp;·&nbsp; "
    "🌐 **Search the web** for the latest findings\n\n"
    "> Upload documents in the sidebar and start chatting below."
)
st.divider()


# ── Chat display — render existing messages ───────────────────────────────────

for msg in st.session_state.chats.get(active_sid, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Sources expander — only for assistant messages that have sources
        if msg["role"] == "assistant":
            sources = msg.get("sources") or []
            if sources:
                with st.expander(
                    f"📄 Sources · {len(sources)} chunks"
                    f" · route: {msg.get('route') or '—'}",
                    expanded=False,
                ):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        label = (
                            meta.get("title") or meta.get("url") or f"chunk {i}"
                        )
                        st.markdown(f"**{i}. {label}**")
                        st.caption(src["content"][:300])


# ── /btw side channel ─────────────────────────────────────────────────────────

def _handle_btw_locally(query: str):
    from backend.btw_handler import handle_btw
    yield from handle_btw(query)


# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    # ── /btw path ─────────────────────────────────────────────────────────────
    if is_btw:
        query = prompt.strip()[4:].strip()
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption("Side channel — not saved to session history.")
        with st.chat_message("assistant"):
            if not query:
                st.markdown(
                    "Please add a question after `/btw`, e.g. `/btw What is attention?`"
                )
            else:
                placeholder = st.empty()
                response_text = ""
                for chunk in _handle_btw_locally(query):
                    response_text += chunk
                    placeholder.markdown(response_text + "▌")
                placeholder.markdown(response_text)
            st.caption("Side channel — not saved to session history.")

    # ── Normal RAG path ───────────────────────────────────────────────────────
    else:
        if active_sid not in st.session_state.chats:
            st.session_state.chats[active_sid] = []
        if active_sid not in st.session_state.turns:
            st.session_state.turns[active_sid] = 0

        is_first_message = len(st.session_state.chats[active_sid]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.turns[active_sid] += 1

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response_text = ""
            done_data: dict = {}
            error_msg: str | None = None

            try:
                with httpx.Client(base_url=BACKEND_URL, timeout=120.0) as client:
                    with client.stream(
                        "POST",
                        f"/sessions/{active_sid}/query",
                        json={"question": prompt},
                    ) as stream_resp:
                        stream_resp.raise_for_status()
                        for raw_line in stream_resp.iter_lines():
                            if not raw_line.strip():
                                continue
                            event = json.loads(raw_line)
                            etype = event.get("type")

                            if etype == "token":
                                response_text += event["data"]
                                placeholder.markdown(response_text + "▌")

                            elif etype == "done":
                                done_data = event["data"]
                                # direct_answer / verify_claim may not stream tokens
                                # fall back to full answer from done event
                                if not response_text:
                                    response_text = done_data.get("answer", "")

                            elif etype == "error":
                                error_msg = event["data"]

            except httpx.HTTPStatusError as e:
                error_msg = (
                    f"Backend error ({e.response.status_code}): {e.response.text}"
                )
            except httpx.ConnectError:
                error_msg = (
                    f"Cannot reach backend at `{BACKEND_URL}`. "
                    "Is the FastAPI server running?"
                )

            # Render final answer
            if error_msg:
                placeholder.error(error_msg)
                response_text = error_msg
                done_data = {}
            else:
                placeholder.markdown(response_text)

            sources = done_data.get("sources") or []
            route = done_data.get("route")

            # Sources expander — show where the answer came from
            if sources:
                with st.expander(
                    f"📄 Sources · {len(sources)} chunks · route: {route or '—'}",
                    expanded=False,
                ):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        label = (
                            meta.get("title") or meta.get("url") or f"chunk {i}"
                        )
                        st.markdown(f"**{i}. {label}**")
                        st.caption(src["content"][:300])

        # Save to in-memory chat history
        st.session_state.chats[active_sid].append({
            "role": "assistant",
            "content": response_text,
            "sources": sources,
            "route": route,
        })

        if is_first_message:
            st.rerun()