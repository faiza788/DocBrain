"""
app.py — Streamlit UI for DocBrain, an agentic document Q&A tool.

Unlike a fixed retrieve-then-generate pipeline, questions here are answered by
an autonomous agent (agent_engine.DocumentAgent): it decides for itself when to
search the indexed documents, what to search for, and whether to refine its
query and search again — via Groq tool calling. Retrieval/embeddings run
locally (sentence-transformers); only the final answer generation calls Groq.

Run with:
    streamlit run app.py

Needs a free Groq API key — see README for setup. Loaded automatically from a
local .env file, or pasted into the sidebar as a session-only fallback.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from agent_engine import DocumentAgent
from rag_engine import RAGEngine

load_dotenv()  # loads GROQ_API_KEY from a local .env file, if present

st.set_page_config(page_title="DocBrain — Agentic Document Q&A", page_icon="🧠", layout="wide")

st.title("🧠 DocBrain")
st.caption(
    "Upload PDFs and ask questions about them. An LLM agent decides when to search your "
    "documents, refines its own queries if needed, and answers based on what it finds — "
    "retrieval runs locally, only the final answer generation calls Groq."
)


@st.cache_resource
def get_engine():
    return RAGEngine()


@st.cache_resource
def get_agent(_engine):
    # Leading underscore tells Streamlit not to try to hash the RAGEngine instance.
    return DocumentAgent(_engine)


engine = get_engine()
agent = get_agent(engine)

with st.sidebar:
    st.header(" Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("➕ Add to knowledge base", use_container_width=True):
            with st.spinner("Reading and indexing PDFs..."):
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        n_chunks = engine.add_document(tmp_path, f.name)
                        st.success(f"Added **{f.name}** ({n_chunks} chunks)")
                    finally:
                        os.remove(tmp_path)

    st.divider()
    st.metric("Indexed chunks", len(engine.chunks))

    indexed_files = sorted({s[0] for s in engine.sources})
    if indexed_files:
        st.write("**Files in knowledge base:**")
        for fn in indexed_files:
            st.write(f"- {fn}")
    else:
        st.write("_No documents indexed yet._")

    st.divider()
    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        engine.clear_index()
        st.rerun()

st.divider()

query = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g. What's the payment schedule in the contract?",
)

if query:
    if len(engine.chunks) == 0:
        st.warning("Upload and add at least one PDF first (sidebar, left).")
    else:
        with st.spinner("Agent is researching..."):
            try:
                answer, trace = agent.answer_question(query)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                answer, trace = None, []

        if answer:
            st.subheader("Answer")
            st.write(answer)

            n = len(trace)
            label = f"🧠 Agent's reasoning ({n} search{'es' if n != 1 else ''})" if n else "🧠 Agent's reasoning"
            with st.expander(label):
                if not trace:
                    st.caption("The agent answered directly, without searching the documents.")
                for step in trace:
                    st.markdown(f"**Step {step['step']} — searched:** `{step['query']}`")
                    if step.get("reasoning"):
                        st.caption(f"Why: {step['reasoning']}")

                    if not step["results"]:
                        st.caption("No relevant results found.")
                    else:
                        for r in step["results"]:
                            fn, idx = r["source"]
                            st.markdown(f"- **{fn}** (chunk {idx}) — relevance {r['score']:.2f}")
                            preview = r["text"][:300] + ("..." if len(r["text"]) > 300 else "")
                            st.caption(preview)
                    st.divider()
