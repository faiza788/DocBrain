# 🧠 DocBrain — Agentic Document Q&A

Upload PDFs and ask questions about them. Answers come from an autonomous LLM
agent that decides **for itself** when to search your documents, what to
search for, and whether to refine its query and search again — not from a
fixed pipeline. Built for an Agentic AI course as a demonstration of
tool-using, self-directing agents.

## Why this is "agentic" and not just RAG

A standard RAG app is a fixed pipeline with no decision-making in it:

```
question -> embed -> search once -> stuff results into a prompt -> generate answer
```

It always searches exactly once, with the user's raw question, no matter
whether that search actually found anything useful. If the first search
misses, the app has no way to notice or recover — it just answers badly.

DocBrain replaces that fixed pipeline with an autonomous agent loop
(**ReAct-style: Reason → Act → Observe → repeat**), implemented in
`agent_engine.py` via Groq's tool-calling API:

1. **Reason** — the LLM reads the question and decides whether it needs to
   search the documents at all, and if so, what query would find the answer.
2. **Act** — it calls the `search_documents(query)` tool with a query of its
   own choosing (not necessarily the user's exact wording).
3. **Observe** — it reads the returned chunks and their relevance scores.
4. **Repeat or answer** — if the results don't actually answer the question,
   it can rephrase and search again with a refined query. Once it has enough,
   it answers directly — grounded only in what it found.

The model itself is making these calls, turn by turn, up to a small safety
cap (`MAX_STEPS = 4` in `agent_engine.py`). The UI's "Agent's reasoning"
section shows this whole trace, so you can see exactly what it searched for,
why, and what it found — which is the difference between "an app that does
RAG" and "an agent that uses a search tool."

Retrieval and embeddings run **fully locally** (`sentence-transformers`, no
API key, no external calls). Only the final agent reasoning/answer generation
calls the Groq API.

## Project structure

| File               | Purpose                                                             |
|--------------------|----------------------------------------------------------------------|
| `app.py`           | Streamlit UI — upload PDFs, ask questions, view the agent's trace   |
| `rag_engine.py`    | `RAGEngine` — PDF parsing, chunking, local embeddings, index, search |
| `agent_engine.py`  | `DocumentAgent` — the ReAct tool-calling loop around `RAGEngine`     |
| `requirements.txt` | Python dependencies                                                  |
| `.env.example`     | Template for your `.env` — copy it, it is *not* your real key       |

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Groq API key
#    -> https://console.groq.com/keys  (sign in with email or Google, no credit card)

# 4. Create your .env file from the template, then edit it with your real key
cp .env.example .env          # on Windows: copy .env.example .env
```

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=your-key-here
```

`.env` is listed in `.gitignore` — it will never be committed. `.env.example`
has no real key in it and is safe to commit as-is.

## Running it

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`. If `GROQ_API_KEY` is set
in `.env`, the sidebar will show "Loaded from .env" automatically — no key
entry needed. If it's not set, you can paste a key into the sidebar as a
fallback; it's kept in memory for that session only and never written to disk.

## Usage

1. Upload one or more PDFs in the sidebar and click **"Add to knowledge base"**.
   The sidebar shows the indexed files and total chunk count.
2. Type a question in the main box — e.g. *"What's the payment schedule in
   this contract?"* or *"Summarize the deliverables across all the uploaded
   briefs."*
3. Read the **Answer**.
4. Expand **"Agent's reasoning"** to see the full trace: each search the
   agent ran, why it ran it (when the model states a reason), the chunks it
   got back, and their relevance scores (0–1, higher = more relevant). A
   question that needed two refined searches will show two steps; a question
   the agent could answer without searching shows zero.
5. Use **"Clear knowledge base"** in the sidebar to wipe the local index and
   start fresh.

## Customizing

- **Model**: change `AGENT_MODEL_NAME` in `agent_engine.py`. Full list of
  Groq-hosted models at https://console.groq.com/docs/models.
- **Search step cap**: change `MAX_STEPS` in `agent_engine.py` if you want
  the agent to be able to search more (or fewer) times before being forced
  to answer.
- **Chunk size**: tweak `CHUNK_SIZE` / `CHUNK_OVERLAP` in `rag_engine.py` if
  answers feel like they're missing context (bigger chunks) or too vague
  (smaller chunks).
- **Chunks per search**: change `TOP_K` in `agent_engine.py` (default 4).

## Troubleshooting

- **"No Groq API key set"** — add `GROQ_API_KEY` to your `.env` file and
  restart the app, or paste a key into the sidebar.
- **Sidebar never shows "Loaded from .env"** — make sure the file is named
  exactly `.env` (not `.env.txt`) and sits in the same folder as `app.py`,
  and that you restarted `streamlit run app.py` after creating it (env vars
  are read on startup).
- **Rate limit / 429 errors** — you've hit Groq's free-tier per-minute or
  per-day cap; wait a bit and try again.
- **PDF text comes out empty / "0 chunks added"** — some PDFs are scanned
  images rather than real text; this tool doesn't do OCR. Run those through
  an OCR tool first.
- **Agent's reasoning shows repeated near-identical searches** — this is
  the model refining its query, not a bug; if it happens excessively, lower
  `MAX_STEPS` in `agent_engine.py`.
- **Answers seem to miss something you know is in the PDF** — try increasing
  `TOP_K` or `CHUNK_SIZE`, or ask a more specific question so the agent's
  search query lands on the right chunk.

## Privacy note

When the agent searches, the relevant document chunks (not the whole PDF)
are sent to Groq's API as part of generating the answer. `rag_index.pkl`
(your local document index) and `.env` (your API key) are both gitignored
and stay on your machine only.
