# Document Q&A (RAG, powered by Groq — free)

Upload PDFs — contracts, briefs, portfolios, whatever — and ask questions about them.
Retrieval (finding the relevant parts of your documents) runs locally and free.
Answer generation uses the Groq API, which is fast and free for this kind of usage —
no credit card required.

## How it works
1. **Parse** — extracts text from your PDFs (`pypdf`)
2. **Chunk** — splits the text into overlapping ~800-character pieces
3. **Embed** — turns each chunk into a vector locally (`sentence-transformers`, model:
   `all-MiniLM-L6-v2`, ~80MB, downloads once then works offline for this step)
4. **Store** — saves chunks + vectors to `rag_index.pkl` in this folder (your local "vector database")
5. **Ask** — when you type a question, it embeds the question, finds the most similar chunks
   (cosine similarity, no external vector DB needed), and sends them as context to Groq
   (running Llama 3.3 70B on their custom LPU hardware), which reads them and writes a
   grounded answer — typically in 1-2 seconds.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

## Getting a (free) API key

1. Go to https://console.groq.com/keys and sign in with email or Google — no credit card needed.
2. Create an API key.
3. Either:
   - Paste it into the "Groq API key" box in the app's sidebar each time, **or**
   - Set it once as an environment variable so you never have to paste it again:
     ```bash
     export GROQ_API_KEY="your-key-here"     # macOS/Linux
     setx GROQ_API_KEY "your-key-here"        # Windows
     ```
     Then restart the app — it'll pick the key up automatically.

**Note on limits:** Groq's free tier gives you roughly 30 requests/minute and 14,400
requests/day, shared across your account — more than enough for personal document Q&A.
If you ever hit a rate limit, the app will show an error; just wait a minute and try again.

**Note on privacy:** when you ask a question, the relevant document chunks (not the
whole PDF) are sent to Groq's API to generate the answer. If you're working with highly
sensitive material and want zero external calls, a fully local/offline version (no API,
just slower) is still an option — ask and I can bring that back.

## Usage

1. Add your free API key in the sidebar (see above).
2. Upload one or more PDFs and click "Add to knowledge base."
3. Type a question in the main box — e.g. *"What's the payment schedule in this contract?"*
   or *"Summarize the deliverables in the Q3 brief."*
4. Expand "Sources used to answer this" to see exactly which chunks of which file the
   answer came from — useful for double-checking it's not making things up.
5. Use "Clear knowledge base" in the sidebar to wipe everything and start fresh.

## Customizing

- **Model**: change `GROQ_MODEL_NAME` in `rag_engine.py`. Other good free-tier options:
  `llama-3.1-8b-instant` (faster, lighter) or `gpt-oss-120b` (strong all-rounder).
  Full list at https://console.groq.com/docs/models.
- **Chunk size**: tweak `CHUNK_SIZE` / `CHUNK_OVERLAP` in `rag_engine.py` if answers feel
  like they're missing context (bigger chunks) or too vague (smaller chunks).
- **Number of chunks retrieved**: change `top_k` in the `answer_question()` call
  (default 4) — more chunks means more context but a bigger/slower API call.

## Troubleshooting

- **"No Groq API key set"** — add your key in the sidebar, or set the `GROQ_API_KEY`
  environment variable and restart the app.
- **Rate limit / 429 errors** — you've hit the free tier's per-minute or per-day cap;
  wait a bit and try again. This is unlikely under normal personal use.
- **PDF text comes out empty** — some PDFs are scanned images rather than real text —
  this tool doesn't do OCR. You'd need to run those through an OCR tool first.
- **Answers seem to miss something you know is in the PDF** — try increasing `top_k`
  or `CHUNK_SIZE` in `rag_engine.py`, since the answer may be in a chunk that wasn't retrieved.