"""
rag_engine.py
Core logic for a RAG (Retrieval-Augmented Generation) document Q&A system.

Pipeline:
  1. Extract text from PDFs (pypdf)
  2. Split text into overlapping chunks
  3. Embed chunks locally with sentence-transformers (free, no API key, runs on your machine)
  4. Store embeddings + chunk text in a simple pickle file (acts as our vector store)
  5. On a question: embed the query, find the most similar chunks (cosine similarity),
     and feed them as context to Groq (fast, free-tier LLM inference) to generate an answer.

Retrieval (steps 1-4) stays fully local and free. Answer generation (step 5) calls the
Groq API, which is free for this kind of usage (no credit card needed) — see README.
"""

import os
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"        # small (~80MB), fast, good general-purpose embedding model
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # fast + accurate, well within Groq's free tier

CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 150  # overlap between consecutive chunks so context isn't cut off mid-thought


class RAGEngine:
    def __init__(self, index_path="rag_index.pkl", api_key=None):
        self.index_path = index_path
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.chunks = []        # list[str] of chunk text
        self.sources = []       # list[(filename, chunk_index)]
        self.embeddings = None  # numpy array, shape (n_chunks, embedding_dim)

        # API key resolution order: explicit arg > env var. Can also be set later via
        # set_api_key(), since Streamlit collects it from the user after the engine exists.
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = None

        if os.path.exists(index_path):
            self.load_index()

    def set_api_key(self, api_key):
        """Update the API key at runtime (e.g. after the user types it into the sidebar)."""
        self._api_key = api_key
        self._client = None  # force client to be rebuilt with the new key

    def has_api_key(self):
        return bool(self._api_key)

    # ---------- Groq client (lazy loaded) ----------

    def _get_client(self):
        if not self._api_key:
            raise RuntimeError("No Groq API key set. Add one in the sidebar first.")
        if self._client is None:
            self._client = Groq(api_key=self._api_key)
        return self._client

    # ---------- PDF handling ----------

    @staticmethod
    def extract_text_from_pdf(file_path):
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts)

    @staticmethod
    def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        text = " ".join(text.split())  # collapse whitespace/newlines
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return [c.strip() for c in chunks if c.strip()]

    # ---------- Index management ----------

    def add_document(self, file_path, filename):
        """Extract, chunk, embed, and add a PDF to the knowledge base. Returns number of chunks added."""
        text = self.extract_text_from_pdf(file_path)
        new_chunks = self.chunk_text(text)
        if not new_chunks:
            return 0

        new_embeddings = self.embedder.encode(new_chunks, show_progress_bar=False)

        if self.embeddings is None:
            self.embeddings = np.array(new_embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.chunks.extend(new_chunks)
        self.sources.extend([(filename, i) for i in range(len(new_chunks))])
        self.save_index()
        return len(new_chunks)

    def save_index(self):
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {"chunks": self.chunks, "sources": self.sources, "embeddings": self.embeddings},
                f,
            )

    def load_index(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.sources = data["sources"]
        self.embeddings = data["embeddings"]

    def clear_index(self):
        self.chunks = []
        self.sources = []
        self.embeddings = None
        if os.path.exists(self.index_path):
            os.remove(self.index_path)

    # ---------- Retrieval ----------

    def search(self, query, top_k=4):
        """Return the top_k most relevant chunks for a query, using cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_emb = self.embedder.encode([query])[0]
        doc_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        denom = doc_norms * query_norm
        denom[denom == 0] = 1e-10

        sims = (self.embeddings @ query_emb) / denom
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(sims[i])}
            for i in top_idx
        ]

    # ---------- Generation ----------

    def answer_question(self, query, top_k=4):
        results = self.search(query, top_k=top_k)
        if not results:
            return "No documents indexed yet — upload and add a PDF first.", []

        context = "\n\n---\n\n".join(
            f"[Source: {r['source'][0]}, chunk {r['source'][1]}]\n{r['text']}" for r in results
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't contained in the context, say you don't know — don't make anything up.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()

        return answer, results