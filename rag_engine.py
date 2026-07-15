"""
rag_engine.py
Local retrieval layer for DocBrain.

Handles everything except talking to an LLM: extracting text from PDFs,
chunking it, embedding chunks locally with sentence-transformers (free, no
API key, runs on your machine), and cosine-similarity search over a local
pickle index.

The Groq API key is tracked here (so the UI and the agent share one source
of truth), loaded from a local .env file via python-dotenv, with a fallback
to a key set manually at runtime (e.g. typed into the Streamlit sidebar).
It is kept in memory only for this process — never logged, and never written
anywhere but the user's own .env (which is gitignored). Actual LLM calls
happen in agent_engine.py, not here.
"""

import os
import pickle

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()  # loads GROQ_API_KEY from a local .env file, if present

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small (~80MB), fast, good general-purpose embedding model

CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 150  # overlap between consecutive chunks so context isn't cut off mid-thought


class RAGEngine:
    def __init__(self, index_path="rag_index.pkl", api_key=None):
        self.index_path = index_path
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.chunks = []        # list[str] of chunk text
        self.sources = []       # list[(filename, chunk_index)]
        self.embeddings = None  # numpy array, shape (n_chunks, embedding_dim)

        # API key resolution order: explicit arg > .env (via python-dotenv, loaded above) >
        # manually set later via set_api_key(), since Streamlit collects it from the sidebar
        # after the engine already exists.
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")

        if os.path.exists(index_path):
            self.load_index()

    @property
    def api_key(self):
        return self._api_key

    def set_api_key(self, api_key):
        """Update the API key at runtime (e.g. after the user types it into the sidebar)."""
        self._api_key = api_key

    def has_api_key(self):
        return bool(self._api_key)

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
