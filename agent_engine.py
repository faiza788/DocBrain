"""
agent_engine.py
The "agentic" part of DocBrain.

DocumentAgent wraps a RAGEngine with an autonomous ReAct-style loop
(Reason -> Act -> Observe -> repeat) driven by Groq's tool-calling API.
Unlike a fixed search-then-answer pipeline, the LLM itself decides — on
every turn — whether it needs to search at all, what query to use, whether
to refine and search again, or whether it already has enough to answer.

The only tool given to the model is search_documents(query), backed by
RAGEngine.search(). The loop is capped at MAX_STEPS turns as a safety
limit against runaway searching.
"""

import json
import re
from types import SimpleNamespace

from groq import Groq

AGENT_MODEL_NAME = "llama-3.1-8b-instant"
MAX_STEPS = 4     # safety cap on how many search turns the agent gets
TOP_K = 24        # chunks returned per search to capture more of a long document
MAX_CONTEXT_CHARS = 24000

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the uploaded documents for passages relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A clear search query for the documents.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = (
    "You are DocBrain, a document research assistant. You answer questions using ONLY "
    "information found in the user's uploaded documents, via the search_documents tool.\n\n"
    "Rules:\n"
    "- Always search before answering a question about the documents — never answer from "
    "general knowledge or guess.\n"
    "- Before calling search_documents, briefly state in one short sentence why you're "
    "searching or what you're refining.\n"
    "- If your first search doesn't return enough relevant information, refine your query "
    "(different wording, narrower or broader terms, a different angle) and search again.\n"
    "- Don't repeat the exact same query twice — refine it or stop.\n"
    "- Once you have enough information, answer the question directly and concisely, "
    "grounded only in what you found.\n"
    "- Use the retrieved passages as your primary evidence. If the retrieved passages do not "
    "contain enough information, say that clearly rather than filling gaps with assumptions.\n"
    "- If the question asks for a detailed summary or explanation, synthesize the relevant "
    "details from the retrieved passages and answer using only the retrieved passages."
)


class DocumentAgent:
    """Wraps a RAGEngine with an autonomous tool-calling (ReAct) agent loop."""

    def __init__(self, rag_engine, top_k=TOP_K):
        self.rag = rag_engine
        self.top_k = top_k

    def _get_client(self):
        if not self.rag.has_api_key():
            raise RuntimeError("No Groq API key set. Add one via .env or the sidebar first.")
        return Groq(api_key=self.rag.api_key)

    def _run_search(self, query):
        return self.rag.search(query, top_k=self.top_k)

    @staticmethod
    def _is_comprehensive_question(question):
        q = (question or "").lower()
        return any(term in q for term in ["all", "full", "complete", "every", "comprehensive", "module", "modules", "details", "detailed", "summary", "summarize", "list", "page", "pages"])

    def _build_search_queries(self, question):
        base = (question or "").strip()
        if not base:
            return [base]

        if self._is_comprehensive_question(base):
            variants = [
                base,
                f"{base} overview",
                f"{base} key details",
                f"{base} important points",
            ]
            return list(dict.fromkeys([v for v in variants if v]))
        return [base]

    @staticmethod
    def _tool_result_payload(results):
        if not results:
            return json.dumps({"message": "No relevant results found in the indexed documents."})
        return json.dumps(
            [
                {
                    "source": f"{r['source'][0]} (chunk {r['source'][1]})",
                    "relevance_score": round(r["score"], 3),
                    "text": r["text"],
                }
                for r in results
            ]
        )

    def answer_question(self, question):
        """
        Run the agent loop for a question.

        Returns (answer: str, trace: list[dict]). Each trace entry looks like:
            {"step": int, "query": str, "reasoning": str, "results": [...]}
        where "results" is the raw output of RAGEngine.search() for that query —
        useful for the UI to show sources and relevance scores directly.
        """
        client = self._get_client()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace = []

        search_queries = self._build_search_queries(question)
        merged_results = []
        seen_results = set()

        for step, query in enumerate(search_queries, start=1):
            results = self._run_search(query)
            for result in results:
                key = (result["source"][0], result["source"][1], result["text"])
                if key in seen_results:
                    continue
                seen_results.add(key)
                merged_results.append(result)
            trace.append({"step": step, "query": query, "reasoning": "Search the uploaded documents for relevant excerpts.", "results": results})

        if self._is_comprehensive_question(question):
            chunks = getattr(self.rag, "chunks", None)
            sources = getattr(self.rag, "sources", None)
            if chunks:
                for idx, text in enumerate(chunks):
                    source = sources[idx] if sources and idx < len(sources) else ("document", idx)
                    key = (source[0], source[1], text)
                    if key in seen_results:
                        continue
                    seen_results.add(key)
                    merged_results.append({"text": text, "source": source, "score": 1.0})

        merged_results.sort(key=lambda item: item["score"], reverse=True)
        if self._is_comprehensive_question(question):
            merged_results = merged_results[: max(len(merged_results), self.top_k * 3)]
        else:
            merged_results = merged_results[: max(self.top_k * 2, 24)]

        context = "\n\n---\n\n".join(
            f"[Source: {r['source'][0]}, chunk {r['source'][1]}]\n{r['text']}"
            for r in merged_results
        )
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n..."

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"You are answering the user's question using only the retrieved passages below. "
            f"Please cover as many distinct relevant modules, points, or details as the passages contain. "
            f"Do not stop after only a few items when the passages clearly include more. "
            f"If the passages do not contain enough information, say so clearly instead of guessing.\n\n"
            f"Document excerpts:\n{context or 'No relevant excerpts found.'}\n\n"
            f"Question: {question}"
        )

        final_response = client.chat.completions.create(
            model=AGENT_MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=800,
        )
        return (final_response.choices[0].message.content or "").strip(), trace
