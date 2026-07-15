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

from groq import Groq

AGENT_MODEL_NAME = "llama-3.3-70b-versatile"
MAX_STEPS = 4     # safety cap on how many search turns the agent gets
TOP_K = 4         # chunks returned per search

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the user's uploaded documents for passages relevant to a query. "
            "Returns the most relevant chunks, each with a relevance score from 0 to 1 "
            "(higher is more relevant) and the source file/chunk it came from. "
            "Call this whenever you need information from the documents to answer the "
            "question. If the results aren't useful, you may call it again with a "
            "refined or differently-worded query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Can be the user's question verbatim on the "
                        "first search, or a refined/narrower/rephrased query on later ones."
                    ),
                }
            },
            "required": ["query"],
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
    "- If, after searching, the documents genuinely don't contain the answer, say so "
    "honestly instead of making something up."
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

        for step in range(1, MAX_STEPS + 1):
            response = client.chat.completions.create(
                model=AGENT_MODEL_NAME,
                messages=messages,
                tools=[SEARCH_TOOL],
                tool_choice="auto",
                parallel_tool_calls=False,
                max_tokens=800,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                return (message.content or "").strip(), trace

            reasoning = (message.content or "").strip()

            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                query = args.get("query") or question

                results = self._run_search(query)
                trace.append({"step": step, "query": query, "reasoning": reasoning, "results": results})

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._tool_result_payload(results),
                    }
                )

        # Safety cap reached — force a final answer from whatever was gathered so far,
        # without offering the search tool again.
        messages.append(
            {
                "role": "user",
                "content": (
                    "You've reached the search limit. Answer the original question now, using "
                    "only the information you've already found. If it's not enough to answer "
                    "fully, say so honestly."
                ),
            }
        )
        final_response = client.chat.completions.create(
            model=AGENT_MODEL_NAME,
            messages=messages,
            max_tokens=800,
        )
        return (final_response.choices[0].message.content or "").strip(), trace
