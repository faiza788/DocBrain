from types import SimpleNamespace

from agent_engine import DocumentAgent


class FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer"))])


class FakeGroq:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=FakeCompletions())


class FakeRAG:
    def __init__(self):
        self.api_key = "fake-key"
        self.queries = []

    def has_api_key(self):
        return True

    def search(self, query, top_k=4):
        self.queries.append(query)
        return [
            {"text": "The contract says payment is due in 30 days.", "source": ("contract.pdf", 1), "score": 0.91},
        ]


def test_answer_question_uses_grounded_retrieval_prompt(monkeypatch):
    fake_rag = FakeRAG()
    agent = DocumentAgent(fake_rag, top_k=2)

    fake_client = FakeGroq("fake-key")
    monkeypatch.setattr("agent_engine.Groq", lambda api_key: fake_client)

    answer, trace = agent.answer_question("When is payment due?")

    assert answer == "Grounded answer"
    assert trace[0]["query"] == "When is payment due?"
    assert "answer using only the retrieved passages" in fake_client.chat.completions.last_kwargs["messages"][1]["content"].lower()
    assert "if the retrieved passages do not contain enough information" in fake_client.chat.completions.last_kwargs["messages"][1]["content"].lower()


def test_document_agent_uses_a_wider_retrieval_window():
    agent = DocumentAgent(FakeRAG())

    assert agent.top_k >= 12


def test_comprehensive_questions_trigger_a_broader_retrieval_pass(monkeypatch):
    fake_rag = FakeRAG()
    agent = DocumentAgent(fake_rag, top_k=2)

    fake_client = FakeGroq("fake-key")
    monkeypatch.setattr("agent_engine.Groq", lambda api_key: fake_client)

    agent.answer_question("Give me all modules and full details")

    assert len(fake_rag.queries) >= 2
    assert any("overview" in query.lower() for query in fake_rag.queries)


def test_comprehensive_questions_use_all_indexed_chunks(monkeypatch):
    fake_rag = FakeRAG()
    fake_rag.chunks = ["Module 1 details", "Module 2 details"]
    fake_rag.sources = [("doc.pdf", 0), ("doc.pdf", 1)]
    agent = DocumentAgent(fake_rag, top_k=2)

    fake_client = FakeGroq("fake-key")
    monkeypatch.setattr("agent_engine.Groq", lambda api_key: fake_client)

    agent.answer_question("Give me all pages and all modules")

    prompt = fake_client.chat.completions.last_kwargs["messages"][1]["content"]
    assert "Module 1 details" in prompt
    assert "Module 2 details" in prompt


def test_comprehensive_questions_limit_context_passages(monkeypatch):
    fake_rag = FakeRAG()
    fake_rag.chunks = [f"module {i} details" for i in range(20)]
    fake_rag.sources = [("doc.pdf", i) for i in range(20)]
    agent = DocumentAgent(fake_rag, top_k=2)

    fake_client = FakeGroq("fake-key")
    monkeypatch.setattr("agent_engine.Groq", lambda api_key: fake_client)

    agent.answer_question("Give me all pages and all modules")

    prompt = fake_client.chat.completions.last_kwargs["messages"][1]["content"]
    assert prompt.count("[Source:") <= 10
