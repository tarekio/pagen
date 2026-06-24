"""Unit tests for pagen.llm (fully mocked — never hits a network)."""

from pagen import llm
from pagen.llm import LLMConfig


# ---------------------------------------------------------------------------
# Fake OpenAI client plumbing
# ---------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, content, capture):
        self._content = content
        self._capture = capture

    def create(self, **kwargs):
        self._capture.update(kwargs)
        msg = type("M", (), {"content": self._content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


def _fake_client(content, capture):
    completions = _FakeCompletions(content, capture)
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


# ---------------------------------------------------------------------------
# _api_key
# ---------------------------------------------------------------------------

def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    cfg = LLMConfig(api_key_env="MY_KEY")
    assert cfg._api_key() == "secret"


def test_api_key_fallback_to_ollama(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert LLMConfig()._api_key() == "ollama"


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def test_chat_returns_content_and_passes_no_think(monkeypatch):
    capture = {}
    cfg = LLMConfig(no_think=True)
    monkeypatch.setattr(cfg, "client", lambda: _fake_client("hello", capture))
    out = llm.chat(cfg, [{"role": "user", "content": "hi"}])
    assert out == "hello"
    assert capture["extra_body"] == {"reasoning_effort": "none"}


def test_chat_strips_think_blocks(monkeypatch):
    cfg = LLMConfig()
    monkeypatch.setattr(cfg, "client",
                        lambda: _fake_client("<think>reasoning</think>answer", {}))
    assert llm.chat(cfg, []) == "answer"


def test_chat_none_content_returns_empty(monkeypatch):
    cfg = LLMConfig()
    monkeypatch.setattr(cfg, "client", lambda: _fake_client(None, {}))
    assert llm.chat(cfg, []) == ""


def test_chat_omits_extra_body_when_no_think_false(monkeypatch):
    capture = {}
    cfg = LLMConfig(no_think=False)
    monkeypatch.setattr(cfg, "client", lambda: _fake_client("x", capture))
    llm.chat(cfg, [])
    assert "extra_body" not in capture


def test_try_load_dotenv_no_crash():
    llm._try_load_dotenv()   # must be a no-op-safe call
