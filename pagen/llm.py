"""OpenAI-compatible LLM client abstraction.

Default backend: Ollama at http://localhost:11434/v1 (no real key needed).
Any OpenAI-compatible provider works via --llm-base-url / --api-key-env.

Secrets are never CLI flags — read from the environment variable named by
``api_key_env`` (default: OPENAI_API_KEY). A ``.env`` file in the working
directory is loaded automatically when python-dotenv is installed.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _try_load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b"
    api_key_env: str = "OPENAI_API_KEY"
    no_think: bool = True  # suppress chain-of-thought tokens on reasoning models
    timeout: float = 120.0  # per-request timeout (s); a stalled backend must not hang a run
    max_tokens: int = 1024  # cap output to bound worst-case latency on slow GPUs
    temperature: Optional[float] = None  # None -> use the model/server default

    def _api_key(self) -> str:
        return os.environ.get(self.api_key_env) or "ollama"

    def client(self):
        """Construct an OpenAI client lazily (safe to call inside worker processes)."""
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("ERROR: openai package not installed — run: pip install openai")
        # max_retries=0: we own retry/fallback policy (see text.fill_template).
        return OpenAI(
            base_url=self.base_url,
            api_key=self._api_key(),
            timeout=self.timeout,
            max_retries=0,
        )


def chat(cfg: LLMConfig, messages: list[dict]) -> str:
    """Send a chat request and return the assistant message content string."""
    client = cfg.client()
    kwargs: dict = dict(model=cfg.model, messages=messages, max_tokens=cfg.max_tokens)
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    if cfg.no_think:
        # Disable chain-of-thought on reasoning models. reasoning_effort="none" is
        # the only switch Ollama honors on its OpenAI-compatible /v1 endpoint — the
        # native think:false flag is silently dropped there, and "low" still thinks.
        # Sent via extra_body because "none" is outside the OpenAI SDK's enum.
        kwargs["extra_body"] = {"reasoning_effort": "none"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    content = _THINK_RE.sub("", content).strip()  # safety net for inline <think> blocks
    return content
